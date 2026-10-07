#python 11_build_preprocessing.py --data-root "..\Data" --output-root "outputs"
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.modelling_common import (
    CHECKPOINTS,
    build_preprocessor,
    get_feature_columns,
    load_checkpoint_data,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create and document unfitted leakage-safe "
            "preprocessing templates for each checkpoint."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to the project Data folder.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Path to the project outputs folder.",
    )

    args = parser.parse_args()

    preprocessing_dir = (
        args.output_root / "preprocessing"
    )

    tables_dir = args.output_root / "tables"

    preprocessing_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tables_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_feature_rows = []

    for checkpoint in CHECKPOINTS:
        development, validation, testing = (
            load_checkpoint_data(
                args.data_root,
                checkpoint,
            )
        )

        feature_columns = get_feature_columns(
            development
        )

        # This template is intentionally NOT fitted here.
        # Each model training script fits it only on the
        # relevant training fold or final development data.
        preprocessor = build_preprocessor(
            development,
            feature_columns,
            scale_numeric=True,
        )

        X = development[feature_columns]

        numeric_features = X.select_dtypes(
            include=["number"]
        ).columns.tolist()

        categorical_features = [
            column
            for column in feature_columns
            if column not in numeric_features
        ]

        schema = {
            "checkpoint": checkpoint,
            "development_semesters": [
                "2024 S1",
                "2024 S2",
                "2025 S1",
            ],
            "validation_semester": "2025 S2",
            "testing_semester": "2026 S1",
            "number_of_features": len(feature_columns),
            "feature_columns": feature_columns,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "excluded_columns": [
                "at_risk",
                "unique_identifier",
                "semester",
                "source_file",
            ],
            "numeric_preprocessing": [
                "median imputation",
                "missing-value indicators",
                "standardisation for Logistic Regression",
            ],
            "categorical_preprocessing": [
                "most-frequent imputation",
                "one-hot encoding",
                "handle_unknown='ignore'",
            ],
            "leakage_control": (
                "This is an unfitted preprocessing template. "
                "It is fitted only within the model pipeline "
                "on each training fold or final development set."
            ),
        }

        # Save the schema in readable JSON format.
        schema_path = (
            preprocessing_dir
            / f"{checkpoint}_preprocessing_schema.json"
        )

        with open(
            schema_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                schema,
                file,
                indent=2,
            )

        # Save the unfitted template for documentation/reuse.
        # Do not use this saved template directly for holdout fitting.
        template_path = (
            preprocessing_dir
            / f"{checkpoint}_preprocessor_template.joblib"
        )

        joblib.dump(
            preprocessor,
            template_path,
        )

        for feature in numeric_features:
            all_feature_rows.append(
                {
                    "checkpoint": checkpoint,
                    "feature": feature,
                    "feature_type": "numeric",
                    "preprocessing": (
                        "median imputation + missing indicator "
                        "+ standardisation for Logistic Regression"
                    ),
                }
            )

        for feature in categorical_features:
            all_feature_rows.append(
                {
                    "checkpoint": checkpoint,
                    "feature": feature,
                    "feature_type": "categorical",
                    "preprocessing": (
                        "most-frequent imputation + one-hot encoding "
                        "+ handle_unknown='ignore'"
                    ),
                }
            )

        print(
            f"{checkpoint}: created unfitted preprocessing template "
            f"with {len(feature_columns)} features "
            f"({len(numeric_features)} numeric, "
            f"{len(categorical_features)} categorical)."
        )

    feature_table = pd.DataFrame(
        all_feature_rows
    )

    feature_table.to_csv(
        tables_dir
        / "preprocessing_feature_summary.csv",
        index=False,
    )

    print("\nPreprocessing documentation completed.")
    print("Schemas:", preprocessing_dir)
    print(
        "Feature summary:",
        tables_dir
        / "preprocessing_feature_summary.csv",
    )


if __name__ == "__main__":
    main()