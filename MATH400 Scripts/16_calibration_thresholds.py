from __future__ import annotations

import argparse
import ast
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from src.modelling_common import (
    CHECKPOINTS,
    TARGET_COLUMN,
    IDENTIFIER_COLUMN,
    get_feature_columns,
    load_checkpoint_data,
)


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42

FORWARD_FOLDS = [
    {
        "fold": 1,
        "training_semesters": ["2024 S1"],
        "validation_semester": "2024 S2",
    },
    {
        "fold": 2,
        "training_semesters": [
            "2024 S1",
            "2024 S2",
        ],
        "validation_semester": "2025 S1",
    },
]

MODEL_NAMES = [
    "logistic_regression",
    "random_forest",
    "xgboost",
]


# ============================================================
# Logging and directories
# ============================================================

def create_directories(output_root: Path) -> None:
    for folder in [
        "predictions",
        "tables",
        "logs",
    ]:
        (output_root / folder).mkdir(
            parents=True,
            exist_ok=True,
        )


def configure_logging(output_root: Path) -> None:
    logging.basicConfig(
        filename=(
            output_root
            / "logs"
            / "oof_prediction_execution.log"
        ),
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


# ============================================================
# Preprocessing
# ============================================================

def build_preprocessor(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    X = dataframe[feature_columns]

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = [
        column
        for column in feature_columns
        if column not in numeric_features
    ]

    numeric_steps = [
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
            ),
        )
    ]

    if scale_numeric:
        numeric_steps.append(
            (
                "scaler",
                StandardScaler(),
            )
        )

    numeric_pipeline = Pipeline(
        steps=numeric_steps
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )


# ============================================================
# Parameter loading
# ============================================================

def parse_parameters(value) -> dict:
    if isinstance(value, dict):
        return value

    if pd.isna(value):
        return {}

    text = str(value)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def load_best_parameters(
    output_root: Path,
    model: str,
    checkpoint: str,
) -> dict:
    filename_map = {
        "logistic_regression": (
            "logistic_regression_"
            "best_parameters_all_checkpoints.csv"
        ),
        "random_forest": (
            "random_forest_"
            "best_parameters_all_checkpoints.csv"
        ),
        "xgboost": (
            "xgboost_"
            "best_parameters_all_checkpoints.csv"
        ),
    }

    file_path = (
        output_root
        / "tables"
        / filename_map[model]
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Best-parameter file not found:\n{file_path}"
        )

    table = pd.read_csv(file_path)

    rows = table[
        table["checkpoint"].astype(str).str.lower()
        == checkpoint.lower()
    ]

    if rows.empty:
        raise ValueError(
            f"No best parameters found for "
            f"{model}, {checkpoint} in {file_path}"
        )

    row = rows.iloc[0]

    # Logistic Regression stores parameters in columns.
    if model == "logistic_regression":
        class_weight = row.get(
            "class_weight",
            None,
        )

        if pd.isna(class_weight):
            class_weight = None
        elif str(class_weight).lower() == "none":
            class_weight = None
        else:
            class_weight = str(class_weight)

        return {
            "C": float(row["C"]),
            "class_weight": class_weight,
        }

    # RF and XGBoost store JSON in best_parameters.
    if "best_parameters" not in row:
        raise ValueError(
            f"'best_parameters' column not found in {file_path}"
        )

    return parse_parameters(
        row["best_parameters"]
    )


# ============================================================
# Model builders
# ============================================================

def build_logistic_pipeline(
    training_dataframe: pd.DataFrame,
    feature_columns: list[str],
    parameters: dict,
    seed: int,
) -> Pipeline:
    preprocessor = build_preprocessor(
        training_dataframe,
        feature_columns,
        scale_numeric=True,
    )

    model = LogisticRegression(
        C=float(parameters["C"]),
        class_weight=parameters.get(
            "class_weight"
        ),
        max_iter=5000,
        random_state=seed,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def build_random_forest_pipeline(
    training_dataframe: pd.DataFrame,
    feature_columns: list[str],
    parameters: dict,
    seed: int,
) -> Pipeline:
    preprocessor = build_preprocessor(
        training_dataframe,
        feature_columns,
        scale_numeric=False,
    )

    cleaned_parameters = {}

    for key, value in parameters.items():
        cleaned_key = key.replace(
            "model__",
            "",
        )
        cleaned_parameters[cleaned_key] = value

    model = RandomForestClassifier(
        random_state=seed,
        n_jobs=-1,
        **cleaned_parameters,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def build_xgboost_pipeline(
    training_dataframe: pd.DataFrame,
    feature_columns: list[str],
    parameters: dict,
    seed: int,
    scale_pos_weight: float,
) -> Pipeline:
    preprocessor = build_preprocessor(
        training_dataframe,
        feature_columns,
        scale_numeric=False,
    )

    cleaned_parameters = {}

    for key, value in parameters.items():
        cleaned_key = key.replace(
            "model__",
            "",
        )
        cleaned_parameters[cleaned_key] = value

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        **cleaned_parameters,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


# ============================================================
# Identifier handling
# ============================================================

def get_identifier_series(
    dataframe: pd.DataFrame,
) -> pd.Series:
    if IDENTIFIER_COLUMN in dataframe.columns:
        return dataframe[
            IDENTIFIER_COLUMN
        ].reset_index(drop=True)

    # Fallback only if the source file has no identifier.
    logging.warning(
        "Identifier column '%s' not found; "
        "using original row position.",
        IDENTIFIER_COLUMN,
    )

    return pd.Series(
        np.arange(len(dataframe)),
        name=IDENTIFIER_COLUMN,
    )


# ============================================================
# OOF generation
# ============================================================

def create_oof_predictions(
    data_root: Path,
    output_root: Path,
    seed: int,
) -> pd.DataFrame:
    all_predictions = []

    for checkpoint in CHECKPOINTS:
        logging.info(
            "Starting OOF generation for %s",
            checkpoint,
        )

        development, _, _ = load_checkpoint_data(
            data_root,
            checkpoint,
        )

        feature_columns = get_feature_columns(
            development
        )

        for model_name in MODEL_NAMES:
            parameters = load_best_parameters(
                output_root,
                model_name,
                checkpoint,
            )

            logging.info(
                "%s %s parameters: %s",
                checkpoint,
                model_name,
                parameters,
            )

            model_predictions = []

            for fold_definition in FORWARD_FOLDS:
                fold_number = fold_definition["fold"]
                training_semesters = (
                    fold_definition[
                        "training_semesters"
                    ]
                )
                validation_semester = (
                    fold_definition[
                        "validation_semester"
                    ]
                )

                fold_training = development[
                    development["semester"].isin(
                        training_semesters
                    )
                ].copy()

                fold_validation = development[
                    development["semester"]
                    == validation_semester
                ].copy()

                X_train = fold_training[
                    feature_columns
                ]

                y_train = fold_training[
                    TARGET_COLUMN
                ]

                X_validation = fold_validation[
                    feature_columns
                ]

                if model_name == "logistic_regression":
                    pipeline = (
                        build_logistic_pipeline(
                            fold_training,
                            feature_columns,
                            parameters,
                            seed,
                        )
                    )

                elif model_name == "random_forest":
                    pipeline = (
                        build_random_forest_pipeline(
                            fold_training,
                            feature_columns,
                            parameters,
                            seed,
                        )
                    )

                elif model_name == "xgboost":
                    negative_cases = int(
                        (y_train == 0).sum()
                    )

                    positive_cases = int(
                        (y_train == 1).sum()
                    )

                    scale_pos_weight = (
                        negative_cases
                        / positive_cases
                        if positive_cases > 0
                        else 1.0
                    )

                    pipeline = (
                        build_xgboost_pipeline(
                            fold_training,
                            feature_columns,
                            parameters,
                            seed,
                            scale_pos_weight,
                        )
                    )

                else:
                    raise ValueError(
                        f"Unknown model: {model_name}"
                    )

                pipeline.fit(
                    X_train,
                    y_train,
                )

                probabilities = (
                    pipeline.predict_proba(
                        X_validation
                    )[:, 1]
                )

                identifiers = (
                    get_identifier_series(
                        fold_validation
                    )
                )

                fold_output = pd.DataFrame(
                    {
                        "unique_identifier": (
                            identifiers.values
                        ),
                        "checkpoint": checkpoint,
                        "model_name": model_name,
                        "model_version": (
                            "day5_oof_frozen_v1"
                        ),
                        "fold": fold_number,
                        "training_semesters": "|".join(
                            training_semesters
                        ),
                        "validation_semester": (
                            validation_semester
                        ),
                        "row_order_within_fold": np.arange(
                            len(fold_validation)
                        ),
                        "true_label": fold_validation[
                            TARGET_COLUMN
                        ].to_numpy(),
                        "predicted_probability": (
                            probabilities
                        ),
                    }
                )

                model_predictions.append(
                    fold_output
                )
                all_predictions.append(
                    fold_output
                )

                logging.info(
                    "Completed %s %s fold %d: "
                    "%d predictions",
                    checkpoint,
                    model_name,
                    fold_number,
                    len(fold_output),
                )

            model_table = pd.concat(
                model_predictions,
                ignore_index=True,
            )

            model_table.to_csv(
                output_root
                / "predictions"
                / f"{checkpoint}_{model_name}_oof.csv",
                index=False,
            )

    combined_table = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    combined_table.to_csv(
        output_root
        / "predictions"
        / "all_models_development_oof_predictions.csv",
        index=False,
    )

    return combined_table


# ============================================================
# OOF summary
# ============================================================

def create_oof_summary(
    oof_table: pd.DataFrame,
    output_root: Path,
) -> None:
    rows = []

    for (
        checkpoint,
        model_name,
    ), group in oof_table.groupby(
        ["checkpoint", "model_name"]
    ):
        rows.append(
            {
                "checkpoint": checkpoint,
                "model_name": model_name,
                "number_predictions": len(group),
                "number_positive": int(
                    group["true_label"].sum()
                ),
                "positive_prevalence": (
                    group["true_label"].mean()
                ),
                "number_folds": group[
                    "fold"
                ].nunique(),
                "validation_semesters": "|".join(
                    sorted(
                        group[
                            "validation_semester"
                        ].unique()
                    )
                ),
            }
        )

    pd.DataFrame(rows).to_csv(
        output_root
        / "tables"
        / "development_oof_summary.csv",
        index=False,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate leakage-safe development "
            "out-of-fold predictions for LR, RF and XGBoost."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
    )

    args = parser.parse_args()

    create_directories(args.output_root)
    configure_logging(args.output_root)

    logging.info(
        "Starting development OOF prediction generation"
    )
    logging.info(
        "Threshold is not applied; probabilities only"
    )
    logging.info(
        "Random seed: %d",
        args.seed,
    )

    oof_table = create_oof_predictions(
        args.data_root,
        args.output_root,
        args.seed,
    )

    create_oof_summary(
        oof_table,
        args.output_root,
    )

    print("\nOOF prediction generation completed.")
    print(
        "Combined OOF predictions:",
        args.output_root
        / "predictions"
        / "all_models_development_oof_predictions.csv",
    )
    print(
        "OOF summary:",
        args.output_root
        / "tables"
        / "development_oof_summary.csv",
    )


if __name__ == "__main__":
    main()