from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.modelling_common import (
    CHECKPOINTS,
    IDENTIFIER_COLUMN,
    TARGET,
    get_feature_columns,
    load_checkpoint_data,
)


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "logistic_regression"
MODEL_LABEL = "Logistic Regression"
HOLDOUT_SEMESTER = "2026 S1"
RANDOM_SEED = 42
TOP_N_FEATURES = 20
TOP_N_LOCAL_FEATURES = 10
BORDERLINE_DISTANCE = 0.05


# ============================================================
# Directories
# ============================================================

def create_directories(output_root: Path) -> None:
    for folder in [
        "figures",
        "tables",
        "predictions",
        "logs",
        "shap",
    ]:
        (output_root / folder).mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# Utilities
# ============================================================

def safe_filename(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text[:150]


def friendly_label(feature_name: str) -> str:
    """
    Conservative automatic translation.

    Review the generated feature dictionary before including it
    in the dissertation or educator-facing material.
    """

    text = str(feature_name)

    # Remove sklearn transformer prefixes.
    text = re.sub(
        r"^(numeric|categorical)__",
        "",
        text,
    )

    text = text.replace(
        "_missing",
        " missing indicator",
    )

    text = text.replace(
        "_imputer",
        "",
    )

    text = text.replace(
        "onehot__",
        "",
    )

    text = text.replace(
        "numeric__",
        "",
    )

    text = text.replace(
        "categorical__",
        "",
    )

    text = text.replace(
        "_",
        " ",
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    replacements = {
        "at risk": "At-risk outcome",
        "unique identifier": "Student record identifier",
        "attendance": "Attendance",
        "engagement": "Engagement",
        "login": "Learning-platform logins",
        "session": "Learning-platform sessions",
        "grade": "Academic grade",
        "gpa": "Grade point average",
        "assessment": "Assessment activity",
        "submission": "Assignment submission activity",
        "course": "Course information",
        "week": "Week",
    }

    lower_text = text.lower()

    for old, new in replacements.items():
        if old in lower_text:
            return new + f" ({text})"

    return text[:1].upper() + text[1:]


def get_identifier_series(
    dataframe: pd.DataFrame,
) -> pd.Series:
    if IDENTIFIER_COLUMN in dataframe.columns:
        return dataframe[
            IDENTIFIER_COLUMN
        ].reset_index(drop=True)

    return pd.Series(
        np.arange(len(dataframe)),
        name=IDENTIFIER_COLUMN,
    )


def locate_model(
    output_root: Path,
    checkpoint: str,
) -> Path:
    candidates = [
        output_root
        / "models"
        / f"{checkpoint}_logistic_regression.joblib",
        output_root
        / "models"
        / f"{checkpoint}_logistic.joblib",
        output_root
        / "models"
        / checkpoint
        / "logistic_regression.joblib",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find Logistic Regression model for "
        f"{checkpoint}. Checked:\n"
        + "\n".join(str(path) for path in candidates)
    )


def load_selected_thresholds(
    output_root: Path,
) -> dict[str, float]:
    file_path = (
        output_root
        / "tables"
        / "selected_operational_thresholds.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Selected-threshold file not found:\n{file_path}"
        )

    dataframe = pd.read_csv(file_path)

    dataframe = dataframe[
        dataframe["model_name"]
        == MODEL_NAME
    ]

    if dataframe.empty:
        raise ValueError(
            "No Logistic Regression thresholds found."
        )

    return dict(
        zip(
            dataframe["checkpoint"],
            dataframe["selected_threshold"].astype(float),
        )
    )


# ============================================================
# SHAP preparation
# ============================================================

def transform_pipeline_data(
    pipeline,
    dataframe: pd.DataFrame,
    feature_columns: list[str],
):
    X = dataframe[feature_columns]

    preprocessor = pipeline.named_steps["preprocess"]

    transformed = preprocessor.transform(X)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    return (
        np.asarray(transformed),
        np.asarray(feature_names),
    )


def calculate_shap_values(
    pipeline,
    background_raw: pd.DataFrame,
    holdout_raw: pd.DataFrame,
    feature_columns: list[str],
):
    background_transformed, feature_names = (
        transform_pipeline_data(
            pipeline,
            background_raw,
            feature_columns,
        )
    )

    holdout_transformed, _ = (
        transform_pipeline_data(
            pipeline,
            holdout_raw,
            feature_columns,
        )
    )

    model = pipeline.named_steps["model"]

    # The preprocessor is already fitted by the saved pipeline.
    # LinearExplainer explains the fitted linear model in the
    # transformed feature space.
    explainer = shap.LinearExplainer(
        model,
        background_transformed,
    )

    shap_result = explainer(
        holdout_transformed
    )

    # Binary Logistic Regression usually returns an
    # Explanation object with one value per observation/feature.
    shap_values = np.asarray(
        shap_result.values
    )

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    if shap_values.ndim != 2:
        raise ValueError(
            "Unexpected SHAP-value shape: "
            f"{shap_values.shape}"
        )

    expected_value = explainer.expected_value

    if np.ndim(expected_value) > 0:
        expected_value = np.asarray(
            expected_value
        ).ravel()[-1]

    return (
        shap_values,
        holdout_transformed,
        feature_names,
        float(expected_value),
    )


# ============================================================
# Global explanations
# ============================================================

def save_global_importance(
    shap_values: np.ndarray,
    feature_names: np.ndarray,
    output_root: Path,
    checkpoint: str,
) -> pd.DataFrame:
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_absolute_shap": np.mean(
                np.abs(shap_values),
                axis=0,
            ),
            "mean_shap": np.mean(
                shap_values,
                axis=0,
            ),
        }
    )

    importance[
        "absolute_mean_shap"
    ] = np.abs(importance["mean_shap"])

    importance["educator_friendly_label"] = (
        importance["feature"].map(
            friendly_label
        )
    )

    importance = importance.sort_values(
        "mean_absolute_shap",
        ascending=False,
    )

    importance["rank"] = np.arange(
        1,
        len(importance) + 1,
    )

    importance.to_csv(
        output_root
        / "tables"
        / f"{checkpoint}_logistic_shap_global_importance.csv",
        index=False,
    )

    return importance


def plot_global_bar(
    importance: pd.DataFrame,
    output_root: Path,
    checkpoint: str,
) -> None:
    subset = importance.head(
        TOP_N_FEATURES
    ).sort_values(
        "mean_absolute_shap",
        ascending=True,
    )

    figure, axis = plt.subplots(
        figsize=(10, 8)
    )

    axis.barh(
        subset["educator_friendly_label"],
        subset["mean_absolute_shap"],
        color="#4472C4",
    )

    axis.set_xlabel(
        "Mean absolute SHAP value"
    )

    axis.set_title(
        f"{MODEL_LABEL}: global SHAP importance — "
        f"{checkpoint.title()}"
    )

    axis.grid(
        axis="x",
        alpha=0.3,
    )

    figure.tight_layout()

    figure.savefig(
        output_root
        / "figures"
        / f"{checkpoint}_logistic_shap_global_bar.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_global_beeswarm(
    shap_values: np.ndarray,
    transformed_values: np.ndarray,
    feature_names: np.ndarray,
    output_root: Path,
    checkpoint: str,
) -> None:
    explanation = shap.Explanation(
        values=shap_values,
        base_values=np.repeat(
            0.0,
            shap_values.shape[0],
        ),
        data=transformed_values,
        feature_names=feature_names,
    )

    shap.plots.beeswarm(
        explanation,
        max_display=TOP_N_FEATURES,
        show=False,
    )

    figure = plt.gcf()

    figure.suptitle(
        f"{MODEL_LABEL}: SHAP beeswarm — "
        f"{checkpoint.title()}",
        y=1.02,
    )

    figure.tight_layout()

    figure.savefig(
        output_root
        / "figures"
        / f"{checkpoint}_logistic_shap_beeswarm.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_dependence_plots(
    shap_values: np.ndarray,
    transformed_values: np.ndarray,
    feature_names: np.ndarray,
    importance: pd.DataFrame,
    output_root: Path,
    checkpoint: str,
) -> None:
    """
    Creates dependence plots for important transformed features
    that have numeric variation.

    One-hot features are excluded where possible.
    """

    selected_features = []

    for feature in importance["feature"]:
        if feature not in feature_names:
            continue

        index = list(feature_names).index(feature)
        values = transformed_values[:, index]

        if (
            "onehot" not in feature
            and np.unique(values).size >= 5
        ):
            selected_features.append(feature)

        if len(selected_features) >= 3:
            break

    for feature in selected_features:
        shap.dependence_plot(
            feature,
            shap_values,
            transformed_values,
            feature_names=feature_names,
            interaction_index=None,
            show=False,
        )

        figure = plt.gcf()

        figure.suptitle(
            f"SHAP dependence: {friendly_label(feature)} — "
            f"{checkpoint.title()}",
            y=1.02,
        )

        figure.tight_layout()

        figure.savefig(
            output_root
            / "figures"
            / (
                f"{checkpoint}_logistic_shap_dependence_"
                f"{safe_filename(feature)}.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(figure)


# ============================================================
# Local explanations
# ============================================================

def choose_prototypes(
    holdout_raw: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    dataframe = holdout_raw.copy().reset_index(
        drop=True
    )

    dataframe["predicted_probability"] = (
        probabilities
    )

    dataframe["selected_threshold"] = threshold

    dataframe["predicted_class"] = (
        probabilities >= threshold
    ).astype(int)

    dataframe["error_group"] = np.select(
        [
            (
                dataframe[TARGET] == 1
            )
            & (
                dataframe["predicted_class"] == 1
            ),
            (
                dataframe[TARGET] == 0
            )
            & (
                dataframe["predicted_class"] == 1
            ),
            (
                dataframe[TARGET] == 1
            )
            & (
                dataframe["predicted_class"] == 0
            ),
            (
                dataframe[TARGET] == 0
            )
            & (
                dataframe["predicted_class"] == 0
            ),
        ],
        [
            "true_positive",
            "false_positive",
            "false_negative",
            "true_negative",
        ],
        default="unknown",
    )

    selected_rows = []

    # One representative row from each available error group.
    for group_name in [
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
    ]:
        group = dataframe[
            dataframe["error_group"]
            == group_name
        ]

        if not group.empty:
            # Select the row closest to the group's
            # median probability.
            median_probability = group[
                "predicted_probability"
            ].median()

            index = (
                (group["predicted_probability"]
                 - median_probability)
                .abs()
                .idxmin()
            )

            selected_rows.append(
                dataframe.loc[index]
            )

    # Borderline case closest to the selected threshold.
    dataframe["threshold_distance"] = (
        dataframe["predicted_probability"]
        - threshold
    ).abs()

    if not dataframe.empty:
        index = dataframe[
            "threshold_distance"
        ].idxmin()

        borderline = dataframe.loc[index].copy()
        borderline["error_group"] = (
            "borderline_probability"
        )
        selected_rows.append(borderline)

    if not selected_rows:
        return pd.DataFrame()

    prototypes = pd.DataFrame(
        selected_rows
    ).drop_duplicates(
        subset=[IDENTIFIER_COLUMN]
        if IDENTIFIER_COLUMN in dataframe.columns
        else None
    )

    prototypes["prototype_number"] = np.arange(
        1,
        len(prototypes) + 1,
    )

    return prototypes


def create_local_explanation(
    prototype_row: pd.Series,
    shap_row: np.ndarray,
    feature_names: np.ndarray,
    expected_value: float,
    checkpoint: str,
    output_root: Path,
) -> tuple[pd.DataFrame, str]:
    contribution_table = pd.DataFrame(
        {
            "feature": feature_names,
            "educator_friendly_label": [
                friendly_label(name)
                for name in feature_names
            ],
            "shap_value": shap_row,
            "absolute_shap_value": np.abs(
                shap_row
            ),
        }
    ).sort_values(
        "absolute_shap_value",
        ascending=False,
    )

    contribution_table["direction"] = np.where(
        contribution_table["shap_value"] >= 0,
        "increases predicted risk",
        "decreases predicted risk",
    )

    contribution_table["prototype_group"] = (
        prototype_row["error_group"]
    )

    contribution_table["predicted_probability"] = (
        prototype_row["predicted_probability"]
    )

    contribution_table["selected_threshold"] = (
        prototype_row["selected_threshold"]
    )

    contribution_table.to_csv(
        output_root
        / "tables"
        / (
            f"{checkpoint}_prototype_"
            f"{prototype_row['prototype_number']}_"
            f"contributions.csv"
        ),
        index=False,
    )

    waterfall_explanation = shap.Explanation(
        values=shap_row,
        base_values=expected_value,
        data=np.zeros(len(feature_names)),
        feature_names=[
            friendly_label(name)
            for name in feature_names
        ],
    )

    shap.plots.waterfall(
        waterfall_explanation,
        max_display=TOP_N_LOCAL_FEATURES,
        show=False,
    )

    figure = plt.gcf()

    figure.suptitle(
        f"{checkpoint.title()} — "
        f"{prototype_row['error_group']} — "
        f"risk={prototype_row['predicted_probability']:.3f}",
        y=1.02,
    )

    figure.text(
        0.01,
        0.01,
        "Associative explanation only; requires human review.",
        fontsize=9,
    )

    figure.tight_layout()

    figure.savefig(
        output_root
        / "figures"
        / (
            f"{checkpoint}_prototype_"
            f"{prototype_row['prototype_number']}_"
            f"waterfall.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return (
        contribution_table,
        str(
            output_root
            / "figures"
            / (
                f"{checkpoint}_prototype_"
                f"{prototype_row['prototype_number']}_"
                f"waterfall.png"
            )
        ),
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

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

    args = parser.parse_args()

    create_directories(args.output_root)

    threshold_map = load_selected_thresholds(
        args.output_root
    )

    all_feature_dictionary_rows = []
    all_local_summary_rows = []

    for checkpoint in CHECKPOINTS:
        print(
            f"Generating SHAP explanations for "
            f"{checkpoint}..."
        )

        development, _, holdout = (
            load_checkpoint_data(
                args.data_root,
                checkpoint,
            )
        )

        feature_columns = get_feature_columns(
            development
        )

        model_path = locate_model(
            args.output_root,
            checkpoint,
        )

        pipeline = joblib.load(
            model_path
        )

        # Development data is the reference/background data.
        background_raw = development[
            feature_columns
        ].copy()

        holdout_raw = holdout.copy()

        shap_values, transformed_holdout, feature_names, expected_value = (
            calculate_shap_values(
                pipeline,
                background_raw,
                holdout_raw,
                feature_columns,
            )
        )

        probabilities = pipeline.predict_proba(
            holdout_raw[feature_columns]
        )[:, 1]

        threshold = threshold_map.get(
            checkpoint,
            0.40,
        )

        # Global importance.
        importance = save_global_importance(
            shap_values,
            feature_names,
            args.output_root,
            checkpoint,
        )

        plot_global_bar(
            importance,
            args.output_root,
            checkpoint,
        )

        plot_global_beeswarm(
            shap_values,
            transformed_holdout,
            feature_names,
            args.output_root,
            checkpoint,
        )

        plot_dependence_plots(
            shap_values,
            transformed_holdout,
            feature_names,
            importance,
            args.output_root,
            checkpoint,
        )

        # Technical-to-educator feature dictionary.
        for feature in feature_names:
            all_feature_dictionary_rows.append(
                {
                    "checkpoint": checkpoint,
                    "technical_feature": feature,
                    "educator_friendly_label": (
                        friendly_label(feature)
                    ),
                    "interpretation_note": (
                        "Review against the feature catalogue "
                        "before educator-facing use."
                    ),
                }
            )

        # Local prototypes.
        prototypes = choose_prototypes(
            holdout_raw,
            probabilities,
            threshold,
        )

        identifiers = get_identifier_series(
            holdout_raw
        )

        prototypes["unique_identifier"] = (
            identifiers.loc[
                prototypes.index
            ].to_numpy()
        )

        for _, prototype in prototypes.iterrows():
            prototype_index = prototype.name

            contribution_table, figure_path = (
                create_local_explanation(
                    prototype,
                    shap_values[
                        prototype_index
                    ],
                    feature_names,
                    expected_value,
                    checkpoint,
                    args.output_root,
                )
            )

            contribution_table = (
                contribution_table.head(
                    TOP_N_LOCAL_FEATURES
                )
            )

            increasing = contribution_table[
                contribution_table["shap_value"] > 0
            ].head(5)

            decreasing = contribution_table[
                contribution_table["shap_value"] < 0
            ].head(5)

            all_local_summary_rows.append(
                {
                    "checkpoint": checkpoint,
                    "prototype_number": (
                        prototype[
                            "prototype_number"
                        ]
                    ),
                    "unique_identifier": (
                        prototype[
                            "unique_identifier"
                        ]
                    ),
                    "prototype_group": (
                        prototype["error_group"]
                    ),
                    "predicted_risk_probability": (
                        prototype[
                            "predicted_probability"
                        ]
                    ),
                    "selected_threshold": threshold,
                    "predicted_class": int(
                        prototype[
                            "predicted_class"
                        ]
                    ),
                    "actual_historical_outcome": int(
                        prototype[TARGET]
                    ),
                    "top_factors_increasing_risk": (
                        "; ".join(
                            increasing[
                                "educator_friendly_label"
                            ].tolist()
                        )
                    ),
                    "top_factors_decreasing_risk": (
                        "; ".join(
                            decreasing[
                                "educator_friendly_label"
                            ].tolist()
                        )
                    ),
                    "explanation_caution": (
                        "Associative explanation only; "
                        "not causal and requires human review."
                    ),
                    "waterfall_figure": figure_path,
                }
            )

    pd.DataFrame(
        all_feature_dictionary_rows
    ).drop_duplicates(
        subset=[
            "checkpoint",
            "technical_feature",
        ]
    ).to_csv(
        args.output_root
        / "tables"
        / "shap_feature_dictionary.csv",
        index=False,
    )

    pd.DataFrame(
        all_local_summary_rows
    ).to_csv(
        args.output_root
        / "tables"
        / "shap_local_prototype_summary.csv",
        index=False,
    )

    print("\nSHAP analysis completed.")
    print(
        "Global importance tables:",
        args.output_root / "tables",
    )
    print(
        "Feature dictionary:",
        args.output_root
        / "tables"
        / "shap_feature_dictionary.csv",
    )
    print(
        "Local prototype summary:",
        args.output_root
        / "tables"
        / "shap_local_prototype_summary.csv",
    )
    print(
        "Figures:",
        args.output_root / "figures",
    )


if __name__ == "__main__":
    main()