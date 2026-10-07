from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)


# ============================================================
# Configuration
# ============================================================

CHECKPOINTS = ["week3", "week6", "week8", "week12"]

MODELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

CHECKPOINT_WEEKS = {
    "week3": 3,
    "week6": 6,
    "week8": 8,
    "week12": 12,
}

REFERENCE_THRESHOLDS = [0.30, 0.40, 0.50]

# Threshold candidates used for operational threshold selection.
THRESHOLD_GRID = np.round(
    np.arange(0.05, 0.96, 0.01),
    2,
)

# F2 gives recall four times the weight of precision.
F_BETA = 2.0

# Small constant used before logit transformation.
EPSILON = 1e-6


# ============================================================
# Directories and logging
# ============================================================

def create_directories(output_root: Path) -> None:
    for folder in [
        "tables",
        "figures",
        "predictions",
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
            / "day5_calibration_thresholds_errors.log"
        ),
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


# ============================================================
# File loading
# ============================================================

def load_oof_predictions(
    output_root: Path,
) -> pd.DataFrame:
    file_path = (
        output_root
        / "predictions"
        / "all_models_development_oof_predictions.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            "Development OOF prediction file not found:\n"
            f"{file_path}\n\n"
            "Run 15_calibration_thresholds.py first."
        )

    dataframe = pd.read_csv(file_path)

    required_columns = {
        "unique_identifier",
        "checkpoint",
        "model_name",
        "true_label",
        "predicted_probability",
    }

    missing_columns = required_columns - set(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            f"OOF file is missing columns: "
            f"{sorted(missing_columns)}"
        )

    return dataframe


def load_prediction_file(
    output_root: Path,
    checkpoint: str,
    model_name: str,
    split: str,
) -> pd.DataFrame:
    file_path = (
        output_root
        / "predictions"
        / f"{checkpoint}_{model_name}_{split}.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n{file_path}"
        )

    dataframe = pd.read_csv(file_path)

    required_columns = {
        "unique_identifier",
        "true_label",
        "predicted_probability",
    }

    missing_columns = required_columns - set(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            f"{file_path} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    dataframe["checkpoint"] = checkpoint
    dataframe["model_name"] = model_name

    return dataframe


# ============================================================
# Calibration utilities
# ============================================================

def clip_probabilities(
    probabilities: np.ndarray,
) -> np.ndarray:
    return np.clip(
        np.asarray(probabilities, dtype=float),
        EPSILON,
        1.0 - EPSILON,
    )


def probability_to_logit(
    probabilities: np.ndarray,
) -> np.ndarray:
    probabilities = clip_probabilities(probabilities)

    return np.log(
        probabilities / (1.0 - probabilities)
    )


def fit_platt_calibrator(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> LogisticRegression:
    """
    Fits sigmoid / Platt calibration using development OOF
    predicted probabilities only.
    """

    logits = probability_to_logit(
        probabilities
    ).reshape(-1, 1)

    calibrator = LogisticRegression(
        solver="lbfgs",
        max_iter=5000,
        random_state=42,
    )

    calibrator.fit(
        logits,
        np.asarray(y_true).astype(int),
    )

    return calibrator


def apply_platt_calibrator(
    calibrator: LogisticRegression,
    probabilities: np.ndarray,
) -> np.ndarray:
    logits = probability_to_logit(
        probabilities
    ).reshape(-1, 1)

    return calibrator.predict_proba(
        logits
    )[:, 1]


def calibration_intercept_slope(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    """
    Calibration model:
        logit(observed outcome) =
        intercept + slope * logit(predicted probability)

    Ideal intercept = 0
    Ideal slope = 1

    A slope below 1 commonly indicates overly extreme
    probabilities; a slope above 1 can indicate probabilities
    that are not extreme enough.
    """

    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if len(np.unique(y_true)) < 2:
        return {
            "calibration_intercept": np.nan,
            "calibration_slope": np.nan,
        }

    try:
        logits = probability_to_logit(
            probabilities
        ).reshape(-1, 1)

        model = LogisticRegression(
            solver="lbfgs",
            max_iter=5000,
            random_state=42,
        )

        model.fit(logits, y_true)

        return {
            "calibration_intercept": float(
                model.intercept_[0]
            ),
            "calibration_slope": float(
                model.coef_[0][0]
            ),
        }

    except Exception as error:
        logging.warning(
            "Could not calculate calibration intercept "
            "and slope: %s",
            repr(error),
        )

        return {
            "calibration_intercept": np.nan,
            "calibration_slope": np.nan,
        }


def calibration_direction(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> str:
    """
    A concise descriptive diagnostic based on average
    predicted probability versus observed event prevalence.
    """

    observed = float(np.mean(y_true))
    mean_predicted = float(np.mean(probabilities))
    difference = mean_predicted - observed

    if abs(difference) < 0.01:
        return "Overall mean probability close to observed prevalence"

    if difference > 0:
        return "Probabilities are high on average (over-prediction)"

    return "Probabilities are low on average (under-prediction)"


def calculate_calibration_statistics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    slope_intercept = calibration_intercept_slope(
        y_true,
        probabilities,
    )

    return {
        "brier_score": brier_score_loss(
            y_true,
            probabilities,
        ),
        "observed_prevalence": float(
            np.mean(y_true)
        ),
        "mean_predicted_probability": float(
            np.mean(probabilities)
        ),
        "mean_prediction_minus_prevalence": float(
            np.mean(probabilities) - np.mean(y_true)
        ),
        "calibration_direction": calibration_direction(
            y_true,
            probabilities,
        ),
        **slope_intercept,
    }


# ============================================================
# Threshold metrics
# ============================================================

def calculate_threshold_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    predicted_class = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predicted_class,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    negative_predictive_value = (
        tn / (tn + fn)
        if (tn + fn) > 0
        else np.nan
    )

    return {
        "threshold": float(threshold),
        "recall": recall_score(
            y_true,
            predicted_class,
            zero_division=0,
        ),
        "precision": precision_score(
            y_true,
            predicted_class,
            zero_division=0,
        ),
        "specificity": specificity,
        "negative_predictive_value": (
            negative_predictive_value
        ),
        "f1": f1_score(
            y_true,
            predicted_class,
            zero_division=0,
        ),
        "f2": fbeta_score(
            y_true,
            predicted_class,
            beta=F_BETA,
            zero_division=0,
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "students_flagged": int(
            predicted_class.sum()
        ),
        "percentage_flagged": float(
            predicted_class.mean() * 100
        ),
    }


def select_operational_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, pd.DataFrame]:
    """
    Threshold rule:
    1. Calculate F2 at every threshold from 0.05 to 0.95.
    2. Select the threshold with the highest F2 score.
    3. If multiple thresholds have the same F2 score, select
       the higher threshold to reduce alert volume.
    """

    rows = []

    for threshold in THRESHOLD_GRID:
        row = calculate_threshold_metrics(
            y_true,
            probabilities,
            float(threshold),
        )
        rows.append(row)

    threshold_table = pd.DataFrame(rows)

    maximum_f2 = threshold_table["f2"].max()

    candidate_rows = threshold_table[
        np.isclose(
            threshold_table["f2"],
            maximum_f2,
        )
    ].copy()

    selected_row = candidate_rows.sort_values(
        "threshold",
        ascending=False,
    ).iloc[0]

    return (
        float(selected_row["threshold"]),
        threshold_table,
    )


# ============================================================
# Reliability plot
# ============================================================

def create_reliability_data(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 8,
) -> pd.DataFrame:
    """
    Equal-frequency bins are preferable here because of the
    small sample and relatively uncommon positive class.
    """

    dataframe = pd.DataFrame(
        {
            "true_label": y_true,
            "predicted_probability": probabilities,
        }
    ).copy()

    number_of_unique_values = dataframe[
        "predicted_probability"
    ].nunique()

    usable_bins = min(
        n_bins,
        number_of_unique_values,
    )

    if usable_bins < 2:
        return pd.DataFrame(
            columns=[
                "mean_predicted_probability",
                "observed_risk",
                "n",
            ]
        )

    dataframe["bin"] = pd.qcut(
        dataframe["predicted_probability"],
        q=usable_bins,
        duplicates="drop",
    )

    output = (
        dataframe.groupby(
            "bin",
            observed=True,
        )
        .agg(
            mean_predicted_probability=(
                "predicted_probability",
                "mean",
            ),
            observed_risk=(
                "true_label",
                "mean",
            ),
            n=("true_label", "size"),
        )
        .reset_index(drop=True)
    )

    return output


def plot_reliability_panel(
    calibration_plot_data: pd.DataFrame,
    output_root: Path,
    dataset_label: str,
    filename: str,
) -> None:
    figure, axes = plt.subplots(
        nrows=4,
        ncols=3,
        figsize=(15, 18),
        sharex=True,
        sharey=True,
    )

    axes = axes.ravel()

    combinations = [
        (checkpoint, model_name)
        for checkpoint in CHECKPOINTS
        for model_name in MODELS
    ]

    for axis, (
        checkpoint,
        model_name,
    ) in zip(axes, combinations):
        selected = calibration_plot_data[
            (
                calibration_plot_data["checkpoint"]
                == checkpoint
            )
            & (
                calibration_plot_data["model_name"]
                == model_name
            )
        ]

        axis.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="black",
            linewidth=1,
            label="Perfect calibration",
        )

        for probability_type, line_style in [
            ("uncalibrated", "-"),
            ("platt_calibrated", "--"),
        ]:
            subset = selected[
                selected["probability_type"]
                == probability_type
            ]

            if subset.empty:
                continue

            label = (
                "Uncalibrated"
                if probability_type == "uncalibrated"
                else "Platt calibrated"
            )

            axis.plot(
                subset[
                    "mean_predicted_probability"
                ],
                subset["observed_risk"],
                marker="o",
                linestyle=line_style,
                linewidth=2,
                label=label,
            )

        axis.set_title(
            f"{checkpoint.title()} — "
            f"{MODELS[model_name]}"
        )

        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.3)

    for axis in axes:
        axis.set_xlabel("Mean predicted probability")
        axis.set_ylabel("Observed at-risk proportion")

    handles, labels = axes[0].get_legend_handles_labels()

    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
    )

    figure.suptitle(
        f"Reliability plots: {dataset_label}",
        fontsize=16,
    )

    figure.tight_layout(
        rect=[0, 0.05, 1, 0.97]
    )

    figure.savefig(
        output_root / "figures" / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# Threshold figures
# ============================================================

def plot_threshold_curves(
    threshold_results: pd.DataFrame,
    output_root: Path,
) -> None:
    plot_definitions = [
        (
            "recall",
            "Recall",
            "threshold_recall_curves.png",
        ),
        (
            "precision",
            "Precision",
            "threshold_precision_curves.png",
        ),
        (
            "students_flagged",
            "Students flagged",
            "threshold_alert_volume_curves.png",
        ),
    ]

    for metric, y_label, filename in plot_definitions:
        figure, axes = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=(12, 10),
            sharex=True,
        )

        axes = axes.ravel()

        for axis, checkpoint in zip(
            axes,
            CHECKPOINTS,
        ):
            for model_name, model_label in MODELS.items():
                subset = threshold_results[
                    (
                        threshold_results["checkpoint"]
                        == checkpoint
                    )
                    & (
                        threshold_results["model_name"]
                        == model_name
                    )
                    & (
                        threshold_results["dataset"]
                        == "development_oof"
                    )
                ].sort_values("threshold")

                axis.plot(
                    subset["threshold"],
                    subset[metric],
                    linewidth=2,
                    label=model_label,
                )

            axis.set_title(
                checkpoint.title()
            )
            axis.set_xlim(0.05, 0.95)
            axis.grid(alpha=0.3)

            if metric != "students_flagged":
                axis.set_ylim(0, 1.05)

        for axis in axes:
            axis.set_xlabel("Alert threshold")
            axis.set_ylabel(y_label)

        handles, labels = (
            axes[0].get_legend_handles_labels()
        )

        figure.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=3,
        )

        figure.suptitle(
            f"Development OOF {y_label.lower()} "
            "versus alert threshold",
            fontsize=15,
        )

        figure.tight_layout(
            rect=[0, 0.05, 1, 0.96]
        )

        figure.savefig(
            output_root / "figures" / filename,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(figure)


# ============================================================
# Error analysis
# ============================================================

def assign_error_group(
    y_true: np.ndarray,
    predicted_class: np.ndarray,
) -> np.ndarray:
    y_true = np.asarray(y_true).astype(int)
    predicted_class = np.asarray(
        predicted_class
    ).astype(int)

    return np.select(
        [
            (y_true == 1)
            & (predicted_class == 1),
            (y_true == 0)
            & (predicted_class == 1),
            (y_true == 1)
            & (predicted_class == 0),
            (y_true == 0)
            & (predicted_class == 0),
        ],
        [
            "true_positive",
            "false_positive",
            "false_negative",
            "true_negative",
        ],
        default="unknown",
    )


def create_error_analysis_records(
    dataframe: pd.DataFrame,
    probability_column: str,
    selected_threshold: float,
    dataset_label: str,
) -> pd.DataFrame:
    output = dataframe.copy()

    output["probability_used"] = output[
        probability_column
    ]

    output["selected_threshold"] = (
        selected_threshold
    )

    output["predicted_class_selected_threshold"] = (
        output["probability_used"]
        >= selected_threshold
    ).astype(int)

    output["error_group"] = assign_error_group(
        output["true_label"],
        output[
            "predicted_class_selected_threshold"
        ],
    )

    output["dataset"] = dataset_label

    keep_columns = [
        "unique_identifier",
        "checkpoint",
        "model_name",
        "dataset",
        "true_label",
        "predicted_probability",
        "probability_used",
        "selected_threshold",
        "predicted_class_selected_threshold",
        "error_group",
    ]

    for optional_column in [
        "semester",
        "fold",
        "validation_semester",
    ]:
        if optional_column in output.columns:
            keep_columns.insert(
                4,
                optional_column,
            )

    return output[keep_columns].copy()


# ============================================================
# Main workflow
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Day 5 calibration, threshold selection and "
            "structured error analysis."
        )
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Path to outputs directory.",
    )

    args = parser.parse_args()

    create_directories(args.output_root)
    configure_logging(args.output_root)

    logging.info("Starting Day 5 analysis.")
    logging.info(
        "Calibration method considered: Platt sigmoid."
    )
    logging.info(
        "Threshold selection rule: maximise F2 "
        "on development OOF predictions; choose higher "
        "threshold if tied."
    )

    oof_predictions = load_oof_predictions(
        args.output_root
    )

    calibration_statistics_rows = []
    calibration_plot_rows = []
    calibration_decision_rows = []
    threshold_rows = []
    selected_threshold_rows = []
    holdout_rows = []
    error_records = []

    # --------------------------------------------------------
    # Process every model and checkpoint independently.
    # --------------------------------------------------------

    for checkpoint in CHECKPOINTS:
        for model_name, model_label in MODELS.items():
            print(
                f"Processing {checkpoint} — {model_label}"
            )

            oof_subset = oof_predictions[
                (
                    oof_predictions["checkpoint"]
                    == checkpoint
                )
                & (
                    oof_predictions["model_name"]
                    == model_name
                )
            ].copy()

            if oof_subset.empty:
                raise ValueError(
                    f"No OOF predictions found for "
                    f"{checkpoint}, {model_name}."
                )

            validation_subset = load_prediction_file(
                args.output_root,
                checkpoint,
                model_name,
                "validation",
            )

            testing_subset = load_prediction_file(
                args.output_root,
                checkpoint,
                model_name,
                "testing",
            )

            # ------------------------------------------------
            # Fit Platt calibration on development OOF only.
            # ------------------------------------------------

            calibrator = fit_platt_calibrator(
                oof_subset["true_label"].to_numpy(),
                oof_subset[
                    "predicted_probability"
                ].to_numpy(),
            )

            # OOF calibrated values are used for threshold
            # selection after calibration decision is assessed.
            oof_subset[
                "platt_calibrated_probability"
            ] = apply_platt_calibrator(
                calibrator,
                oof_subset[
                    "predicted_probability"
                ].to_numpy(),
            )

            validation_subset[
                "platt_calibrated_probability"
            ] = apply_platt_calibrator(
                calibrator,
                validation_subset[
                    "predicted_probability"
                ].to_numpy(),
            )

            testing_subset[
                "platt_calibrated_probability"
            ] = apply_platt_calibrator(
                calibrator,
                testing_subset[
                    "predicted_probability"
                ].to_numpy(),
            )

            # ------------------------------------------------
            # Calibration statistics:
            # OOF = development diagnostic;
            # 2025 S2 = calibration retention decision;
            # 2026 S1 = final frozen evaluation only.
            # ------------------------------------------------

            datasets = [
                (
                    "development_oof",
                    oof_subset,
                ),
                (
                    "2025 S2 validation",
                    validation_subset,
                ),
                (
                    "2026 S1 holdout",
                    testing_subset,
                ),
            ]

            for dataset_label, dataset in datasets:
                for probability_type, column in [
                    (
                        "uncalibrated",
                        "predicted_probability",
                    ),
                    (
                        "platt_calibrated",
                        "platt_calibrated_probability",
                    ),
                ]:
                    statistics = (
                        calculate_calibration_statistics(
                            dataset["true_label"].to_numpy(),
                            dataset[column].to_numpy(),
                        )
                    )

                    calibration_statistics_rows.append(
                        {
                            "checkpoint": checkpoint,
                            "checkpoint_week": (
                                CHECKPOINT_WEEKS[
                                    checkpoint
                                ]
                            ),
                            "model_name": model_name,
                            "model_label": model_label,
                            "dataset": dataset_label,
                            "probability_type": (
                                probability_type
                            ),
                            "n": len(dataset),
                            "positive_cases": int(
                                dataset[
                                    "true_label"
                                ].sum()
                            ),
                            **statistics,
                        }
                    )

                    reliability = (
                        create_reliability_data(
                            dataset[
                                "true_label"
                            ].to_numpy(),
                            dataset[column].to_numpy(),
                        )
                    )

                    if not reliability.empty:
                        reliability[
                            "checkpoint"
                        ] = checkpoint

                        reliability[
                            "model_name"
                        ] = model_name

                        reliability[
                            "dataset"
                        ] = dataset_label

                        reliability[
                            "probability_type"
                        ] = probability_type

                        calibration_plot_rows.append(
                            reliability
                        )

            # ------------------------------------------------
            # Retain Platt calibration only if Brier score
            # improves on the separate 2025 S2 validation set.
            # No 2026 S1 information is used here.
            # ------------------------------------------------

            validation_uncalibrated_brier = brier_score_loss(
                validation_subset["true_label"],
                validation_subset[
                    "predicted_probability"
                ],
            )

            validation_calibrated_brier = brier_score_loss(
                validation_subset["true_label"],
                validation_subset[
                    "platt_calibrated_probability"
                ],
            )

            retain_calibration = (
                validation_calibrated_brier
                < validation_uncalibrated_brier
            )

            selected_probability_column = (
                "platt_calibrated_probability"
                if retain_calibration
                else "predicted_probability"
            )

            calibration_method_retained = (
                "platt_sigmoid"
                if retain_calibration
                else "none"
            )

            calibration_decision_rows.append(
                {
                    "checkpoint": checkpoint,
                    "checkpoint_week": (
                        CHECKPOINT_WEEKS[checkpoint]
                    ),
                    "model_name": model_name,
                    "model_label": model_label,
                    "calibration_fitted_on": (
                        "development OOF only "
                        "(2024 S2 and 2025 S1)"
                    ),
                    "calibration_decision_dataset": (
                        "2025 S2 validation"
                    ),
                    "uncalibrated_validation_brier": (
                        validation_uncalibrated_brier
                    ),
                    "platt_validation_brier": (
                        validation_calibrated_brier
                    ),
                    "brier_difference_platt_minus_"
                    "uncalibrated": (
                        validation_calibrated_brier
                        - validation_uncalibrated_brier
                    ),
                    "calibration_retained": (
                        retain_calibration
                    ),
                    "retained_calibration_method": (
                        calibration_method_retained
                    ),
                    "probability_used_for_thresholds": (
                        selected_probability_column
                    ),
                }
            )

            # ------------------------------------------------
            # Threshold selection: development OOF only.
            # ------------------------------------------------

            selected_threshold, threshold_grid = (
                select_operational_threshold(
                    oof_subset["true_label"].to_numpy(),
                    oof_subset[
                        selected_probability_column
                    ].to_numpy(),
                )
            )

            threshold_grid["checkpoint"] = checkpoint
            threshold_grid["checkpoint_week"] = (
                CHECKPOINT_WEEKS[checkpoint]
            )
            threshold_grid["model_name"] = model_name
            threshold_grid["model_label"] = model_label
            threshold_grid["dataset"] = (
                "development_oof"
            )
            threshold_grid["probability_type_used"] = (
                calibration_method_retained
            )
            threshold_grid["is_selected_threshold"] = (
                np.isclose(
                    threshold_grid["threshold"],
                    selected_threshold,
                )
            )

            threshold_rows.append(threshold_grid)

            selected_threshold_rows.append(
                {
                    "checkpoint": checkpoint,
                    "checkpoint_week": (
                        CHECKPOINT_WEEKS[checkpoint]
                    ),
                    "model_name": model_name,
                    "model_label": model_label,
                    "threshold_selection_data": (
                        "development OOF only"
                    ),
                    "threshold_selection_rule": (
                        "Maximise F2; choose higher "
                        "threshold if tied"
                    ),
                    "f_beta": F_BETA,
                    "calibration_method_used": (
                        calibration_method_retained
                    ),
                    "selected_threshold": (
                        selected_threshold
                    ),
                }
            )

            # ------------------------------------------------
            # Required threshold table:
            # 0.30, 0.40, 0.50 on development OOF.
            # ------------------------------------------------

            reference_rows = []

            for threshold in REFERENCE_THRESHOLDS:
                metrics = calculate_threshold_metrics(
                    oof_subset["true_label"].to_numpy(),
                    oof_subset[
                        selected_probability_column
                    ].to_numpy(),
                    threshold,
                )

                reference_rows.append(
                    {
                        "checkpoint": checkpoint,
                        "checkpoint_week": (
                            CHECKPOINT_WEEKS[
                                checkpoint
                            ]
                        ),
                        "model_name": model_name,
                        "model_label": model_label,
                        "dataset": "development_oof",
                        "probability_type_used": (
                            calibration_method_retained
                        ),
                        **metrics,
                    }
                )

            threshold_rows.append(
                pd.DataFrame(reference_rows)
            )

            # ------------------------------------------------
            # Apply frozen calibration decision and frozen
            # OOF-selected threshold to 2026 S1 holdout.
            # ------------------------------------------------

            holdout_metrics = calculate_threshold_metrics(
                testing_subset["true_label"].to_numpy(),
                testing_subset[
                    selected_probability_column
                ].to_numpy(),
                selected_threshold,
            )

            holdout_rows.append(
                {
                    "checkpoint": checkpoint,
                    "checkpoint_week": (
                        CHECKPOINT_WEEKS[checkpoint]
                    ),
                    "model_name": model_name,
                    "model_label": model_label,
                    "dataset": "2026 S1 holdout",
                    "calibration_method_used": (
                        calibration_method_retained
                    ),
                    "selected_threshold": (
                        selected_threshold
                    ),
                    **holdout_metrics,
                }
            )

            # ------------------------------------------------
            # Non-identifying structured error records:
            # final 2026 S1 holdout only.
            # ------------------------------------------------

            error_records.append(
                create_error_analysis_records(
                    testing_subset,
                    selected_probability_column,
                    selected_threshold,
                    "2026 S1 holdout",
                )
            )

    # --------------------------------------------------------
    # Save tables
    # --------------------------------------------------------

    calibration_statistics = pd.DataFrame(
        calibration_statistics_rows
    )

    calibration_statistics.to_csv(
        args.output_root
        / "tables"
        / "calibration_statistics.csv",
        index=False,
    )

    calibration_decisions = pd.DataFrame(
        calibration_decision_rows
    )

    calibration_decisions.to_csv(
        args.output_root
        / "tables"
        / "calibration_decisions.csv",
        index=False,
    )

    selected_thresholds = pd.DataFrame(
        selected_threshold_rows
    )

    selected_thresholds.to_csv(
        args.output_root
        / "tables"
        / "selected_operational_thresholds.csv",
        index=False,
    )

    holdout_threshold_results = pd.DataFrame(
        holdout_rows
    )

    holdout_threshold_results.to_csv(
        args.output_root
        / "tables"
        / "holdout_selected_threshold_results.csv",
        index=False,
    )

    all_threshold_rows = pd.concat(
        threshold_rows,
        ignore_index=True,
    )

    # Removes duplicated reference rows if a threshold appears
    # both in the full grid and reference-threshold table.
    all_threshold_rows = all_threshold_rows.drop_duplicates(
        subset=[
            "checkpoint",
            "model_name",
            "dataset",
            "threshold",
        ],
        keep="first",
    )

    all_threshold_rows.to_csv(
        args.output_root
        / "tables"
        / "threshold_analysis_development_oof.csv",
        index=False,
    )

    reference_threshold_table = all_threshold_rows[
        all_threshold_rows["threshold"].isin(
            REFERENCE_THRESHOLDS
        )
    ].copy()

    reference_threshold_table.to_csv(
        args.output_root
        / "tables"
        / "threshold_table_030_040_050.csv",
        index=False,
    )

    operational_alert_volumes = (
        holdout_threshold_results[
            [
                "checkpoint",
                "checkpoint_week",
                "model_name",
                "model_label",
                "calibration_method_used",
                "selected_threshold",
                "students_flagged",
                "percentage_flagged",
                "true_positive",
                "false_positive",
                "false_negative",
                "true_negative",
            ]
        ]
        .copy()
    )

    operational_alert_volumes.to_csv(
        args.output_root
        / "tables"
        / "operational_alert_volume_estimates.csv",
        index=False,
    )

    error_analysis_records = pd.concat(
        error_records,
        ignore_index=True,
    )

    error_analysis_records.to_csv(
        args.output_root
        / "predictions"
        / "error_analysis_records_2026_s1.csv",
        index=False,
    )

    error_analysis_summary = (
        error_analysis_records.groupby(
            [
                "checkpoint",
                "model_name",
                "selected_threshold",
                "error_group",
            ],
            observed=True,
        )
        .size()
        .reset_index(name="number_records")
    )

    error_analysis_summary.to_csv(
        args.output_root
        / "tables"
        / "error_analysis_summary_2026_s1.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Reliability figures
    # --------------------------------------------------------

    calibration_plot_data = pd.concat(
        calibration_plot_rows,
        ignore_index=True,
    )

    calibration_plot_data.to_csv(
        args.output_root
        / "tables"
        / "reliability_plot_data.csv",
        index=False,
    )

    for dataset_key, dataset_label, filename in [
        (
            "development_oof",
            "Development OOF predictions",
            "calibration_plots_development_oof.png",
        ),
        (
            "2025 S2 validation",
            "2025 S2 validation predictions",
            "calibration_plots_2025_s2_validation.png",
        ),
        (
            "2026 S1 holdout",
            "2026 S1 holdout predictions",
            "calibration_plots_2026_s1_holdout.png",
        ),
    ]:
        subset = calibration_plot_data[
            calibration_plot_data["dataset"] == dataset_key
        ].copy()

        if not subset.empty:
            plot_reliability_panel(
                subset,
                args.output_root,
                dataset_label,
                filename,
            )

    # --------------------------------------------------------
    # Threshold figures
    # --------------------------------------------------------

    plot_threshold_curves(
        all_threshold_rows,
        args.output_root,
    )

    # --------------------------------------------------------
    # Decision log
    # --------------------------------------------------------

    decision_log = {
        "day": "Day 5",
        "development_calibration_data": (
            "OOF predictions for 2024 S2 and 2025 S1"
        ),
        "calibration_method_considered": (
            "Platt sigmoid calibration"
        ),
        "isotonic_calibration": (
            "Not used because the development OOF sample "
            "is limited (170 predictions, 32 positive cases)."
        ),
        "calibration_retention_rule": (
            "Retain Platt calibration only if the 2025 S2 "
            "validation Brier score is lower than the "
            "uncalibrated Brier score."
        ),
        "threshold_selection_data": (
            "Development OOF predictions only"
        ),
        "threshold_selection_rule": (
            "Select the threshold that maximises F2 on "
            "the development OOF predictions; select the "
            "higher threshold if F2 is tied."
        ),
        "f_beta": F_BETA,
        "reference_thresholds_reported": (
            REFERENCE_THRESHOLDS
        ),
        "final_holdout": "2026 S1",
        "holdout_use": (
            "Final evaluation only. It must not be used to "
            "change calibration or threshold decisions."
        ),
    }

    with open(
        args.output_root
        / "logs"
        / "day5_decision_log.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            decision_log,
            file,
            indent=2,
        )

    print("\nDay 5 analysis completed.")
    print(
        "Calibration statistics:",
        args.output_root
        / "tables"
        / "calibration_statistics.csv",
    )
    print(
        "Calibration decisions:",
        args.output_root
        / "tables"
        / "calibration_decisions.csv",
    )
    print(
        "Threshold table:",
        args.output_root
        / "tables"
        / "threshold_table_030_040_050.csv",
    )
    print(
        "Selected thresholds:",
        args.output_root
        / "tables"
        / "selected_operational_thresholds.csv",
    )
    print(
        "Final holdout threshold results:",
        args.output_root
        / "tables"
        / "holdout_selected_threshold_results.csv",
    )
    print(
        "Error-analysis records:",
        args.output_root
        / "predictions"
        / "error_analysis_records_2026_s1.csv",
    )
    print(
        "Figures:",
        args.output_root / "figures",
    )


if __name__ == "__main__":
    main()