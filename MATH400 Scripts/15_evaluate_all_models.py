#It reads the existing prediction and fold-result files, calculates identical metrics, performs ordinary row-level bootstrap confidence intervals, and creates the requested comparison figures.


from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
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

THRESHOLD = 0.40
BOOTSTRAP_REPLICATES = 2000
RANDOM_SEED = 42


# ============================================================
# Directories
# ============================================================

def create_directories(output_root: Path) -> None:
    for folder in [
        "tables",
        "figures",
        "predictions",
    ]:
        (output_root / folder).mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# Metric calculation
# ============================================================

def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = THRESHOLD,
) -> dict:
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities).astype(float)

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
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
        "roc_auc": (
            roc_auc_score(
                y_true,
                probabilities,
            )
            if len(np.unique(y_true)) == 2
            else np.nan
        ),
        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "specificity": specificity,
        "negative_predictive_value": (
            negative_predictive_value
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "balanced_accuracy": (
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        ),
        "brier_score": brier_score_loss(
            y_true,
            probabilities,
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "number_flagged": int(predictions.sum()),
        "percentage_flagged": float(
            predictions.mean() * 100
        ),
        "n": int(len(y_true)),
        "threshold": threshold,
    }


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    replicates: int,
    seed: int,
) -> dict:
    """
    Ordinary row-level bootstrap.

    This is appropriate here because the identifiers were confirmed
    to be unique in the evaluated files.
    """

    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities).astype(float)

    rng = np.random.default_rng(seed)
    n_rows = len(y_true)

    bootstrap_results = []

    for _ in range(replicates):
        indices = rng.integers(
            0,
            n_rows,
            size=n_rows,
        )

        sampled_y = y_true[indices]
        sampled_probabilities = probabilities[
            indices
        ]

        # ROC-AUC is undefined if a sample contains one class.
        if len(np.unique(sampled_y)) < 2:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            bootstrap_results.append(
                calculate_metrics(
                    sampled_y,
                    sampled_probabilities,
                    threshold,
                )
            )

    bootstrap_table = pd.DataFrame(
        bootstrap_results
    )

    metrics = [
        "roc_auc",
        "pr_auc",
        "recall",
        "precision",
        "specificity",
        "negative_predictive_value",
        "f1",
        "balanced_accuracy",
        "brier_score",
        "number_flagged",
        "percentage_flagged",
    ]

    output = {
        "bootstrap_replicates_requested": replicates,
        "bootstrap_replicates_used": len(
            bootstrap_table
        ),
    }

    for metric in metrics:
        if (
            metric not in bootstrap_table.columns
            or bootstrap_table[metric].dropna().empty
        ):
            output[f"{metric}_ci_lower"] = np.nan
            output[f"{metric}_ci_upper"] = np.nan
            continue

        values = bootstrap_table[metric].dropna()

        output[f"{metric}_ci_lower"] = np.percentile(
            values,
            2.5,
        )

        output[f"{metric}_ci_upper"] = np.percentile(
            values,
            97.5,
        )

    return output


# ============================================================
# File loading
# ============================================================

def load_prediction_file(
    output_root: Path,
    checkpoint: str,
    model: str,
) -> pd.DataFrame:
    file_path = (
        output_root
        / "predictions"
        / f"{checkpoint}_{model}_testing.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n{file_path}"
        )

    dataframe = pd.read_csv(file_path)

    required_columns = {
        "true_label",
        "predicted_probability",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{file_path} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    return dataframe


def load_best_fold_results(
    output_root: Path,
    model: str,
) -> pd.DataFrame:
    rows = []

    for checkpoint in CHECKPOINTS:
        candidate_files = [
            (
                output_root
                / "tables"
                / f"{checkpoint}_{model}_best_fold_results.csv"
            )
        ]

        # Logistic Regression used "logistic" rather than
        # "logistic_regression" in its fold-result filename.
        if model == "logistic_regression":
            candidate_files.append(
                output_root
                / "tables"
                / f"{checkpoint}_logistic_best_fold_results.csv"
            )

        file_path = next(
            (
                candidate
                for candidate in candidate_files
                if candidate.exists()
            ),
            None,
        )

        if file_path is None:
            print(
                f"Warning: fold file not found for "
                f"{checkpoint} {model}"
            )
            continue

        dataframe = pd.read_csv(file_path)
        dataframe["checkpoint"] = checkpoint
        dataframe["model"] = model
        rows.append(dataframe)

    if not rows:
        return pd.DataFrame()

    return pd.concat(
        rows,
        ignore_index=True,
    )
# ============================================================
# Holdout metrics and tables
# ============================================================

def evaluate_holdout(
    output_root: Path,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    confusion_rows = []

    for checkpoint in CHECKPOINTS:
        for model in MODELS:
            dataframe = load_prediction_file(
                output_root,
                checkpoint,
                model,
            )

            y_true = dataframe[
                "true_label"
            ].to_numpy()

            probabilities = dataframe[
                "predicted_probability"
            ].to_numpy()

            metrics = calculate_metrics(
                y_true,
                probabilities,
                THRESHOLD,
            )

            intervals = (
                bootstrap_confidence_intervals(
                    y_true,
                    probabilities,
                    THRESHOLD,
                    BOOTSTRAP_REPLICATES,
                    seed,
                )
            )

            metric_row = {
                "checkpoint": checkpoint,
                "checkpoint_week": CHECKPOINT_WEEKS[
                    checkpoint
                ],
                "model": model,
                "model_label": MODELS[model],
                "semester": "2026 S1",
                **metrics,
                **intervals,
            }

            metric_rows.append(metric_row)

            confusion_rows.append(
                {
                    "checkpoint": checkpoint,
                    "checkpoint_week": CHECKPOINT_WEEKS[
                        checkpoint
                    ],
                    "model": model,
                    "model_label": MODELS[model],
                    "true_negative": metrics[
                        "true_negative"
                    ],
                    "false_positive": metrics[
                        "false_positive"
                    ],
                    "false_negative": metrics[
                        "false_negative"
                    ],
                    "true_positive": metrics[
                        "true_positive"
                    ],
                    "threshold": THRESHOLD,
                }
            )

    metrics_table = pd.DataFrame(metric_rows)
    confusion_table = pd.DataFrame(confusion_rows)

    metrics_table = metrics_table.sort_values(
        [
            "checkpoint_week",
            "model_label",
        ]
    )

    confusion_table = confusion_table.sort_values(
        [
            "checkpoint_week",
            "model_label",
        ]
    )

    metrics_table.to_csv(
        output_root
        / "tables"
        / "all_models_2026_s1_holdout_metrics.csv",
        index=False,
    )

    confusion_table.to_csv(
        output_root
        / "tables"
        / "all_models_2026_s1_confusion_matrices.csv",
        index=False,
    )

    return metrics_table, confusion_table


# ============================================================
# Figure utilities
# ============================================================

def apply_common_legend(figure, axes) -> None:
    handles = []
    labels = []

    for axis in axes:
        current_handles, current_labels = (
            axis.get_legend_handles_labels()
        )

        handles.extend(current_handles)
        labels.extend(current_labels)

    unique = dict(zip(labels, handles))

    figure.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
    )


def plot_roc_curves(
    output_root: Path,
) -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 10),
        sharex=True,
        sharey=True,
    )

    axes = axes.ravel()

    for axis, checkpoint in zip(
        axes,
        CHECKPOINTS,
    ):
        for model, model_label in MODELS.items():
            dataframe = load_prediction_file(
                output_root,
                checkpoint,
                model,
            )

            y_true = dataframe[
                "true_label"
            ].to_numpy()

            probabilities = dataframe[
                "predicted_probability"
            ].to_numpy()

            fpr, tpr, _ = roc_curve(
                y_true,
                probabilities,
            )

            auc = roc_auc_score(
                y_true,
                probabilities,
            )

            axis.plot(
                fpr,
                tpr,
                linewidth=2,
                label=f"{model_label} (AUC={auc:.3f})",
            )

        axis.plot(
            [0, 1],
            [0, 1],
            "k--",
            linewidth=1,
        )

        axis.set_title(
            f"{checkpoint.title()} ROC curve"
        )
        axis.set_xlabel("False-positive rate")
        axis.set_ylabel("True-positive rate")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.3)

    apply_common_legend(figure, axes)

    figure.suptitle(
        "2026 S1 holdout ROC curves by checkpoint",
        fontsize=15,
    )

    figure.tight_layout()

    figure.savefig(
        output_root
        / "figures"
        / "roc_curves_by_checkpoint.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_pr_curves(
    output_root: Path,
) -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 10),
        sharex=True,
        sharey=True,
    )

    axes = axes.ravel()

    for axis, checkpoint in zip(
        axes,
        CHECKPOINTS,
    ):
        for model, model_label in MODELS.items():
            dataframe = load_prediction_file(
                output_root,
                checkpoint,
                model,
            )

            y_true = dataframe[
                "true_label"
            ].to_numpy()

            probabilities = dataframe[
                "predicted_probability"
            ].to_numpy()

            precision, recall, _ = (
                precision_recall_curve(
                    y_true,
                    probabilities,
                )
            )

            pr_auc = average_precision_score(
                y_true,
                probabilities,
            )

            axis.plot(
                recall,
                precision,
                linewidth=2,
                label=f"{model_label} (PR-AUC={pr_auc:.3f})",
            )

        prevalence = y_true.mean()

        axis.axhline(
            prevalence,
            color="k",
            linestyle="--",
            linewidth=1,
            label="Prevalence",
        )

        axis.set_title(
            f"{checkpoint.title()} precision–recall curve"
        )
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.3)

    apply_common_legend(figure, axes)

    figure.suptitle(
        "2026 S1 holdout precision–recall curves by checkpoint",
        fontsize=15,
    )

    figure.tight_layout()

    figure.savefig(
        output_root
        / "figures"
        / "precision_recall_curves_by_checkpoint.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_metric_by_checkpoint(
    metrics_table: pd.DataFrame,
    output_root: Path,
    metric: str,
    title: str,
    ylabel: str,
    filename: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    for model, model_label in MODELS.items():
        subset = metrics_table[
            metrics_table["model"] == model
        ].sort_values("checkpoint_week")

        axis.plot(
            subset["checkpoint_week"],
            subset[metric],
            marker="o",
            linewidth=2,
            label=model_label,
        )

    axis.set_title(title)
    axis.set_xlabel("Checkpoint week")
    axis.set_ylabel(ylabel)
    axis.set_xticks([3, 6, 8, 12])

    if ylim is not None:
        axis.set_ylim(*ylim)

    axis.grid(alpha=0.3)
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_root / "figures" / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_recall_precision(
    metrics_table: pd.DataFrame,
    output_root: Path,
) -> None:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
        sharex=True,
        sharey=True,
    )

    for model, model_label in MODELS.items():
        subset = metrics_table[
            metrics_table["model"] == model
        ].sort_values("checkpoint_week")

        axes[0].plot(
            subset["checkpoint_week"],
            subset["recall"],
            marker="o",
            linewidth=2,
            label=model_label,
        )

        axes[1].plot(
            subset["checkpoint_week"],
            subset["precision"],
            marker="o",
            linewidth=2,
            label=model_label,
        )

    axes[0].set_title("Recall versus checkpoint week")
    axes[1].set_title("Precision versus checkpoint week")

    for axis in axes:
        axis.set_xlabel("Checkpoint week")
        axis.set_ylabel("Score")
        axis.set_xticks([3, 6, 8, 12])
        axis.set_ylim(0, 1.05)
        axis.grid(alpha=0.3)

    axes[0].legend()

    figure.suptitle(
        "2026 S1 holdout recall and precision",
        fontsize=15,
    )

    figure.tight_layout()

    figure.savefig(
        output_root
        / "figures"
        / "recall_precision_by_checkpoint.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_confusion_matrices(
    confusion_table: pd.DataFrame,
    output_root: Path,
) -> None:
    figure, axes = plt.subplots(
        len(CHECKPOINTS),
        len(MODELS),
        figsize=(12, 14),
    )

    for row_number, checkpoint in enumerate(
        CHECKPOINTS
    ):
        for column_number, model in enumerate(
            MODELS
        ):
            axis = axes[
                row_number,
                column_number,
            ]

            selected = confusion_table[
                (
                    confusion_table["checkpoint"]
                    == checkpoint
                )
                & (
                    confusion_table["model"]
                    == model
                )
            ].iloc[0]

            matrix = np.array(
                [
                    [
                        selected["true_negative"],
                        selected["false_positive"],
                    ],
                    [
                        selected["false_negative"],
                        selected["true_positive"],
                    ],
                ]
            )

            image = axis.imshow(
                matrix,
                cmap="Blues",
                vmin=0,
                vmax=matrix.max()
                if matrix.max() > 0
                else 1,
            )

            axis.set_title(
                f"{checkpoint.title()} — "
                f"{MODELS[model]}"
            )

            axis.set_xticks([0, 1])
            axis.set_yticks([0, 1])
            axis.set_xticklabels(
                ["Predicted 0", "Predicted 1"]
            )
            axis.set_yticklabels(
                ["Actual 0", "Actual 1"]
            )

            for row in range(2):
                for column in range(2):
                    axis.text(
                        column,
                        row,
                        int(matrix[row, column]),
                        ha="center",
                        va="center",
                        fontsize=13,
                    )

    figure.suptitle(
        "2026 S1 holdout confusion matrices "
        f"(threshold={THRESHOLD:.2f})",
        fontsize=15,
    )

    figure.tight_layout()

    figure.savefig(
        output_root
        / "figures"
        / "confusion_matrices.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_fold_and_holdout_comparison(
    output_root: Path,
    holdout_metrics: pd.DataFrame,
) -> None:
    rows = []

    for model in MODELS:
        fold_table = load_best_fold_results(
            output_root,
            model,
        )

        if fold_table.empty:
            continue

        for checkpoint in CHECKPOINTS:
            subset = fold_table[
                fold_table["checkpoint"]
                == checkpoint
            ]

            if subset.empty:
                continue

            rows.append(
                {
                    "checkpoint": checkpoint,
                    "checkpoint_week": CHECKPOINT_WEEKS[
                        checkpoint
                    ],
                    "model": model,
                    "dataset": "Development forward folds",
                    "roc_auc": subset[
                        "roc_auc"
                    ].astype(float).mean(),
                    "pr_auc": subset[
                        "pr_auc"
                    ].astype(float).mean(),
                }
            )

    holdout_subset = holdout_metrics.copy()
    holdout_subset["dataset"] = (
        "2026 S1 temporal holdout"
    )

    rows.extend(
        holdout_subset[
            [
                "checkpoint",
                "checkpoint_week",
                "model",
                "dataset",
                "roc_auc",
                "pr_auc",
            ]
        ].to_dict("records")
    )

    comparison = pd.DataFrame(rows)

    comparison.to_csv(
        output_root
        / "tables"
        / "fold_and_holdout_performance_comparison.csv",
        index=False,
    )

    for metric in ["roc_auc", "pr_auc"]:
        figure, axis = plt.subplots(
            figsize=(10, 6)
        )

        for model, model_label in MODELS.items():
            for dataset, line_style in [
                (
                    "Development forward folds",
                    "--",
                ),
                (
                    "2026 S1 temporal holdout",
                    "-",
                ),
            ]:
                subset = comparison[
                    (
                        comparison["model"]
                        == model
                    )
                    & (
                        comparison["dataset"]
                        == dataset
                    )
                ].sort_values(
                    "checkpoint_week"
                )

                if subset.empty:
                    continue

                axis.plot(
                    subset["checkpoint_week"],
                    subset[metric],
                    marker="o",
                    linestyle=line_style,
                    linewidth=2,
                    label=(
                        f"{model_label} — "
                        f"{dataset}"
                    ),
                )

        axis.set_title(
            f"{metric.upper()} — development versus "
            "2026 S1 holdout"
        )
        axis.set_xlabel("Checkpoint week")
        axis.set_ylabel(metric.upper())
        axis.set_xticks([3, 6, 8, 12])
        axis.set_ylim(0, 1.05)
        axis.grid(alpha=0.3)
        axis.legend()

        figure.tight_layout()

        figure.savefig(
            output_root
            / "figures"
            / f"fold_and_holdout_{metric}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(figure)


# ============================================================
# Main
# ============================================================

def main() -> None:
    global BOOTSTRAP_REPLICATES
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("..") / "outputs",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
    )

    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=BOOTSTRAP_REPLICATES,
    )

    args = parser.parse_args()


    BOOTSTRAP_REPLICATES = (
        args.bootstrap_replicates
    )

    create_directories(args.output_root)

    print("Evaluating all models on 2026 S1 holdout...")
    print("Threshold:", THRESHOLD)
    print(
        "Bootstrap replicates:",
        BOOTSTRAP_REPLICATES,
    )

    holdout_metrics, confusion_table = (
        evaluate_holdout(
            args.output_root,
            args.seed,
        )
    )

    plot_roc_curves(args.output_root)
    plot_pr_curves(args.output_root)

    plot_metric_by_checkpoint(
        holdout_metrics,
        args.output_root,
        metric="roc_auc",
        title="ROC-AUC versus checkpoint week",
        ylabel="ROC-AUC",
        filename="roc_auc_by_checkpoint.png",
        ylim=(0, 1.05),
    )

    plot_metric_by_checkpoint(
        holdout_metrics,
        args.output_root,
        metric="pr_auc",
        title="PR-AUC versus checkpoint week",
        ylabel="PR-AUC",
        filename="pr_auc_by_checkpoint.png",
        ylim=(0, 1.05),
    )

    plot_recall_precision(
        holdout_metrics,
        args.output_root,
    )

    plot_metric_by_checkpoint(
        holdout_metrics,
        args.output_root,
        metric="brier_score",
        title="Brier score versus checkpoint week",
        ylabel="Brier score",
        filename="brier_score_by_checkpoint.png",
        ylim=None,
    )

    plot_confusion_matrices(
        confusion_table,
        args.output_root,
    )

    plot_fold_and_holdout_comparison(
        args.output_root,
        holdout_metrics,
    )

    print("\nEvaluation completed.")
    print(
        "Metrics:",
        args.output_root
        / "tables"
        / "all_models_2026_s1_holdout_metrics.csv",
    )
    print(
        "Confidence intervals included in the metrics table."
    )
    print(
        "Figures:",
        args.output_root / "figures",
    )


if __name__ == "__main__":
    main()