from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier

from src.modelling_common import (
    load_checkpoint_data,
    get_feature_columns,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
# ============================================================
# Configuration
# ============================================================

CHECKPOINTS = ["week3", "week6", "week8", "week12"]

TARGET_COLUMN = "at_risk"
IDENTIFIER_COLUMN = "unique_identifier"

RANDOM_SEED = 42
N_ITER_SEARCH = 20

FORWARD_FOLDS = [
    {
        "fold": 1,
        "training_semesters": ["2024 S1"],
        "validation_semester": "2024 S2",
    },
    {
        "fold": 2,
        "training_semesters": ["2024 S1", "2024 S2"],
        "validation_semester": "2025 S1",
    },
]

XGBOOST_PARAMETER_DISTRIBUTIONS = {
    "model__n_estimators": [
        100,
        200,
        300,
        500,
        800,
    ],
    "model__learning_rate": [
        0.01,
        0.03,
        0.05,
        0.10,
        0.15,
        0.20,
    ],
    "model__max_depth": [
        2,
        3,
        4,
        5,
        6,
    ],
    "model__min_child_weight": [
        1,
        3,
        5,
        8,
        10,
    ],
    "model__subsample": [
        0.60,
        0.80,
        1.00,
    ],
    "model__colsample_bytree": [
        0.60,
        0.80,
        1.00,
    ],
    "model__gamma": [
        0,
        1,
        3,
        5,
    ],
    "model__reg_alpha": [
        0,
        0.5,
        1,
        2,
    ],
    "model__reg_lambda": [
        1,
        3,
        5,
        10,
    ],
}


# ============================================================
# Directories and logging
# ============================================================

def create_directories(output_root: Path) -> None:
    for folder in [
        "models",
        "tables",
        "predictions",
        "logs",
    ]:
        (output_root / folder).mkdir(
            parents=True,
            exist_ok=True,
        )


def configure_logging(output_root: Path) -> None:
    logging.basicConfig(
        filename=output_root
        / "logs"
        / "xgboost_execution.log",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


# ============================================================
# Preprocessing and XGBoost pipeline
# ============================================================

def build_tree_preprocessor(
    training_df: pd.DataFrame,
    feature_columns: list[str],
) -> ColumnTransformer:

    X = training_df[feature_columns]

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = [
        column
        for column in feature_columns
        if column not in numeric_features
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            )
        ]
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


def build_xgboost_pipeline(
    training_df: pd.DataFrame,
    feature_columns: list[str],
    parameters: dict,
    scale_pos_weight: float,
    seed: int,
) -> Pipeline:

    preprocessor = build_tree_preprocessor(
        training_df,
        feature_columns,
    )

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,

        # Calculated only from development training data.
        scale_pos_weight=scale_pos_weight,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    pipeline.set_params(**parameters)

    return pipeline
def bootstrap_metric_intervals(
    y_true,
    probabilities,
    threshold: float = 0.40,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities)

    rng = np.random.default_rng(seed)
    n_rows = len(y_true)
    metric_rows = []

    for _ in range(n_bootstrap):
        sampled_indices = rng.integers(
            0,
            n_rows,
            size=n_rows,
        )

        sampled_y = y_true[sampled_indices]
        sampled_probabilities = (
            probabilities[sampled_indices]
        )

        # ROC-AUC is undefined if a bootstrap sample
        # contains only one class.
        if len(np.unique(sampled_y)) < 2:
            continue

        metric_rows.append(
            calculate_metrics(
                sampled_y,
                sampled_probabilities,
                threshold=threshold,
            )
        )

    bootstrap_table = pd.DataFrame(metric_rows)

    interval_metrics = [
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

    intervals = {}

    for metric in interval_metrics:
        if metric not in bootstrap_table:
            continue

        values = bootstrap_table[metric].dropna()

        if len(values) == 0:
            continue

        intervals[f"{metric}_ci_lower"] = np.percentile(
            values,
            2.5,
        )

        intervals[f"{metric}_ci_upper"] = np.percentile(
            values,
            97.5,
        )

    intervals["bootstrap_replicates"] = len(
        bootstrap_table
    )

    return intervals

# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold: float = 0.40,
) -> dict:
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities)
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
        "threshold": threshold,
        "n": int(len(y_true)),
    }

# ============================================================
# Main procedure
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
        default=Path("outputs"),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
    )

    parser.add_argument(
        "--n-iter",
        type=int,
        default=N_ITER_SEARCH,
    )

    args = parser.parse_args()

    create_directories(args.output_root)
    configure_logging(args.output_root)

    logging.info("Starting XGBoost training")
    logging.info("Data root: %s", args.data_root)
    logging.info("Output root: %s", args.output_root)
    logging.info("Random seed: %d", args.seed)
    logging.info("Search iterations: %d", args.n_iter)

    all_fold_results = []
    all_development_rows = []
    all_best_parameters = []

    for checkpoint in CHECKPOINTS:
        start_time = time.time()

        logging.info("Starting checkpoint %s", checkpoint)

        development, validation, testing = (
            load_checkpoint_data(
                args.data_root,
                checkpoint,
            )
        )

        feature_columns = get_feature_columns(
            development
        )

        # This is deliberately calculated from development data only.
        negative_cases = int(
            (development[TARGET_COLUMN] == 0).sum()
        )
        positive_cases = int(
            (development[TARGET_COLUMN] == 1).sum()
        )

        scale_pos_weight = (
            negative_cases / positive_cases
            if positive_cases > 0
            else 1.0
        )

        logging.info(
            "%s scale_pos_weight=%f, calculated from "
            "development data only",
            checkpoint,
            scale_pos_weight,
        )

        parameter_list = list(
            ParameterSampler(
                XGBOOST_PARAMETER_DISTRIBUTIONS,
                n_iter=args.n_iter,
                random_state=args.seed,
            )
        )

        search_history = []
        warning_rows = []

        # ----------------------------------------------------
        # Forward-semester parameter evaluation
        # ----------------------------------------------------

        for parameter_number, parameters in enumerate(
            parameter_list,
            start=1,
        ):
            fold_scores = []

            for fold_definition in FORWARD_FOLDS:
                fold_number = fold_definition["fold"]
                training_semesters = (
                    fold_definition["training_semesters"]
                )
                validation_semester = (
                    fold_definition["validation_semester"]
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

                X_fold_validation = fold_validation[
                    feature_columns
                ]
                y_fold_validation = fold_validation[
                    TARGET_COLUMN
                ]

                # Recalculate the imbalance ratio from the
                # fold's training data only.
                fold_negative_cases = int(
                    (y_train == 0).sum()
                )
                fold_positive_cases = int(
                    (y_train == 1).sum()
                )

                fold_scale_pos_weight = (
                    fold_negative_cases
                    / fold_positive_cases
                    if fold_positive_cases > 0
                    else 1.0
                )

                pipeline = build_xgboost_pipeline(
                    fold_training,
                    feature_columns,
                    parameters,
                    fold_scale_pos_weight,
                    args.seed,
                )

                with warnings.catch_warnings(
                    record=True
                ) as caught_warnings:
                    warnings.simplefilter("always")

                    try:
                        pipeline.fit(X_train, y_train)

                        probabilities = (
                            pipeline.predict_proba(
                                X_fold_validation
                            )[:, 1]
                        )

                        fold_metrics = calculate_metrics(
                            y_fold_validation,
                            probabilities,
                            threshold=0.40,
                        )

                        fit_status = "completed"
                        error_message = ""

                    except Exception as error:
                        fold_metrics = {
                            "roc_auc": np.nan,
                            "pr_auc": np.nan,
                            "recall": np.nan,
                            "precision": np.nan,
                            "brier_score": np.nan,
                        }

                        fit_status = "failed"
                        error_message = repr(error)

                for caught_warning in caught_warnings:
                    warning_rows.append(
                        {
                            "checkpoint": checkpoint,
                            "model": "xgboost",
                            "fold": fold_number,
                            "parameter_number": parameter_number,
                            "warning_category": (
                                caught_warning.category.__name__
                            ),
                            "warning_message": str(
                                caught_warning.message
                            ),
                        }
                    )

                fold_result = {
                    "checkpoint": checkpoint,
                    "model": "xgboost",
                    "fold": fold_number,
                    "training_semesters": "|".join(
                        training_semesters
                    ),
                    "validation_semester": validation_semester,
                    "parameter_number": parameter_number,
                    "parameters": json.dumps(
                        parameters,
                        default=str,
                    ),
                    "fold_scale_pos_weight": (
                        fold_scale_pos_weight
                    ),
                    "roc_auc": fold_metrics["roc_auc"],
                    "pr_auc": fold_metrics["pr_auc"],
                    "recall": fold_metrics["recall"],
                    "precision": fold_metrics["precision"],
                    "brier_score": fold_metrics[
                        "brier_score"
                    ],
                    "fit_status": fit_status,
                    "error_message": error_message,
                }

                all_fold_results.append(fold_result)

                fold_scores.append(fold_metrics)

            completed_pr_scores = [
                result["pr_auc"]
                for result in fold_scores
                if not np.isnan(result["pr_auc"])
            ]

            roc_scores = [
                result["roc_auc"]
                for result in fold_scores
                if not np.isnan(result["roc_auc"])
            ]

            recall_scores = [
                result["recall"]
                for result in fold_scores
                if not np.isnan(result["recall"])
            ]

            precision_scores = [
                result["precision"]
                for result in fold_scores
                if not np.isnan(result["precision"])
            ]

            brier_scores = [
                result["brier_score"]
                for result in fold_scores
                if not np.isnan(result["brier_score"])
            ]

            search_history.append(
                {
                    "checkpoint": checkpoint,
                    "model": "xgboost",
                    "parameter_number": parameter_number,
                    "parameters": json.dumps(
                        parameters,
                        default=str,
                    ),
                    "development_scale_pos_weight": (
                        scale_pos_weight
                    ),
                    "mean_roc_auc": np.mean(
                        roc_scores
                    ),
                    "std_roc_auc": np.std(
                        roc_scores,
                        ddof=1,
                    )
                    if len(roc_scores) > 1
                    else 0.0,
                    "mean_pr_auc": np.mean(
                        completed_pr_scores
                    ),
                    "std_pr_auc": np.std(
                        completed_pr_scores,
                        ddof=1,
                    )
                    if len(completed_pr_scores) > 1
                    else 0.0,
                    "mean_recall": np.mean(
                        recall_scores
                    ),
                    "mean_precision": np.mean(
                        precision_scores
                    ),
                    "mean_brier_score": np.mean(
                        brier_scores
                    ),
                    "number_of_folds": len(
                        completed_pr_scores
                    ),
                }
            )

        search_history_df = pd.DataFrame(
            search_history
        )

        search_history_df = search_history_df.sort_values(
            [
                "mean_pr_auc",
                "mean_roc_auc",
            ],
            ascending=False,
        )

        search_history_df.to_csv(
            args.output_root
            / "tables"
            / f"{checkpoint}_xgboost_search_history.csv",
            index=False,
        )

        best_row = search_history_df.iloc[0]

        best_parameter_number = int(
            best_row["parameter_number"]
        )

        best_parameters = next(
            parameters
            for number, parameters in enumerate(
                parameter_list,
                start=1,
            )
            if number == best_parameter_number
        )

        all_best_parameters.append(
            {
                "checkpoint": checkpoint,
                "model": "xgboost",
                "best_parameter_number": (
                    best_parameter_number
                ),
                "best_parameters": json.dumps(
                    best_parameters,
                    default=str,
                ),
                "development_scale_pos_weight": (
                    scale_pos_weight
                ),
                "mean_roc_auc": best_row[
                    "mean_roc_auc"
                ],
                "std_roc_auc": best_row[
                    "std_roc_auc"
                ],
                "mean_pr_auc": best_row[
                    "mean_pr_auc"
                ],
                "std_pr_auc": best_row[
                    "std_pr_auc"
                ],
                "mean_recall": best_row[
                    "mean_recall"
                ],
                "mean_precision": best_row[
                    "mean_precision"
                ],
                "mean_brier_score": best_row[
                    "mean_brier_score"
                ],
            }
        )

        # ----------------------------------------------------
        # Save best fold results
        # ----------------------------------------------------

        all_fold_df = pd.DataFrame(
            all_fold_results
        )

        best_fold_df = all_fold_df[
            (
                all_fold_df["checkpoint"]
                == checkpoint
            )
            & (
                all_fold_df["parameter_number"]
                == best_parameter_number
            )
        ]

        best_fold_df.to_csv(
            args.output_root
            / "tables"
            / f"{checkpoint}_xgboost_best_fold_results.csv",
            index=False,
        )

        # ----------------------------------------------------
        # Refit final model on development data
        # ----------------------------------------------------

        final_pipeline = build_xgboost_pipeline(
            development,
            feature_columns,
            best_parameters,
            scale_pos_weight,
            args.seed,
        )

        with warnings.catch_warnings(
            record=True
        ) as caught_warnings:
            warnings.simplefilter("always")

            final_pipeline.fit(
                development[feature_columns],
                development[TARGET_COLUMN],
            )

        for caught_warning in caught_warnings:
            warning_rows.append(
                {
                    "checkpoint": checkpoint,
                    "model": "xgboost",
                    "fold": "final_refit",
                    "parameter_number": "selected",
                    "warning_category": (
                        caught_warning.category.__name__
                    ),
                    "warning_message": str(
                        caught_warning.message
                    ),
                }
            )

        model_path = (
            args.output_root
            / "models"
            / f"{checkpoint}_xgboost.joblib"
        )

        joblib.dump(
            final_pipeline,
            model_path,
        )

        # ----------------------------------------------------
        # Validation and testing predictions
        # ----------------------------------------------------

        validation_probability = (
            final_pipeline.predict_proba(
                validation[feature_columns]
            )[:, 1]
        )

        testing_probability = (
            final_pipeline.predict_proba(
                testing[feature_columns]
            )[:, 1]
        )

        validation_metrics = calculate_metrics(
            validation[TARGET_COLUMN],
            validation_probability,
            threshold=0.40,
        )

        testing_metrics = calculate_metrics(
            testing[TARGET_COLUMN],
            testing_probability,
            threshold=0.40,
        )

        testing_intervals = (
            bootstrap_metric_intervals(
                testing[TARGET_COLUMN],
                testing_probability,
                threshold=0.40,
                n_bootstrap=2000,
                seed=args.seed,
            )
        )

        testing_metrics.update(
            testing_intervals
        )


        all_development_rows.append(
            {
                "checkpoint": checkpoint,
                "model": "xgboost",
                "mean_roc_auc": best_row[
                    "mean_roc_auc"
                ],
                "mean_pr_auc": best_row[
                    "mean_pr_auc"
                ],
                "mean_recall": best_row[
                    "mean_recall"
                ],
                "mean_precision": best_row[
                    "mean_precision"
                ],
                "mean_brier_score": best_row[
                    "mean_brier_score"
                ],
                "fold_variability_pr_auc": best_row[
                    "std_pr_auc"
                ],
                "fold_variability_roc_auc": best_row[
                    "std_roc_auc"
                ],
            }
        )

        for split_name, df, probabilities, metrics in [
            (
                "validation",
                validation,
                validation_probability,
                validation_metrics,
            ),
            (
                "testing",
                testing,
                testing_probability,
                testing_metrics,
            ),
        ]:
            identifiers = df.get(
                IDENTIFIER_COLUMN,
                pd.Series(
                    range(len(df)),
                    index=df.index,
                ),
            )

            prediction_table = pd.DataFrame(
                {
                    "unique_identifier": identifiers,
                    "semester": df["semester"],
                    "true_label": df[TARGET_COLUMN],
                    "predicted_probability": probabilities,
                    "predicted_class_0_40": (
                        probabilities >= 0.40
                    ).astype(int),
                    "model_name": "xgboost",
                    "checkpoint": checkpoint,
                }
            )

            prediction_table.to_csv(
                args.output_root
                / "predictions"
                / f"{checkpoint}_xgboost_{split_name}.csv",
                index=False,
            )

            metrics_row = {
                "checkpoint": checkpoint,
                "model": "xgboost",
                "split": split_name,
                "semester": df["semester"].iloc[0],
                **metrics,
            }

            pd.DataFrame([metrics_row]).to_csv(
                args.output_root
                / "tables"
                / f"{checkpoint}_xgboost_{split_name}_metrics.csv",
                index=False,
            )

        warning_columns = [
            "checkpoint",
            "model",
            "fold",
            "parameter_number",
            "warning_category",
            "warning_message",
        ]

        pd.DataFrame(
            warning_rows,
            columns=warning_columns,
        ).to_csv(
            args.output_root
            / "tables"
            / f"{checkpoint}_xgboost_warnings.csv",
            index=False,
        )

        elapsed = time.time() - start_time

        logging.info(
            "Completed %s in %.2f seconds. Best parameters: %s",
            checkpoint,
            elapsed,
            best_parameters,
        )

        print(
            f"Completed {checkpoint}: "
            f"mean forward PR-AUC="
            f"{best_row['mean_pr_auc']:.4f}"
        )

    # --------------------------------------------------------
    # Combined output tables
    # --------------------------------------------------------

    all_fold_df = pd.DataFrame(
        all_fold_results
    )

    all_fold_df.to_csv(
        args.output_root
        / "tables"
        / "xgboost_all_fold_results.csv",
        index=False,
    )

    pd.DataFrame(all_best_parameters).to_csv(
        args.output_root
        / "tables"
        / "xgboost_best_parameters_all_checkpoints.csv",
        index=False,
    )

    development_table = pd.DataFrame(
        all_development_rows
    )

    development_table.to_csv(
        args.output_root
        / "tables"
        / "xgboost_development_results.csv",
        index=False,
    )

    all_warning_files = list(
        (args.output_root / "tables").glob(
            "*_xgboost_warnings.csv"
        )
    )

    warning_columns = [
        "checkpoint",
        "model",
        "fold",
        "parameter_number",
        "warning_category",
        "warning_message",
    ]

    warning_tables = []

    for warning_file in all_warning_files:
        # Skip genuinely empty files produced when no warnings occurred.
        if warning_file.stat().st_size == 0:
            continue

        try:
            warning_table = pd.read_csv(warning_file)

            if not warning_table.empty:
                warning_tables.append(warning_table)

        except pd.errors.EmptyDataError:
            continue

    if warning_tables:
        combined_warnings = pd.concat(
            warning_tables,
            ignore_index=True,
        )
    else:
        combined_warnings = pd.DataFrame(
            columns=warning_columns
        )

    combined_warnings.to_csv(
        args.output_root
        / "tables"
        / "xgboost_all_warnings.csv",
        index=False,
    )

    warning_tables = []

    for warning_file in all_warning_files:
        warning_table = pd.read_csv(
            warning_file
        )

        if not warning_table.empty:
            warning_tables.append(warning_table)

    if warning_tables:
        pd.concat(
            warning_tables,
            ignore_index=True,
        ).to_csv(
            args.output_root
            / "tables"
            / "xgboost_all_warnings.csv",
            index=False,
        )
    else:
        pd.DataFrame(
            columns=[
                "checkpoint",
                "model",
                "fold",
                "parameter_number",
                "warning_category",
                "warning_message",
            ]
        ).to_csv(
            args.output_root
            / "tables"
            / "xgboost_all_warnings.csv",
            index=False,
        )

    print("\nXGBoost training completed.")
    print("Models saved to:", args.output_root / "models")
    print("Tables saved to:", args.output_root / "tables")
    print("Predictions saved to:", args.output_root / "predictions")
    print("Logs saved to:", args.output_root / "logs")


if __name__ == "__main__":
    main()