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
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.modelling_common import (
    load_checkpoint_data,
    get_feature_columns,
)


# ============================================================
# Configuration
# ============================================================

CHECKPOINTS = ["week3", "week6", "week8", "week12"]

TARGET_COLUMN = "at_risk"
IDENTIFIER_COLUMN = "unique_identifier"

# Reference classification threshold for every reported fold metric.
# This matches the threshold used by the Random Forest and XGBoost
# development scripts, so that recall, precision, specificity and
# Brier score are directly comparable across the three models.
REFERENCE_THRESHOLD = 0.40

TRAINING_SEMESTER_COUNTS = {
    "2024 S1": 141,
    "2024 S2": 68,
    "2025 S1": 170,
}

# Forward-semester folds.
# 2025 S2 is retained as the separate validation semester.
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

HYPERPARAMETER_GRID = [
    {"C": 0.01, "class_weight": None},
    {"C": 0.01, "class_weight": "balanced"},
    {"C": 0.1, "class_weight": None},
    {"C": 0.1, "class_weight": "balanced"},
    {"C": 1.0, "class_weight": None},
    {"C": 1.0, "class_weight": "balanced"},
    {"C": 10.0, "class_weight": None},
    {"C": 10.0, "class_weight": "balanced"},
    {"C": 100.0, "class_weight": None},
    {"C": 100.0, "class_weight": "balanced"},
]

# The training semester label written to the testing split. The data
# are the second half of 2025 S2; the earlier design intended a 2026
# semester, and this residual label is documented in Chapter 5.
TESTING_SPLIT_LABEL = "2026 S1"


# ============================================================
# Folder and logging utilities
# ============================================================

def create_output_directories(output_root: Path) -> None:
    directories = [
        output_root / "models",
        output_root / "tables",
        output_root / "predictions",
        output_root / "logs",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def configure_logging(output_root: Path) -> None:
    log_file = (
        output_root / "logs" / "logistic_regression_execution.log"
    )

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


# ============================================================
# File and column handling
# ============================================================

def locate_file(
    data_root: Path,
    checkpoint: str,
    split: str,
) -> Path:
    """
    Locate engineered-feature files created by the Week 3, 6, 8
    and 12 feature-engineering scripts.

    Expected examples:

        Data/week3/week3trainingengineeredfeatures.csv
        Data/week3/week3validationengineeredfeatures.csv
        Data/week3/week3testingengineeredfeatures.csv
    """

    data_root = Path(data_root)

    # Supports both "week3" and "week_3".
    week_number = (
        checkpoint
        .replace("week", "")
        .replace("_", "")
    )

    folder_candidates = [
        data_root / f"week{week_number}",
        data_root / f"week_{week_number}",
        data_root / checkpoint,
    ]

    filename_candidates = {
        "training": [
            f"week{week_number}trainingengineeredfeatures.csv",
            f"week_{week_number}trainingengineeredfeatures.csv",
            f"week{week_number}_training_engineered_features.csv",
            f"week_{week_number}_training_engineered_features.csv",
            f"{checkpoint}trainingengineeredfeatures.csv",
            f"{checkpoint}_training_engineered_features.csv",
            "training.csv",
        ],
        "validation": [
            f"week{week_number}validationengineeredfeatures.csv",
            f"week_{week_number}validationengineeredfeatures.csv",
            f"week{week_number}_validation_engineered_features.csv",
            f"week_{week_number}_validation_engineered_features.csv",
            f"{checkpoint}validationengineeredfeatures.csv",
            f"{checkpoint}_validation_engineered_features.csv",
            "validation.csv",
        ],
        "testing": [
            f"week{week_number}testingengineeredfeatures.csv",
            f"week_{week_number}testingengineeredfeatures.csv",
            f"week{week_number}_testing_engineered_features.csv",
            f"week_{week_number}_testing_engineered_features.csv",
            f"{checkpoint}testingengineeredfeatures.csv",
            f"{checkpoint}_testing_engineered_features.csv",
            "testing.csv",
            "test.csv",
        ],
    }

    if split not in filename_candidates:
        raise ValueError(
            f"Unknown split '{split}'. "
            "Use training, validation or testing."
        )

    for folder in folder_candidates:
        if not folder.exists():
            continue

        for filename in filename_candidates[split]:
            candidate = folder / filename

            if candidate.exists():
                logging.info(
                    "Located %s %s file: %s",
                    checkpoint,
                    split,
                    candidate,
                )

                return candidate

    attempted_folders = "\n".join(
        f"  {folder}" for folder in folder_candidates
    )

    attempted_filenames = "\n".join(
        f"  {filename}"
        for filename in filename_candidates[split]
    )

    raise FileNotFoundError(
        f"\nCould not find the {split} file for {checkpoint}.\n"
        f"Searched folders:\n{attempted_folders}\n\n"
        f"Tried filenames:\n{attempted_filenames}"
    )


def normalise_column_name(column_name: str) -> str:
    text = str(column_name).strip().lower()
    text = text.replace(" ", "_")
    text = text.replace("-", "_")
    text = text.replace("/", "_")
    text = text.replace("(", "")
    text = text.replace(")", "")
    return text


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_dictionary = {}

    for column in df.columns:
        normalised = normalise_column_name(column)

        if normalised in {
            "at_risk",
            "atrisk",
            "risk",
            "target",
            "label",
        }:
            rename_dictionary[column] = TARGET_COLUMN

        elif normalised in {
            "unique_identifier",
            "student_id",
            "student_enrolment_id",
            "id",
        }:
            rename_dictionary[column] = IDENTIFIER_COLUMN

        else:
            rename_dictionary[column] = normalised

    result = df.rename(columns=rename_dictionary).copy()

    if TARGET_COLUMN not in result.columns:
        raise ValueError(
            "Target column was not found. Expected a column named "
            "'at risk' or 'at_risk'."
        )

    return result


def encode_target(series: pd.Series) -> pd.Series:
    """
    Converts common binary target formats into 0 and 1.
    """

    if pd.api.types.is_numeric_dtype(series):
        numeric_values = pd.to_numeric(series, errors="coerce")
        unique_values = set(numeric_values.dropna().unique())

        if unique_values.issubset({0, 1}):
            return numeric_values

    mapping = {
        "yes": 1,
        "y": 1,
        "true": 1,
        "1": 1,
        "at risk": 1,
        "at_risk": 1,
        "no": 0,
        "n": 0,
        "false": 0,
        "0": 0,
        "not at risk": 0,
        "not_at_risk": 0,
    }

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )


def read_data_file(
    data_root: Path,
    checkpoint: str,
    split: str,
) -> pd.DataFrame:
    file_path = locate_file(data_root, checkpoint, split)

    logging.info("Reading %s", file_path)

    df = pd.read_csv(file_path)
    df = normalise_columns(df)
    df[TARGET_COLUMN] = encode_target(df[TARGET_COLUMN])

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"Missing or unrecognised target values in {file_path}"
        )

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    return df


# ============================================================
# Semester assignment
# ============================================================

def add_training_semesters(df: pd.DataFrame) -> pd.DataFrame:
    expected_rows = sum(TRAINING_SEMESTER_COUNTS.values())

    if len(df) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} training data rows, "
            f"but found {len(df)} rows."
        )

    semester_labels = []

    for semester, count in TRAINING_SEMESTER_COUNTS.items():
        semester_labels.extend([semester] * count)

    result = df.copy()
    result["semester"] = semester_labels

    return result


# ============================================================
# Preprocessing and model creation
# ============================================================

def build_preprocessor(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> ColumnTransformer:

    X = df[feature_columns]

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
            ),
            (
                "scaler",
                StandardScaler(),
            ),
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

    preprocessor = ColumnTransformer(
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

    return preprocessor


def build_logistic_pipeline(
    preprocessor: ColumnTransformer,
    parameters: dict,
    random_seed: int,
) -> Pipeline:

    model = LogisticRegression(
        C=parameters["C"],
        class_weight=parameters["class_weight"],
        solver="lbfgs",
        max_iter=5000,
        random_state=random_seed,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


# ============================================================
# Metrics
# ============================================================

def calculate_binary_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float = 0.40,
) -> dict:

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "roc_auc": (
            roc_auc_score(y_true, probabilities)
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
        "specificity": (
            tn / (tn + fp)
            if (tn + fp) > 0
            else np.nan
        ),
        "negative_predictive_value": (
            tn / (tn + fn)
            if (tn + fn) > 0
            else np.nan
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            predictions,
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
        "percentage_flagged": float(predictions.mean() * 100),
        "threshold": threshold,
        "n": int(len(y_true)),
    }


METRIC_DEFAULTS = {
    "roc_auc": np.nan,
    "pr_auc": np.nan,
    "recall": np.nan,
    "precision": np.nan,
    "specificity": np.nan,
    "negative_predictive_value": np.nan,
    "f1": np.nan,
    "balanced_accuracy": np.nan,
    "brier_score": np.nan,
    "true_negative": np.nan,
    "false_positive": np.nan,
    "false_negative": np.nan,
    "true_positive": np.nan,
    "number_flagged": np.nan,
    "percentage_flagged": np.nan,
}


# ============================================================
# Argument parsing
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate Logistic Regression models for "
            "checkpoint-based at-risk prediction."
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
        default=42,
    )

    return parser.parse_args()


# ============================================================
# Main procedure
# ============================================================

def main() -> None:
    args = parse_arguments()

    create_output_directories(args.output_root)
    configure_logging(args.output_root)

    logging.info("Starting Logistic Regression training")
    logging.info("Data root: %s", args.data_root)
    logging.info("Output root: %s", args.output_root)
    logging.info("Reference threshold: %s", REFERENCE_THRESHOLD)

    all_fold_results = []
    all_best_parameters = []
    all_validation_metrics = []
    all_testing_metrics = []
    all_warnings = []

    for checkpoint in CHECKPOINTS:
        start_time = time.time()

        logging.info("Starting checkpoint: %s", checkpoint)

        training, validation, testing = load_checkpoint_data(
            args.data_root,
            checkpoint,
        )

        feature_columns = get_feature_columns(training)

        logging.info(
            "%s: development rows=%d, validation rows=%d, "
            "testing rows=%d",
            checkpoint,
            len(training),
            len(validation),
            len(testing),
        )

        logging.info(
            "%s: predictors=%d, target prevalence=%.4f",
            checkpoint,
            len(feature_columns),
            training[TARGET_COLUMN].mean(),
        )

        checkpoint_fold_results = []

        # ----------------------------------------------------
        # Forward-semester validation
        # ----------------------------------------------------

        for fold_definition in FORWARD_FOLDS:
            fold_number = fold_definition["fold"]
            train_semesters = fold_definition["training_semesters"]
            validation_semester = (
                fold_definition["validation_semester"]
            )

            fold_training = training[
                training["semester"].isin(train_semesters)
            ].copy()

            fold_validation = training[
                training["semester"] == validation_semester
            ].copy()

            X_train = fold_training[feature_columns]
            y_train = fold_training[TARGET_COLUMN]

            X_fold_validation = fold_validation[feature_columns]
            y_fold_validation = fold_validation[TARGET_COLUMN]

            logging.info(
                "%s fold %d: train semesters=%s, validation=%s, "
                "training rows=%d, validation rows=%d",
                checkpoint,
                fold_number,
                train_semesters,
                validation_semester,
                len(fold_training),
                len(fold_validation),
            )

            if y_train.nunique() < 2:
                logging.warning(
                    "%s fold %d skipped because training data "
                    "contains only one target class",
                    checkpoint,
                    fold_number,
                )
                continue

            if y_fold_validation.nunique() < 2:
                logging.warning(
                    "%s fold %d validation data contains only "
                    "one class",
                    checkpoint,
                    fold_number,
                )

            for parameter_number, parameters in enumerate(
                HYPERPARAMETER_GRID,
                start=1,
            ):
                pipeline = build_logistic_pipeline(
                    build_preprocessor(
                        fold_training,
                        feature_columns,
                    ),
                    parameters,
                    args.seed,
                )

                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.simplefilter("always")

                    try:
                        pipeline.fit(X_train, y_train)

                        probabilities = pipeline.predict_proba(
                            X_fold_validation
                        )[:, 1]

                        # All fold metrics are computed at the shared
                        # reference threshold, so that recall,
                        # precision, specificity and Brier score are
                        # comparable with the ensemble fold results.
                        fold_metrics = calculate_binary_metrics(
                            y_fold_validation,
                            probabilities,
                            threshold=REFERENCE_THRESHOLD,
                        )

                        fit_status = "completed"
                        error_message = ""

                    except Exception as error:
                        fold_metrics = {}
                        fit_status = "failed"
                        error_message = repr(error)

                fold_metrics = {
                    **METRIC_DEFAULTS,
                    **fold_metrics,
                }

                for caught_warning in caught_warnings:
                    warning_record = {
                        "checkpoint": checkpoint,
                        "fold": fold_number,
                        "parameter_number": parameter_number,
                        "warning_category": (
                            caught_warning.category.__name__
                        ),
                        "warning_message": str(
                            caught_warning.message
                        ),
                    }

                    all_warnings.append(warning_record)

                    if issubclass(
                        caught_warning.category,
                        ConvergenceWarning,
                    ):
                        logging.warning(
                            "%s fold %d parameter %d convergence "
                            "warning: %s",
                            checkpoint,
                            fold_number,
                            parameter_number,
                            caught_warning.message,
                        )

                result = {
                    "checkpoint": checkpoint,
                    "model": "logistic_regression",
                    "fold": fold_number,
                    "training_semesters": "|".join(
                        train_semesters
                    ),
                    "validation_semester": validation_semester,
                    "parameter_number": parameter_number,
                    "C": parameters["C"],
                    "class_weight": str(
                        parameters["class_weight"]
                    ),
                    "roc_auc": fold_metrics["roc_auc"],
                    "pr_auc": fold_metrics["pr_auc"],
                    "recall": fold_metrics["recall"],
                    "precision": fold_metrics["precision"],
                    "specificity": fold_metrics["specificity"],
                    "negative_predictive_value": (
                        fold_metrics["negative_predictive_value"]
                    ),
                    "f1": fold_metrics["f1"],
                    "balanced_accuracy": (
                        fold_metrics["balanced_accuracy"]
                    ),
                    "brier_score": fold_metrics["brier_score"],
                    "true_negative": fold_metrics["true_negative"],
                    "false_positive": (
                        fold_metrics["false_positive"]
                    ),
                    "false_negative": (
                        fold_metrics["false_negative"]
                    ),
                    "true_positive": fold_metrics["true_positive"],
                    "number_flagged": (
                        fold_metrics["number_flagged"]
                    ),
                    "percentage_flagged": (
                        fold_metrics["percentage_flagged"]
                    ),
                    "threshold": REFERENCE_THRESHOLD,
                    "fit_status": fit_status,
                    "error_message": error_message,
                }

                checkpoint_fold_results.append(result)
                all_fold_results.append(result)

        # ----------------------------------------------------
        # Select parameters using mean forward-validation PR-AUC
        # ----------------------------------------------------

        fold_table = pd.DataFrame(checkpoint_fold_results)

        if fold_table.empty:
            raise RuntimeError(
                f"No successful Logistic Regression folds for "
                f"{checkpoint}."
            )

        successful = fold_table[
            fold_table["fit_status"] == "completed"
        ].copy()

        parameter_summary = (
            successful.groupby(
                [
                    "parameter_number",
                    "C",
                    "class_weight",
                ],
                dropna=False,
            )
            .agg(
                mean_pr_auc=("pr_auc", "mean"),
                std_pr_auc=("pr_auc", "std"),
                mean_roc_auc=("roc_auc", "mean"),
                std_roc_auc=("roc_auc", "std"),
                mean_recall=("recall", "mean"),
                std_recall=("recall", "std"),
                mean_precision=("precision", "mean"),
                std_precision=("precision", "std"),
                mean_specificity=("specificity", "mean"),
                std_specificity=("specificity", "std"),
                mean_f1=("f1", "mean"),
                mean_balanced_accuracy=(
                    "balanced_accuracy",
                    "mean",
                ),
                mean_brier_score=("brier_score", "mean"),
                std_brier_score=("brier_score", "std"),
                number_of_folds=("fold", "nunique"),
            )
            .reset_index()
        )

        # Selection remains on PR-AUC then ROC-AUC. Adding the
        # classification metrics above collects more evidence
        # without changing which parameters are selected.
        parameter_summary = parameter_summary.sort_values(
            ["mean_pr_auc", "mean_roc_auc"],
            ascending=False,
        )

        best_row = parameter_summary.iloc[0]

        best_parameters = {
            "C": float(best_row["C"]),
            "class_weight": (
                None
                if str(best_row["class_weight"]) == "None"
                else "balanced"
            ),
        }

        best_fold_results = successful[
            successful["parameter_number"]
            == int(best_row["parameter_number"])
        ].copy()

        best_fold_results.to_csv(
            args.output_root
            / "tables"
            / f"{checkpoint}_logistic_best_fold_results.csv",
            index=False,
        )

        parameter_summary.to_csv(
            args.output_root
            / "tables"
            / f"{checkpoint}_logistic_parameter_summary.csv",
            index=False,
        )

        all_best_parameters.append(
            {
                "checkpoint": checkpoint,
                "model": "logistic_regression",
                **best_parameters,
                "mean_forward_pr_auc": best_row["mean_pr_auc"],
                "std_forward_pr_auc": best_row["std_pr_auc"],
                "mean_forward_roc_auc": best_row["mean_roc_auc"],
                "std_forward_roc_auc": best_row["std_roc_auc"],
                "mean_forward_recall": best_row["mean_recall"],
                "std_forward_recall": best_row["std_recall"],
                "mean_forward_precision": (
                    best_row["mean_precision"]
                ),
                "std_forward_precision": (
                    best_row["std_precision"]
                ),
                "mean_forward_specificity": (
                    best_row["mean_specificity"]
                ),
                "mean_forward_f1": best_row["mean_f1"],
                "mean_forward_balanced_accuracy": (
                    best_row["mean_balanced_accuracy"]
                ),
                "mean_forward_brier_score": (
                    best_row["mean_brier_score"]
                ),
                "std_forward_brier_score": (
                    best_row["std_brier_score"]
                ),
                "number_of_folds": best_row["number_of_folds"],
                "reference_threshold": REFERENCE_THRESHOLD,
            }
        )

        # ----------------------------------------------------
        # Refit final development model on all 2024 S1-2025 S1
        # ----------------------------------------------------

        X_development = training[feature_columns]
        y_development = training[TARGET_COLUMN]

        final_pipeline = build_logistic_pipeline(
            build_preprocessor(
                training,
                feature_columns,
            ),
            best_parameters,
            args.seed,
        )

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            final_pipeline.fit(X_development, y_development)

        for caught_warning in caught_warnings:
            all_warnings.append(
                {
                    "checkpoint": checkpoint,
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
            / f"{checkpoint}_logistic_regression.joblib"
        )

        joblib.dump(final_pipeline, model_path)

        # ----------------------------------------------------
        # Coefficient table
        # ----------------------------------------------------

        fitted_preprocessor = final_pipeline.named_steps[
            "preprocess"
        ]
        fitted_model = final_pipeline.named_steps["model"]

        feature_names = (
            fitted_preprocessor.get_feature_names_out()
        )

        coefficients = fitted_model.coef_[0]

        coefficient_table = pd.DataFrame(
            {
                "checkpoint": checkpoint,
                "model": "logistic_regression",
                "feature": feature_names,
                "coefficient": coefficients,
                "absolute_coefficient": np.abs(coefficients),
                "sign": np.where(
                    coefficients >= 0,
                    "positive",
                    "negative",
                ),
                "odds_ratio": np.exp(coefficients),
            }
        ).sort_values(
            "absolute_coefficient",
            ascending=False,
        )

        coefficient_table.to_csv(
            args.output_root
            / "tables"
            / f"{checkpoint}_logistic_coefficients.csv",
            index=False,
        )

        # ----------------------------------------------------
        # Validation and final-test predictions
        # ----------------------------------------------------

        validation_probabilities = final_pipeline.predict_proba(
            validation[feature_columns]
        )[:, 1]

        testing_probabilities = final_pipeline.predict_proba(
            testing[feature_columns]
        )[:, 1]

        validation_predictions = pd.DataFrame(
            {
                "unique_identifier": validation.get(
                    IDENTIFIER_COLUMN,
                    pd.Series(
                        range(len(validation)),
                        index=validation.index,
                    ),
                ),
                "semester": "2025 S2",
                "true_label": validation[TARGET_COLUMN],
                "predicted_probability": validation_probabilities,
                "predicted_class_0_40": (
                    validation_probabilities >= REFERENCE_THRESHOLD
                ).astype(int),
                "model_name": "logistic_regression",
                "checkpoint": checkpoint,
            }
        )

        testing_predictions = pd.DataFrame(
            {
                "unique_identifier": testing.get(
                    IDENTIFIER_COLUMN,
                    pd.Series(
                        range(len(testing)),
                        index=testing.index,
                    ),
                ),
                "semester": TESTING_SPLIT_LABEL,
                "true_label": testing[TARGET_COLUMN],
                "predicted_probability": testing_probabilities,
                "predicted_class_0_40": (
                    testing_probabilities >= REFERENCE_THRESHOLD
                ).astype(int),
                "model_name": "logistic_regression",
                "checkpoint": checkpoint,
            }
        )

        validation_predictions.to_csv(
            args.output_root
            / "predictions"
            / f"{checkpoint}_logistic_regression_validation.csv",
            index=False,
        )

        testing_predictions.to_csv(
            args.output_root
            / "predictions"
            / f"{checkpoint}_logistic_regression_testing.csv",
            index=False,
        )

        # ----------------------------------------------------
        # Metrics at reference threshold 0.40
        # ----------------------------------------------------

        validation_metrics = calculate_binary_metrics(
            validation[TARGET_COLUMN],
            validation_probabilities,
            threshold=REFERENCE_THRESHOLD,
        )

        validation_metrics.update(
            {
                "checkpoint": checkpoint,
                "model": "logistic_regression",
                "model_label": "Logistic Regression",
                "split": "2025 S2 validation",
            }
        )

        testing_metrics = calculate_binary_metrics(
            testing[TARGET_COLUMN],
            testing_probabilities,
            threshold=REFERENCE_THRESHOLD,
        )

        testing_metrics.update(
            {
                "checkpoint": checkpoint,
                "model": "logistic_regression",
                "model_label": "Logistic Regression",
                "split": f"{TESTING_SPLIT_LABEL} testing",
            }
        )

        all_validation_metrics.append(validation_metrics)
        all_testing_metrics.append(testing_metrics)

        elapsed = time.time() - start_time

        logging.info(
            "Completed %s in %.2f seconds. Best parameters: %s",
            checkpoint,
            elapsed,
            best_parameters,
        )

        print(
            f"Completed {checkpoint}: "
            f"best mean forward PR-AUC="
            f"{best_row['mean_pr_auc']:.4f}, "
            f"mean recall={best_row['mean_recall']:.4f}, "
            f"mean precision={best_row['mean_precision']:.4f}"
        )

    # --------------------------------------------------------
    # Save combined outputs
    # --------------------------------------------------------

    pd.DataFrame(all_fold_results).to_csv(
        args.output_root
        / "tables"
        / "logistic_regression_all_fold_results.csv",
        index=False,
    )

    pd.DataFrame(all_best_parameters).to_csv(
        args.output_root
        / "tables"
        / "logistic_regression_best_parameters_all_checkpoints.csv",
        index=False,
    )

    pd.DataFrame(all_validation_metrics).to_csv(
        args.output_root
        / "tables"
        / "logistic_regression_validation_metrics.csv",
        index=False,
    )

    pd.DataFrame(all_testing_metrics).to_csv(
        args.output_root
        / "tables"
        / "logistic_regression_testing_metrics.csv",
        index=False,
    )

    warnings_file = (
        args.output_root
        / "tables"
        / "logistic_regression_warnings.csv"
    )

    pd.DataFrame(all_warnings).to_csv(
        warnings_file,
        index=False,
    )

    decision_log = {
        "model": "Logistic Regression",
        "checkpoints": CHECKPOINTS,
        "development_semesters": [
            "2024 S1",
            "2024 S2",
            "2025 S1",
        ],
        "validation_semester": "2025 S2",
        "testing_split_label": TESTING_SPLIT_LABEL,
        "testing_split_note": (
            "The testing split holds the second half of 2025 S2. The "
            "label is a residual from an earlier design in which a "
            "2026 S1 cohort was the intended evaluation semester; data "
            "from that semester were not covered by the ethics approval "
            "under which this study was conducted."
        ),
        "training_row_counts": TRAINING_SEMESTER_COUNTS,
        "primary_tuning_metric": "average_precision",
        "secondary_tuning_metric": "roc_auc",
        "reference_threshold": REFERENCE_THRESHOLD,
        "fold_metrics_collected": [
            "roc_auc",
            "pr_auc",
            "recall",
            "precision",
            "specificity",
            "negative_predictive_value",
            "f1",
            "balanced_accuracy",
            "brier_score",
            "confusion_matrix_counts",
            "number_flagged",
        ],
        "preprocessing": [
            "numeric median imputation with missing indicators",
            "numeric standardisation",
            "categorical most-frequent imputation",
            "categorical one-hot encoding",
            "handle_unknown='ignore'",
        ],
        "random_seed": args.seed,
    }

    with open(
        args.output_root
        / "logs"
        / "logistic_regression_decision_log.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(decision_log, file, indent=2)

    print("\nLogistic Regression training completed.")
    print("Models saved in:", args.output_root / "models")
    print("Tables saved in:", args.output_root / "tables")
    print("Predictions saved in:", args.output_root / "predictions")
    print("Logs saved in:", args.output_root / "logs")


if __name__ == "__main__":
    main()
