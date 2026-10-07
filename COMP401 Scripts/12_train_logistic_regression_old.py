#Train_LR

from __future__ import annotations

import argparse
import ast
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


# ============================================================
# Configuration
# ============================================================

CHECKPOINTS = ["week3", "week6", "week8", "week12"]

TARGET_COLUMN = "at_risk"
IDENTIFIER_COLUMN = "unique_identifier"

TRAINING_SEMESTER_COUNTS = {
    "2024 S1": 104,
    "2024 S2": 59,
    "2025 S1": 111,
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
    {
        "C": 0.01,
        "class_weight": None,
    },
    {
        "C": 0.01,
        "class_weight": "balanced",
    },
    {
        "C": 0.1,
        "class_weight": None,
    },
    {
        "C": 0.1,
        "class_weight": "balanced",
    },
    {
        "C": 1.0,
        "class_weight": None,
    },
    {
        "C": 1.0,
        "class_weight": "balanced",
    },
    {
        "C": 10.0,
        "class_weight": None,
    },
    {
        "C": 10.0,
        "class_weight": "balanced",
    },
    {
        "C": 100.0,
        "class_weight": None,
    },
    {
        "C": 100.0,
        "class_weight": "balanced",
    },
]


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
    log_file = output_root / "logs" / "logistic_regression_execution.log"

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
    Finds files such as:

    Data/week3/week_3_training_engineered_features.csv
    Data/week6/week_6_validation_engineered_features.csv
    """

    week_number = checkpoint.replace("week", "")

    folder_candidates = [
        data_root / f"week{week_number}",
        data_root / f"week_{week_number}",
        data_root / checkpoint,
    ]

    filename_candidates = {
        "training": [
            f"week_{week_number}_training_engineered_features.csv",
            f"week{week_number}_training_engineered_features.csv",
            f"{checkpoint}_training_engineered_features.csv",
            "training.csv",
        ],
        "validation": [
            f"week_{week_number}_validation_engineered_features.csv",
            f"week{week_number}_validation_engineered_features.csv",
            f"{checkpoint}_validation_engineered_features.csv",
            "validation.csv",
        ],
        "testing": [
            f"week_{week_number}_testing_engineered_features.csv",
            f"week{week_number}_testing_engineered_features.csv",
            f"{checkpoint}_testing_engineered_features.csv",
            f"week_{week_number}_test_engineered_features.csv",
            "testing.csv",
            "test.csv",
        ],
    }

    for folder in folder_candidates:
        for filename in filename_candidates[split]:
            candidate = folder / filename

            if candidate.exists():
                return candidate

    attempted_folders = "\n".join(
        f"  {folder}" for folder in folder_candidates
    )

    raise FileNotFoundError(
        f"\nCould not find the {split} file for {checkpoint}.\n"
        f"Searched folders:\n{attempted_folders}\n"
        f"Tried filenames:\n{filename_candidates[split]}"
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


def load_checkpoint_data(
    data_root: Path,
    checkpoint: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    training = read_data_file(data_root, checkpoint, "training")
    validation = read_data_file(data_root, checkpoint, "validation")
    testing = read_data_file(data_root, checkpoint, "testing")

    training = add_training_semesters(training)

    validation = validation.copy()
    validation["semester"] = "2025 S2"

    testing = testing.copy()
    testing["semester"] = "2026 S1"

    return training, validation, testing


# ============================================================
# Preprocessing and model creation
# ============================================================

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded_columns = {
        TARGET_COLUMN,
        IDENTIFIER_COLUMN,
        "semester",
    }

    return [
        column
        for column in df.columns
        if column not in excluded_columns
    ]


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
    }


# ============================================================
# Main training procedure
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to the Data folder.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Path for models, tables, predictions and logs.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    create_output_directories(args.output_root)
    configure_logging(args.output_root)

    logging.info("Starting Logistic Regression training")
    logging.info("Data root: %s", args.data_root)
    logging.info("Output root: %s", args.output_root)

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
        preprocessor_template = build_preprocessor(
            training,
            feature_columns,
        )

        logging.info(
            "%s: development rows=%d, validation rows=%d, testing rows=%d",
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
            validation_semester = fold_definition["validation_semester"]

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
                    "%s fold %d skipped because training data contains "
                    "only one target class",
                    checkpoint,
                    fold_number,
                )
                continue

            if y_fold_validation.nunique() < 2:
                logging.warning(
                    "%s fold %d validation data contains only one class",
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

                warning_count_before = len(all_warnings)

                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.simplefilter("always")

                    try:
                        pipeline.fit(X_train, y_train)
                        probabilities = pipeline.predict_proba(
                            X_fold_validation
                        )[:, 1]

                        fold_pr_auc = average_precision_score(
                            y_fold_validation,
                            probabilities,
                        )

                        fold_roc_auc = (
                            roc_auc_score(
                                y_fold_validation,
                                probabilities,
                            )
                            if y_fold_validation.nunique() == 2
                            else np.nan
                        )

                        fit_status = "completed"
                        error_message = ""

                    except Exception as error:
                        fold_pr_auc = np.nan
                        fold_roc_auc = np.nan
                        fit_status = "failed"
                        error_message = repr(error)

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
                            "%s fold %d parameter %d convergence warning: %s",
                            checkpoint,
                            fold_number,
                            parameter_number,
                            caught_warning.message,
                        )

                result = {
                    "checkpoint": checkpoint,
                    "fold": fold_number,
                    "training_semesters": "|".join(train_semesters),
                    "validation_semester": validation_semester,
                    "parameter_number": parameter_number,
                    "C": parameters["C"],
                    "class_weight": str(parameters["class_weight"]),
                    "pr_auc": fold_pr_auc,
                    "roc_auc": fold_roc_auc,
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
                f"No successful Logistic Regression folds for {checkpoint}."
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
                number_of_folds=("fold", "nunique"),
            )
            .reset_index()
        )

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
                **best_parameters,
                "mean_forward_pr_auc": best_row["mean_pr_auc"],
                "std_forward_pr_auc": best_row["std_pr_auc"],
                "mean_forward_roc_auc": best_row["mean_roc_auc"],
                "std_forward_roc_auc": best_row["std_roc_auc"],
                "number_of_folds": best_row["number_of_folds"],
            }
        )

        # ----------------------------------------------------
        # Refit final development model on all 2024 S1–2025 S1
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

        fitted_preprocessor = final_pipeline.named_steps["preprocess"]
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
                    validation_probabilities >= 0.40
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
                "semester": "2026 S1",
                "true_label": testing[TARGET_COLUMN],
                "predicted_probability": testing_probabilities,
                "predicted_class_0_40": (
                    testing_probabilities >= 0.40
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
            threshold=0.40,
        )

        validation_metrics.update(
            {
                "checkpoint": checkpoint,
                "model": "logistic_regression",
                "split": "2025 S2 validation",
            }
        )

        testing_metrics = calculate_binary_metrics(
            testing[TARGET_COLUMN],
            testing_probabilities,
            threshold=0.40,
        )

        testing_metrics.update(
            {
                "checkpoint": checkpoint,
                "model": "logistic_regression",
                "split": "2026 S1 testing",
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
            f"{best_row['mean_pr_auc']:.4f}"
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
        "testing_semester": "2026 S1",
        "training_row_counts": TRAINING_SEMESTER_COUNTS,
        "primary_tuning_metric": "average_precision",
        "reference_threshold": 0.40,
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