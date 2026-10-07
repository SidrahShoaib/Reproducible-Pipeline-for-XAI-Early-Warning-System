from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# Project configuration
# ============================================================

CHECKPOINTS = [
    "week3",
    "week6",
    "week8",
    "week12",
]

TARGET = "at_risk"
IDENTIFIER_COLUMN = "unique_identifier"

TRAINING_SEMESTER_COUNTS = {
    "2024 S1": 104,
    "2024 S2": 59,
    "2025 S1": 111,
}


# ============================================================
# Output and logging utilities
# ============================================================

def ensure_dirs(output_root: Path) -> None:
    folders = [
        "models",
        "tables",
        "predictions",
        "logs",
        "preprocessing",
        "temporal_splits",
    ]

    for folder in folders:
        (Path(output_root) / folder).mkdir(
            parents=True,
            exist_ok=True,
        )


def create_output_directories(output_root: Path) -> None:
    """
    Alias retained for compatibility with earlier scripts.
    """
    ensure_dirs(output_root)


def configure_logging(output_root: Path) -> None:
    log_file = Path(output_root) / "logs" / "run.log"

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def save_json(data: dict, file_path: Path) -> None:
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=str)


def save_model(model, file_path: Path) -> None:
    joblib.dump(model, file_path)


def load_model(file_path: Path):
    return joblib.load(file_path)


# ============================================================
# File discovery
# ============================================================

def locate_file(
    data_root: Path,
    checkpoint: str,
    split: str,
) -> Path:
    """
    Locates files in folders such as:

        Data/week3
        Data/week6
        Data/week8
        Data/week12

    Expected filenames include:

        week_3_training_engineered_features.csv
        week_3_validation_engineered_features.csv
        week_3_testing_engineered_features.csv
    """

    data_root = Path(data_root)

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

    searched_folders = "\n".join(
        f"    {folder}" for folder in folder_candidates
    )

    searched_files = "\n".join(
        f"    {filename}"
        for filename in filename_candidates[split]
    )

    raise FileNotFoundError(
        f"\nNo {split} file found for {checkpoint}.\n\n"
        f"Searched folders:\n{searched_folders}\n\n"
        f"Tried filenames:\n{searched_files}\n"
    )


# ============================================================
# Column and target preparation
# ============================================================

def normalise_column_name(column_name: str) -> str:
    """
    Converts names such as:

        at risk
        Attendance Rate
        Missing-sessions rate

    to:

        at_risk
        attendance_rate
        missing_sessions_rate
    """

    value = str(column_name).strip().lower()

    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    value = value.strip("_")

    return value


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
            rename_dictionary[column] = TARGET

        elif normalised in {
            "unique_identifier",
            "student_id",
            "student_enrolment_id",
            "enrolment_id",
            "id",
        }:
            rename_dictionary[column] = IDENTIFIER_COLUMN

        else:
            rename_dictionary[column] = normalised

    result = df.rename(
        columns=rename_dictionary
    ).copy()

    if TARGET not in result.columns:
        raise ValueError(
            "Target column not found. Expected a column named "
            "'at risk' or 'at_risk'."
        )

    return result


def encode_target(series: pd.Series) -> pd.Series:
    """
    Converts common binary target formats into numeric 0 and 1.
    """

    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(
            series,
            errors="coerce",
        )

        unique_values = set(
            numeric.dropna().unique()
        )

        if unique_values.issubset({0, 1}):
            return numeric

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
    """
    Loads and prepares one CSV file.
    """

    file_path = locate_file(
        data_root=data_root,
        checkpoint=checkpoint,
        split=split,
    )

    dataframe = pd.read_csv(file_path)

    dataframe = normalise_columns(dataframe)

    dataframe[TARGET] = encode_target(
        dataframe[TARGET]
    )

    if dataframe[TARGET].isna().any():
        bad_values = dataframe.loc[
            dataframe[TARGET].isna(),
            TARGET,
        ].unique()

        raise ValueError(
            f"Unrecognised or missing target values in "
            f"{file_path}: {bad_values}"
        )

    dataframe[TARGET] = (
        dataframe[TARGET].astype(int)
    )

    logging.info(
        "Loaded %s: %d rows, %d columns",
        file_path,
        len(dataframe),
        len(dataframe.columns),
    )

    return dataframe


# ============================================================
# Semester reconstruction
# ============================================================

def add_training_semesters(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adds semester labels based on confirmed row order:

        Data rows 2–105   = 2024 S1, 104 rows
        Data rows 106–164 = 2024 S2, 59 rows
        Data rows 165–275 = 2025 S1, 111 rows

    Pandas data-row positions:

        0–103   = 2024 S1
        104–162 = 2024 S2
        163–273 = 2025 S1
    """

    expected_rows = sum(
        TRAINING_SEMESTER_COUNTS.values()
    )

    actual_rows = len(dataframe)

    if actual_rows != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} training rows, "
            f"but found {actual_rows} rows."
        )

    semester_labels = []

    for semester, count in (
        TRAINING_SEMESTER_COUNTS.items()
    ):
        semester_labels.extend(
            [semester] * count
        )

    result = dataframe.copy()
    result["semester"] = semester_labels

    return result


# ============================================================
# Load all checkpoint data
# ============================================================

def load_checkpoint_data(
    data_root: Path,
    checkpoint: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:

        development:
            2024 S1, 2024 S2 and 2025 S1

        validation:
            2025 S2

        testing:
            2026 S1
    """

    training = read_data_file(
        data_root,
        checkpoint,
        "training",
    )

    validation = read_data_file(
        data_root,
        checkpoint,
        "validation",
    )

    testing = read_data_file(
        data_root,
        checkpoint,
        "testing",
    )

    training = add_training_semesters(
        training
    )

    validation = validation.copy()
    validation["semester"] = "2025 S2"

    testing = testing.copy()
    testing["semester"] = "2026 S1"

    return training, validation, testing


# Compatibility alias used by earlier scripts.
def load_all(
    data_root: Path,
    checkpoint: str,
):
    return load_checkpoint_data(
        data_root,
        checkpoint,
    )


# ============================================================
# Feature selection
# ============================================================

def get_feature_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Excludes the target, identifier and reconstructed semester.
    """

    excluded_columns = {
        TARGET,
        IDENTIFIER_COLUMN,
        "semester",
        "source_file",
    }

    feature_columns = [
        column
        for column in dataframe.columns
        if column not in excluded_columns
    ]

    if not feature_columns:
        raise ValueError(
            "No predictor columns remain after exclusions."
        )

    return feature_columns


# Compatibility alias used by earlier scripts.
def make_feature_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    return get_feature_columns(dataframe)


# ============================================================
# Preprocessing pipelines
# ============================================================

def build_preprocessor(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """
    Builds leakage-safe preprocessing.

    Numeric:
        median imputation
        optional standardisation

    Categorical:
        most-frequent imputation
        one-hot encoding

    The preprocessor is fitted only within each model pipeline.
    """

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


# Compatibility alias used by earlier scripts.
def make_preprocessor(
    dataframe: pd.DataFrame,
    feature_columns: list[str] | None = None,
    scale_numeric: bool = True,
):
    if feature_columns is None:
        feature_columns = get_feature_columns(
            dataframe
        )

    return build_preprocessor(
        dataframe,
        feature_columns,
        scale_numeric=scale_numeric,
    )


# ============================================================
# Prediction helper
# ============================================================

def prediction_frame(
    dataframe: pd.DataFrame,
    probabilities: np.ndarray,
    model_name: str,
    threshold: float = 0.40,
) -> pd.DataFrame:
    if IDENTIFIER_COLUMN in dataframe.columns:
        identifiers = dataframe[
            IDENTIFIER_COLUMN
        ].values
    else:
        identifiers = np.arange(
            len(dataframe)
        )

    return pd.DataFrame(
        {
            "unique_identifier": identifiers,
            "semester": dataframe["semester"].values,
            "true_label": dataframe[
                TARGET
            ].astype(int).values,
            "predicted_probability": probabilities,
            "predicted_class_0_40": (
                probabilities >= 0.40
            ).astype(int),
            "predicted_class_selected": (
                probabilities >= threshold
            ).astype(int),
            "model_name": model_name,
        }
    )