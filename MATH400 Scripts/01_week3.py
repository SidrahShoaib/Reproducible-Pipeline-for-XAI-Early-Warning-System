from pathlib import Path

import pandas as pd


# ==============================================================
# File locations
# ==============================================================

INPUT_FILE = Path(
    r"..\Data\Master data set MATH400 final.xlsx"
)

OUTPUT_DIR = Path(
    r"..\Data\week3"
)


# ==============================================================
# Excel sheet names
# ==============================================================

# There is one training sheet with this exact name.
TRAINING_SHEET = "2024, 2024 S1"

S2_SHEET = "S2 2025"


# ==============================================================
# Input columns used for Week 3
# ==============================================================

FINAL_COLUMNS = [
    "Final Grade",
    "Final Outcome",
]

ATTENDANCE_COLUMNS = [
    "D1 attendance",
    "D2 attendance",
    "D3 attendance",
    "D4 attendance",
    "D5 attendance",
    "D6 attendance",
]

MYIMATHS_COLUMNS = [
    "Squares and cubes",
    "Equivalent fractions",
    "Negative numbers 2",
    "Simplifying 1",
    "Substitution 1",
    "Adding subtracting fractions",
    "Dividing fractions",
    "Indices 1",
    "Multiplying fractions",
    "Substitution 2",
    "Adding algebraic fractions",
    "Multiplying algebraic fractions",
]

ALLOWED_INPUT_COLUMNS = (
    FINAL_COLUMNS
    + ATTENDANCE_COLUMNS
    + MYIMATHS_COLUMNS
)


# ==============================================================
# Functions
# ==============================================================

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove leading and trailing spaces from column names.
    """
    df = df.copy()
    df.columns = [
        str(column).strip()
        for column in df.columns
    ]
    return df


def validate_required_columns(
    df: pd.DataFrame,
    sheet_name: str,
) -> None:
    """
    Check that all required columns exist.
    """
    missing_columns = [
        column
        for column in ALLOWED_INPUT_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"\nSheet '{sheet_name}' is missing these required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_columns
            )
        )


def read_and_validate_sheet(
    excel_file: pd.ExcelFile,
    sheet_name: str,
) -> pd.DataFrame:
    """
    Read one Excel sheet and validate its columns.
    """
    df = pd.read_excel(
        excel_file,
        sheet_name=sheet_name,
    )

    df = clean_column_names(df)

    validate_required_columns(
        df,
        sheet_name,
    )

    return df


def create_engineered_features(
    df: pd.DataFrame,
    starting_identifier: int,
) -> pd.DataFrame:
    """
    Create the Week 3 engineered-feature dataset.

    Only the columns specified in ALLOWED_INPUT_COLUMNS
    are used in the calculations.
    """

    df = clean_column_names(df)

    # Keep only the permitted input columns.
    df = df[ALLOWED_INPUT_COLUMNS].copy()

    # ----------------------------------------------------------
    # Convert attendance columns to numeric values.
    #
    # Blank or invalid values are treated as zero.
    # ----------------------------------------------------------

    attendance = df[ATTENDANCE_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    ).fillna(0)

    # ----------------------------------------------------------
    # Convert MyiMaths task scores to numeric values.
    #
    # Blank or invalid values remain as NaN.
    # ----------------------------------------------------------

    myimaths = df[MYIMATHS_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Only scores greater than zero are included in the
    # attempted count and average-score calculation.
    included_myiMaths_scores = myimaths.where(
        myimaths > 0
    )

    # ----------------------------------------------------------
    # Attendance calculations
    # ----------------------------------------------------------

    total_sessions_attended = attendance.sum(
        axis=1
    )

    missing_sessions = (
        6 - total_sessions_attended
    )

    # ----------------------------------------------------------
    # Create output dataframe
    # ----------------------------------------------------------

    output = pd.DataFrame(index=df.index)

    # Unique identifiers begin at 1001.
    output["unique_identifier"] = range(
        starting_identifier,
        starting_identifier + len(df),
    )


    # ----------------------------------------------------------
    # At-risk feature
    #
    # At risk = 1 when Final Grade is C- or D.
    # Otherwise, at risk = 0.
    # ----------------------------------------------------------

    final_grade_cleaned = (
        df["Final Grade"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    output["at risk"] = (
        final_grade_cleaned.isin(["C-", "D"])
        .astype(int)
    )

    # ----------------------------------------------------------
    # MyiMaths features
    # ----------------------------------------------------------

    output["Myimaths_Tasks_attempted"] = (
        (myimaths > 0).sum(axis=1)
    )

    output["Myimaths_Tasks_above_90"] = (
        (myimaths > 0.9).sum(axis=1)
    )

    output["Average_MyiMaths_score"] = (
        included_myiMaths_scores.mean(axis=1)
        .fillna(0)
        * 100
    )

    # ----------------------------------------------------------
    # Attendance features
    # ----------------------------------------------------------

    output["Total_Sessions_Attended"] = (
        total_sessions_attended
    )

    output["Attendance_Rate"] = (
        total_sessions_attended / 6
    )

    output["Attendance_Week1_3"] = (
        total_sessions_attended
    )

    output["Missing_sessions"] = (
        missing_sessions
    )

    output["Missing_sessions_rate"] = (
        missing_sessions / 6
    )

    return output


# ==============================================================
# Main program
# ==============================================================

def main() -> None:

    # Check that the input workbook exists.
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Input Excel file was not found:\n"
            f"{INPUT_FILE.resolve()}"
        )

    # Create the output directory if it does not exist.
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Open the Excel workbook.
    excel_file = pd.ExcelFile(
        INPUT_FILE
    )

    available_sheets = excel_file.sheet_names

    # Check that the required sheets exist.
    required_sheets = [
        TRAINING_SHEET,
        S2_SHEET,
    ]

    missing_sheets = [
        sheet
        for sheet in required_sheets
        if sheet not in available_sheets
    ]

    if missing_sheets:
        raise ValueError(
            "The following required sheets were not found:\n"
            + "\n".join(
                f"  - {sheet}"
                for sheet in missing_sheets
            )
            + "\n\nAvailable sheets are:\n"
            + "\n".join(
                f"  - {sheet}"
                for sheet in available_sheets
            )
        )

    # ----------------------------------------------------------
    # Training dataset
    #
    # The complete "2024, 2024 S1" sheet is used for training.
    # ----------------------------------------------------------

    training_input = read_and_validate_sheet(
        excel_file,
        TRAINING_SHEET,
    )

    training_output = create_engineered_features(
        training_input,
        starting_identifier=1001,
    )

    # ----------------------------------------------------------
    # Validation and testing datasets
    #
    # The "S2 2025" sheet is split into two halves:
    #
    # - First half  = testing
    # - Second half = validation
    # ----------------------------------------------------------

    s2_input = read_and_validate_sheet(
        excel_file,
        S2_SHEET,
    )

    midpoint = len(s2_input) // 2

    testing_input = s2_input.iloc[
        :midpoint
    ].copy()

    validation_input = s2_input.iloc[
        midpoint:
    ].copy()

    # Continue unique identifiers after the training dataset.
    testing_start_identifier = (
        1001 + len(training_output)
    )

    validation_start_identifier = (
        testing_start_identifier
        + len(testing_input)
    )

    testing_output = create_engineered_features(
        testing_input,
        starting_identifier=testing_start_identifier,
    )

    validation_output = create_engineered_features(
        validation_input,
        starting_identifier=validation_start_identifier,
    )

    # ----------------------------------------------------------
    # Output file paths
    # ----------------------------------------------------------

    training_file = (
        OUTPUT_DIR
        / "week3trainingengineeredfeatures.csv"
    )

    validation_file = (
        OUTPUT_DIR
        / "week3validationengineeredfeatures.csv"
    )

    testing_file = (
        OUTPUT_DIR
        / "week3testingengineeredfeatures.csv"
    )

    # ----------------------------------------------------------
    # Save the three CSV files
    # ----------------------------------------------------------

    training_output.to_csv(
        training_file,
        index=False,
    )

    validation_output.to_csv(
        validation_file,
        index=False,
    )

    testing_output.to_csv(
        testing_file,
        index=False,
    )

    # ----------------------------------------------------------
    # Display a summary
    # ----------------------------------------------------------

    print("Week 3 datasets created successfully.\n")

    print(
        f"Training file:\n"
        f"  {training_file.resolve()}\n"
        f"Rows: {len(training_output)}\n"
    )

    print(
        f"Validation file:\n"
        f"  {validation_file.resolve()}\n"
        f"Rows: {len(validation_output)}\n"
    )

    print(
        f"Testing file:\n"
        f"  {testing_file.resolve()}\n"
        f"Rows: {len(testing_output)}\n"
    )


if __name__ == "__main__":
    main()