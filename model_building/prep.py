Predictive Maintenance - Data Preparation Pipeline

Final MLOps pipeline:
1. Load raw data from Hugging Face Dataset Hub
2. Standardize raw column names
3. Clean data
4. Perform feature engineering
5. Create stratified 80:20 train/test split
6. Save train.csv and test.csv
7. Upload train/test datasets to Hugging Face
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi


# ==========================================================
# CONFIGURATION
# ==========================================================

PROJECT_DIR = Path(
    "predictive_maintenance_project"
)

DATA_DIR = PROJECT_DIR / "data"

TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"

TARGET = "Engine_Condition"

EXPECTED_COLUMNS = [
    "Engine_RPM",
    "Lub_Oil_Pressure",
    "Fuel_Pressure",
    "Coolant_Pressure",
    "Lub_Oil_Temp",
    "Coolant_Temp",
    "Engine_Condition"
]

ENGINEERED_FEATURES = [
    "Temperature_Difference",
    "Total_Pressure",
    "Pressure_Ratio",
    "RPM_Temperature_Interaction"
]

HF_USERNAME = os.getenv(
    "HF_USERNAME",
    "hiteshsharma"
)

HF_DATASET_REPO = os.getenv(
    "HF_DATASET_REPO",
    f"{HF_USERNAME}/predictive_maintenance-dataset"
)

HF_RAW_URI = (
    f"hf://datasets/"
    f"{HF_DATASET_REPO}/engine_data.csv"
)

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ==========================================================
# CREATE DIRECTORIES
# ==========================================================

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# LOAD RAW DATA
# ==========================================================

def load_raw_data():

    print("=" * 80)
    print("LOADING RAW DATA FROM HUGGING FACE")
    print("=" * 80)

    print(
        f"Source: {HF_RAW_URI}"
    )

    df = pd.read_csv(
        HF_RAW_URI
    )

    print(
        f"Dataset shape: {df.shape}"
    )

    print(
        "\nRaw columns:"
    )

    print(
        df.columns.tolist()
    )

    return df


# ==========================================================
# CLEAN DATA + FEATURE ENGINEERING
# ==========================================================

def clean_and_engineer(df):

    print("\n" + "=" * 80)
    print("DATA CLEANING AND FEATURE ENGINEERING")
    print("=" * 80)

    df = df.copy()

    # ------------------------------------------------------
    # Remove unnecessary columns
    # ------------------------------------------------------

    unnecessary_columns = [
        col
        for col in df.columns
        if col.lower().startswith("unnamed")
    ]

    if unnecessary_columns:

        print(
            "\nRemoving unnecessary columns:"
        )

        print(
            unnecessary_columns
        )

        df = df.drop(
            columns=unnecessary_columns
        )

    # ------------------------------------------------------
    # Standardize raw column names
    # ------------------------------------------------------

    if len(df.columns) == len(EXPECTED_COLUMNS):

        if list(df.columns) != EXPECTED_COLUMNS:

            print(
                "\nStandardizing raw column names..."
            )

            df.columns = EXPECTED_COLUMNS

    else:

        raise ValueError(
            "Unexpected number of columns in raw dataset. "
            f"Expected {len(EXPECTED_COLUMNS)}, "
            f"found {len(df.columns)}."
        )

    print(
        "\nStandardized columns:"
    )

    print(
        df.columns.tolist()
    )

    # ------------------------------------------------------
    # Remove duplicates
    # ------------------------------------------------------

    duplicate_count = (
        df.duplicated().sum()
    )

    print(
        f"\nDuplicate rows found: "
        f"{duplicate_count}"
    )

    df = (
        df
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # ------------------------------------------------------
    # Convert numeric columns
    # ------------------------------------------------------

    numeric_columns = [
        "Engine_RPM",
        "Lub_Oil_Pressure",
        "Fuel_Pressure",
        "Coolant_Pressure",
        "Lub_Oil_Temp",
        "Coolant_Temp",
        "Engine_Condition"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # ------------------------------------------------------
    # Missing values
    # ------------------------------------------------------

    missing_before = (
        df[
            EXPECTED_COLUMNS
        ]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"Missing values before cleaning: "
        f"{missing_before}"
    )

    df = (
        df
        .dropna(
            subset=EXPECTED_COLUMNS
        )
        .reset_index(drop=True)
    )

    # ------------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------------

    print(
        "\nCreating engineered features..."
    )

    # 1. Temperature Difference
    df[
        "Temperature_Difference"
    ] = (
        df["Coolant_Temp"]
        - df["Lub_Oil_Temp"]
    )

    # 2. Total Pressure
    df[
        "Total_Pressure"
    ] = (
        df["Fuel_Pressure"]
        + df["Lub_Oil_Pressure"]
        + df["Coolant_Pressure"]
    )

    # 3. Pressure Ratio
    df[
        "Pressure_Ratio"
    ] = (
        df["Fuel_Pressure"]
        /
        df["Lub_Oil_Pressure"].replace(
            0,
            np.nan
        )
    )

    # 4. RPM × Temperature Interaction
    df[
        "RPM_Temperature_Interaction"
    ] = (
        df["Engine_RPM"]
        * df["Coolant_Temp"]
    )

    # ------------------------------------------------------
    # Handle invalid values
    # ------------------------------------------------------

    df = (
        df
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
        .reset_index(drop=True)
    )

    print(
        "\nEngineered features:"
    )

    for feature in ENGINEERED_FEATURES:

        print(
            f"  ✓ {feature}"
        )

    print(
        f"\nFinal cleaned dataset shape: "
        f"{df.shape}"
    )

    return df


# ==========================================================
# TRAIN / TEST SPLIT
# ==========================================================

def create_train_test(df):

    print("\n" + "=" * 80)
    print("STRATIFIED TRAIN / TEST SPLIT")
    print("=" * 80)

    X = df.drop(
        columns=[TARGET]
    )

    y = df[TARGET]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )
    )

    train_df = X_train.copy()

    train_df[TARGET] = (
        y_train.values
    )

    test_df = X_test.copy()

    test_df[TARGET] = (
        y_test.values
    )

    # ------------------------------------------------------
    # Save locally
    # ------------------------------------------------------

    train_df.to_csv(
        TRAIN_FILE,
        index=False
    )

    test_df.to_csv(
        TEST_FILE,
        index=False
    )

    print(
        f"Train dataset: "
        f"{train_df.shape}"
    )

    print(
        f"Test dataset : "
        f"{test_df.shape}"
    )

    print(
        f"\n✓ Saved: {TRAIN_FILE}"
    )

    print(
        f"✓ Saved: {TEST_FILE}"
    )

    return train_df, test_df


# ==========================================================
# UPLOAD TRAIN / TEST TO HUGGING FACE
# ==========================================================

def upload_train_test_to_huggingface():

    print("\n" + "=" * 80)
    print("UPLOADING TRAIN / TEST TO HUGGING FACE")
    print("=" * 80)

    token = os.getenv(
        "HF_TOKEN"
    )

    if not token:

        raise ValueError(
            "HF_TOKEN environment variable is not configured."
        )

    api = HfApi(
        token=token
    )

    for file_path in [
        TRAIN_FILE,
        TEST_FILE
    ]:

        if not file_path.exists():

            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        print(
            f"\nUploading {file_path.name}..."
        )

        api.upload_file(
            path_or_fileobj=str(
                file_path
            ),
            path_in_repo=file_path.name,
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            token=token
        )

        print(
            f"✓ Uploaded {file_path.name}"
        )

    print(
        "\nDataset repository:"
    )

    print(
        HF_DATASET_REPO
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("\n")
    print("=" * 80)
    print("PREDICTIVE MAINTENANCE - DATA PREPARATION")
    print("=" * 80)

    # Step 1
    raw_df = load_raw_data()

    # Step 2
    cleaned_df = clean_and_engineer(
        raw_df
    )

    # Step 3
    train_df, test_df = (
        create_train_test(
            cleaned_df
        )
    )

    # Step 4
    upload_train_test_to_huggingface()

    print("\n" + "=" * 80)
    print(
        "DATA PREPARATION COMPLETED SUCCESSFULLY"
    )
    print("=" * 80)

    print(
        f"\nTrain shape: {train_df.shape}"
    )

    print(
        f"Test shape : {test_df.shape}"
    )


if __name__ == "__main__":
    main()
'''

prep_path = Path(
    "predictive_maintenance_project/model_building/prep.py"
)

prep_path.write_text(
    prep_code,
    encoding="utf-8"
)

print("✅ prep.py completely replaced.")
print(f"File: {prep_path}")
