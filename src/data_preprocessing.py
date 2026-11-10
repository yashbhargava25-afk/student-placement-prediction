"""
data_preprocessing.py
---------------------
Author (Primary): Sachin
Responsibility  : Dataset collection, data cleaning, and preprocessing

This module handles:
  - Loading the CSV dataset
  - Displaying basic schema information
  - Missing-value and duplicate detection/handling
  - Outlier identification (IQR method)
  - Target-variable mapping (Yes/No → 1/0)
  - Categorical encoding
  - Numerical feature scaling
  - Train/test splitting with stratified sampling
  - Building a scikit-learn ColumnTransformer pipeline

IMPORTANT: All transformers are *fitted only on training data* to prevent
data leakage. The same fitted transformer is then applied to the test set.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DEFAULT_CSV = "college_student_placement.csv"   # rename your downloaded file to this
TARGET_COL  = "Placement"                        # adjust if actual column name differs
RANDOM_SEED = 42
TEST_SIZE   = 0.20   # 80/20 stratified split

# Columns to drop (identifiers only — college ID carries no predictive student signal)
ID_COLS = ["College_ID", "college_id", "CollegeID", "Student_ID", "student_id"]

# Known categorical columns (will be detected automatically if not listed)
KNOWN_CATEGORICAL = []

# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_dataset(csv_path: str | None = None) -> pd.DataFrame:
    """
    Load the dataset from *csv_path*.
    Falls back to DATA_DIR/DEFAULT_CSV when no path is given.
    Raises a clear FileNotFoundError with setup instructions.
    """
    if csv_path is None:
        csv_path = os.path.join(DATA_DIR, DEFAULT_CSV)
        if not os.path.exists(csv_path):
            alt_path = os.path.join(DATA_DIR, "college_student_placement_dataset.csv")
            if os.path.exists(alt_path):
                csv_path = alt_path

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"\n[ERROR] Dataset not found at: {csv_path}\n"
            "─────────────────────────────────────────────────\n"
            "HOW TO FIX:\n"
            "  1. Download the dataset from Kaggle:\n"
            "     https://www.kaggle.com/datasets/sahilislam007/"
            "college-student-placement-factors-dataset\n"
            "  2. Rename the downloaded CSV to:\n"
            f"     {DEFAULT_CSV}\n"
            "  3. Place it inside the  data/  directory:\n"
            f"     {DATA_DIR}\n"
            "─────────────────────────────────────────────────\n"
        )

    df = pd.read_csv(csv_path)
    print(f"[✓] Dataset loaded  →  shape: {df.shape}")
    return df


def display_schema(df: pd.DataFrame) -> None:
    """Print shape, dtypes, and a sample of rows."""
    print("\n" + "=" * 60)
    print("DATASET SCHEMA")
    print("=" * 60)
    print(f"Shape     : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nColumns   : {list(df.columns)}")
    print("\nData types:")
    print(df.dtypes.to_string())
    print("\nFirst 5 rows:")
    print(df.head().to_string())
    print("=" * 60 + "\n")


def detect_target_column(df: pd.DataFrame, candidate: str = TARGET_COL) -> str:
    """
    Detect the placement target column.
    Tries the exact name first, then a case-insensitive search.
    """
    if candidate in df.columns:
        return candidate
    lower_map = {c.lower(): c for c in df.columns}
    key = candidate.lower()
    if key in lower_map:
        return lower_map[key]
    # Broader search
    for col in df.columns:
        if "placement" in col.lower() or "placed" in col.lower():
            return col
    raise KeyError(
        f"Cannot find placement target column. "
        f"Available columns: {list(df.columns)}"
    )


def report_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame summarising missing values per column."""
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    summary = pd.DataFrame({"missing_count": missing, "missing_%": pct})
    summary = summary[summary["missing_count"] > 0].sort_values(
        "missing_%", ascending=False
    )
    print("\n── Missing Values ──────────────────────────────────────")
    if summary.empty:
        print("  No missing values found.")
    else:
        print(summary.to_string())
    print()
    return summary


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values:
      - Numerical columns → median (robust to outliers)
      - Categorical columns → mode
    """
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if df[col].dtype in [np.float64, np.int64, float, int]:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  [impute] '{col}' → filled {df[col].isnull().sum()} NaN with median={median_val:.4f}")
        else:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"  [impute] '{col}' → filled NaN with mode='{mode_val}'")
    return df


def report_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove and report duplicate rows."""
    n_dup = df.duplicated().sum()
    print(f"\n── Duplicates: {n_dup} rows")
    if n_dup > 0:
        df = df.drop_duplicates()
        print(f"  [removed] {n_dup} duplicate rows → new shape: {df.shape}")
    return df


def detect_outliers_iqr(df: pd.DataFrame, numeric_cols: list) -> dict:
    """
    Identify outliers using the IQR method (1.5 × IQR rule).
    Returns a dict {col: count_of_outliers}.
    Does NOT remove outliers — tree-based models are robust to them;
    Logistic Regression is handled by StandardScaler.
    """
    outlier_counts = {}
    print("\n── IQR Outlier Detection ────────────────────────────────")
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_counts[col] = int(n_out)
        if n_out > 0:
            print(f"  {col:30s} : {n_out} outliers  [bounds: {lower:.2f} – {upper:.2f}]")
    if not any(outlier_counts.values()):
        print("  No significant outliers detected.")
    print()
    return outlier_counts


def map_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Map the target column to binary integers: 1 = Placed, 0 = Not Placed.
    Handles Yes/No, 1/0, and similar variants robustly.
    """
    df = df.copy()
    unique_vals = df[target_col].dropna().unique()
    print(f"\n── Target column '{target_col}' unique values: {unique_vals}")

    # Already numeric binary
    if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
        df[target_col] = df[target_col].astype(int)
        print(f"  Target already binary — no remapping needed.")
        return df

    # String mapping
    positive_labels = {"yes", "placed", "1", "true", "y"}
    df[target_col] = df[target_col].apply(
        lambda v: 1 if str(v).strip().lower() in positive_labels else 0
    )
    print(f"  Mapped → 1 (Placed): {df[target_col].sum():,}  |  0 (Not Placed): {(df[target_col] == 0).sum():,}")
    return df


def drop_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier columns that carry no predictive signal."""
    cols_to_drop = [c for c in ID_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"\n── Dropped ID columns: {cols_to_drop}")
    return df


def identify_column_types(df: pd.DataFrame, target_col: str):
    """
    Automatically separate numerical and categorical feature columns.
    Returns (numeric_features, categorical_features).
    """
    feature_cols = [c for c in df.columns if c != target_col]
    numeric_features = df[feature_cols].select_dtypes(
        include=[np.number]
    ).columns.tolist()
    categorical_features = df[feature_cols].select_dtypes(
        include=["object", "category", "str", "string"]
    ).columns.tolist()
    print(f"\n── Numerical features   ({len(numeric_features)}): {numeric_features}")
    print(f"── Categorical features ({len(categorical_features)}): {categorical_features}")
    return numeric_features, categorical_features


def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """
    Build a scikit-learn ColumnTransformer:
      - Numerical → StandardScaler
      - Categorical → OneHotEncoder (handle_unknown='ignore')

    This transformer will be fitted ONLY on training data.
    """
    transformers = []

    if numeric_features:
        num_pipeline = Pipeline([("scaler", StandardScaler())])
        transformers.append(("num", num_pipeline, numeric_features))

    if categorical_features:
        cat_pipeline = Pipeline([
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_pipeline, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


def split_data(df: pd.DataFrame, target_col: str):
    """
    Stratified 80/20 train-test split.
    Returns X_train, X_test, y_train, y_test.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"\n── Train/Test split (stratified, test={TEST_SIZE*100:.0f}%)")
    print(f"   Training set : {X_train.shape[0]:,} rows")
    print(f"   Test set     : {X_test.shape[0]:,} rows")
    print(f"   Class balance (train) → {y_train.value_counts().to_dict()}")
    return X_train, X_test, y_train, y_test


def run_full_preprocessing(csv_path: str | None = None):
    """
    End-to-end preprocessing pipeline.
    Returns:
        X_train, X_test, y_train, y_test,
        preprocessor (fitted),
        df_clean (cleaned DataFrame before split),
        numeric_features, categorical_features, target_col
    """
    print("\n" + "=" * 60)
    print("  SACHIN — DATA COLLECTION, CLEANING & PREPROCESSING")
    print("=" * 60)

    # 1. Load
    df = load_dataset(csv_path)

    # 2. Schema
    display_schema(df)

    # 3. Detect target
    target_col = detect_target_column(df)

    # 4. Missing values
    report_missing_values(df)
    df = handle_missing_values(df)

    # 5. Duplicates
    df = report_duplicates(df)

    # 6. Drop identifier columns
    df = drop_id_columns(df)

    # 7. Map target
    df = map_target(df, target_col)

    # 8. Identify column types
    numeric_features, categorical_features = identify_column_types(df, target_col)

    # 9. Outlier report (informational only)
    detect_outliers_iqr(df, numeric_features)

    # 10. Split BEFORE fitting any transformer (prevents leakage)
    X_train, X_test, y_train, y_test = split_data(df, target_col)

    # 11. Build and fit preprocessor on training data ONLY
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    preprocessor.fit(X_train)
    print("\n[✓] Preprocessor fitted on training data only.")

    print("\n[✓] Preprocessing complete.\n")
    return (
        X_train, X_test, y_train, y_test,
        preprocessor,
        df,
        numeric_features, categorical_features, target_col
    )


if __name__ == "__main__":
    run_full_preprocessing()
