"""
Handles chunked streaming ingestion of the full 6-million-row online fraud
CSV, constructs a stratified 500,000-row subset that preserves the exact
isFraud class ratio of the source population, applies feature engineering,
and returns model-ready arrays.

This module instead:
1. Streams the full CSV in chunks of 100,000 rows.
2. Accumulates per-class row indices from each chunk.
3. Draws a proportional stratified sample of 500,000 rows.
4. Applies One-Hot Encoding on the ``type`` column (replacing Label Encoding
   which imposed a spurious ordinal relationship on categorical levels).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


PIPELINE_CONFIG: dict = {
    "target_column": "isFraud",
    "drop_columns": ["nameOrig", "nameDest", "isFlaggedFraud"],
    "categorical_column": "type",
    "chunksize": 100_000,
    "target_n_rows": 500_000,
    "random_state": 42,
}


def _compute_population_ratio(csv_path: str, chunksize: int):
    """Stream the CSV once to compute the global fraud prevalence ratio.
    Args:
        csv_path: Path to the raw CSV file.
        chunksize: Number of rows per streaming chunk.
    Returns:
        Fraud prevalence: count(isFraud==1) / total_rows as a float.
    """
    total_rows: int = 0
    fraud_rows: int = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, usecols=["isFraud"]):
        total_rows += len(chunk)
        fraud_rows += int(chunk["isFraud"].sum())
    return fraud_rows / total_rows if total_rows > 0 else 0.0


def build_stratified_subset(csv_path: str):
    """Stream the full CSV and produce a stratified 500,000-row DataFrame.
    The function makes two passes over the file:
      Pass 1 — Determine the global fraud prevalence ratio.
      Pass 2 — Collect all fraud rows and a random sample of non-fraud rows
               scaled to match the population ratio at 500,000 total rows.

    Args:
        csv_path: Absolute or relative path to the onlinefraud.csv file.

    Returns:
        DataFrame of exactly min(target_n_rows, available_rows) rows where
        the isFraud class ratio mirrors the source population.
        
    Raises:
        FileNotFoundError: If csv_path is not accessible.
        ValueError: If the file contains zero rows or no fraud examples.
    """
    chunksize: int = PIPELINE_CONFIG["chunksize"]
    target_n: int = PIPELINE_CONFIG["target_n_rows"]
    rng = np.random.default_rng(PIPELINE_CONFIG["random_state"])

    print("  [pipeline] Pass 1 — computing population fraud ratio ...")
    fraud_ratio: float = _compute_population_ratio(csv_path, chunksize)
    if fraud_ratio == 0.0:
        raise ValueError("No fraud examples detected in the dataset.")
    print(f"  [pipeline] Population fraud ratio: {fraud_ratio:.6f}")

    n_fraud_target: int = round(target_n * fraud_ratio)
    n_legit_target: int = target_n - n_fraud_target

    fraud_chunks: list[pd.DataFrame] = []
    legit_chunks: list[pd.DataFrame] = []

    print("  [pipeline] Pass 2 — collecting stratified rows ...")
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        fraud_mask: pd.Series = chunk["isFraud"] == 1
        fraud_chunks.append(chunk[fraud_mask])
        legit_chunks.append(chunk[~fraud_mask])

    all_fraud: pd.DataFrame = pd.concat(fraud_chunks, ignore_index=True)
    all_legit: pd.DataFrame = pd.concat(legit_chunks, ignore_index=True)

    # Sample from each class; cap at available rows
    n_fraud_sample = min(n_fraud_target, len(all_fraud))
    n_legit_sample = min(n_legit_target, len(all_legit))

    fraud_sample = all_fraud.sample(n=n_fraud_sample, random_state=PIPELINE_CONFIG["random_state"])
    legit_sample = all_legit.sample(n=n_legit_sample, random_state=PIPELINE_CONFIG["random_state"])

    subset: pd.DataFrame = (
        pd.concat([fraud_sample, legit_sample], ignore_index=True)
        .sample(frac=1, random_state=PIPELINE_CONFIG["random_state"])
        .reset_index(drop=True)
    )
    print(
        f"  [pipeline] Subset shape: {subset.shape}  |  "
        f"Fraud rows: {fraud_sample.shape[0]}  |  "
        f"Legit rows: {legit_sample.shape[0]}"
    )
    return subset


def preprocess_features(
    df: pd.DataFrame,
):
    """Apply feature engineering to the stratified DataFrame.

    Steps:
    1. Drop metadata string columns and isFlaggedFraud (leakage risk).
    2. One-Hot Encode the ``type`` categorical column via pd.get_dummies.
    3. Separate target from features.
    4. Fill any residual NaN with column-wise mean.
    5. Apply StandardScaler.

    Args:
        df: Stratified subset DataFrame produced by build_stratified_subset.

    Returns:
        A tuple of:
            - X: 2-D float32 array of scaled features, shape (n_samples, n_features).
            - y: 1-D integer array of isFraud labels, shape (n_samples,).
            - feature_names: List of column names matching X columns.

    Raises:
        KeyError: If the target column or categorical column is missing.
    """
    target: str = PIPELINE_CONFIG["target_column"]
    drop_cols: list[str] = [
        c for c in PIPELINE_CONFIG["drop_columns"] if c in df.columns
    ]

    df = df.drop(columns=drop_cols, errors="ignore")
    df = df.dropna(subset=[target])

    cat_col: str = PIPELINE_CONFIG["categorical_column"]
    if cat_col in df.columns:
        df = pd.get_dummies(df, columns=[cat_col], drop_first=False)

    y: np.ndarray = df[target].values.astype(int)
    X_df: pd.DataFrame = df.drop(columns=[target])
    feature_names: list[str] = X_df.columns.tolist()

    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = X_df.fillna(X_df.mean())

    scaler: StandardScaler = StandardScaler()
    X: np.ndarray = scaler.fit_transform(X_df.values).astype(np.float32)

    return X, y, feature_names
