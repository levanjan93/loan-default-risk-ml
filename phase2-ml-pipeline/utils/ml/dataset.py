"""Load and prepare modeling datasets from gold training tables."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from utils.config import GOLD_DIRS
from utils.datamart.data_processing_gold_table import _partition_path
from utils.datamart.parquet_io import list_partition_paths, read_pandas_partition


def _training_table_path() -> str:
    return os.path.join(GOLD_DIRS["g2_training_table"], "g2_training_table.parquet")


def load_training_frame(as_of: str) -> pd.DataFrame:
    """Load point-in-time training table rows with labels observable by *as_of*."""
    path = _training_table_path()
    if not os.path.isdir(path):
        return pd.DataFrame()

    df = read_pandas_partition(path)
    if df.empty:
        return df

    as_of_ts = pd.to_datetime(as_of)
    df["label_snapshot_date"] = pd.to_datetime(df["label_snapshot_date"])
    df = df[df["label_snapshot_date"] <= as_of_ts].copy()
    df = df.rename(columns={"label": "default_flag"})
    return df


def load_g2_feature_store(snapshot_date: str) -> pd.DataFrame:
    path = _partition_path(GOLD_DIRS["g2_feature_store"], "g2_feature_store", snapshot_date)
    if not os.path.isdir(path):
        return pd.DataFrame()
    return read_pandas_partition(path)


def list_g2_feature_snapshots() -> list[str]:
    paths = list_partition_paths(GOLD_DIRS["g2_feature_store"], "g2_feature_store")
    snapshots = []
    for path in paths:
        basename = os.path.basename(path)
        token = basename[len("g2_feature_store_") : -len(".parquet")]
        snapshots.append(token.replace("_", "-"))
    return sorted(snapshots)


def attach_loan_ids(features: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
    """Join customer-level g2 features to loan applications on *snapshot_date*."""
    from utils.config import RAW_FILES

    loans = pd.read_csv(RAW_FILES["loans"], parse_dates=["loan_start_date", "snapshot_date"])
    snapshot_ts = pd.to_datetime(snapshot_date)
    app_loans = loans[loans["loan_start_date"] == snapshot_ts][
        ["loan_id", "Customer_ID", "loan_start_date", "loan_amt", "tenure"]
    ].drop_duplicates(subset=["loan_id"])

    if app_loans.empty or features.empty:
        return pd.DataFrame()

    merged = app_loans.merge(features, on="Customer_ID", how="inner")
    merged["snapshot_date"] = snapshot_ts
    return merged


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        "loan_id",
        "Customer_ID",
        "loan_start_date",
        "loan_amt",
        "tenure",
        "snapshot_date",
        "default_flag",
        "label",
        "label_def",
        "label_snapshot_date",
        "feature_snapshot_date",
        "cs_snapshot_date",
        "att_snapshot_date",
        "fin_snapshot_date",
        # Legacy columns from pre-engineering gold partitions (drop after full backfill)
        "att_Age",
        "fin_Credit_History_Age",
        "fin_Payment_Behaviour_None",
    }
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    return [c for c in numeric_cols + bool_cols if c not in exclude]


def coerce_feature_matrix(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Ensure sklearn receives numeric features (handles pandas OHE bool/string mixes)."""
    matrix = df[feature_cols].copy()
    for column in feature_cols:
        if matrix[column].dtype == bool:
            matrix[column] = matrix[column].astype(int)
        elif matrix[column].dtype == object:
            matrix[column] = pd.to_numeric(matrix[column], errors="coerce")
    return matrix


def oot_holdout(
    df: pd.DataFrame,
    n_oot_cohorts: int = 3,
    date_col: str = "loan_start_date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (development, OOT) with the most recent application cohorts held out."""
    frame = df.copy()
    if date_col not in frame.columns:
        date_col = "label_snapshot_date"
    frame[date_col] = pd.to_datetime(frame[date_col])
    cohorts = sorted(frame[date_col].dropna().unique())
    if len(cohorts) <= n_oot_cohorts:
        n_oot_cohorts = min(1, len(cohorts) - 1)
    oot_cohorts = set(cohorts[len(cohorts) - n_oot_cohorts :]) if n_oot_cohorts else set()
    oot = frame[frame[date_col].isin(oot_cohorts)].copy()
    dev = frame[~frame[date_col].isin(oot_cohorts)].copy()
    return dev, oot


def random_dev_oot_split(
    df: pd.DataFrame,
    n_oot_cohorts: int = 3,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    date_col: str = "loan_start_date",
    label_col: str = "default_flag",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train / val / test via stratified random split; OOT via temporal cohort holdout.

    Only the OOT fold respects application time order (most recent cohorts sealed).
    Train, validation, and in-sample test are drawn randomly from older cohorts so
    hyper-parameter work can use i.i.d. folds while deployment generalisation is
    still judged out-of-time.
    """
    from sklearn.model_selection import train_test_split

    dev, oot = oot_holdout(df, n_oot_cohorts=n_oot_cohorts, date_col=date_col)
    if dev.empty or oot.empty:
        raise ValueError("Not enough cohorts for random dev + OOT split.")

    y = dev[label_col]
    train_val, test = train_test_split(
        dev,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    rel_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=rel_val,
        stratify=train_val[label_col],
        random_state=random_state,
    )
    return train.copy(), val.copy(), test.copy(), oot.copy()


def temporal_split(
    df: pd.DataFrame,
    date_col: str = "loan_start_date",
    n_val_cohorts: int = 2,
    n_test_cohorts: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows by **application cohort** to respect time order (no leakage).

    The model scores at the point of application, so we split on the
    *application date* (`loan_start_date`): train on older application cohorts and
    validate / test on the most recent ones, mirroring deployment (train on the
    past, score the future). A random split would leak cohort-level macro signal
    across folds and inflate metrics. (Splitting on `label_snapshot_date` gives an
    identical ordering since the label matures a fixed `mob` months after
    application, but we key on the application date to match the business framing.)
    """
    frame = df.copy()
    if date_col not in frame.columns:  # backward-compat fallback
        date_col = "label_snapshot_date"
    frame[date_col] = pd.to_datetime(frame[date_col])
    cohorts = sorted(frame[date_col].dropna().unique())
    if len(cohorts) <= (n_val_cohorts + n_test_cohorts):
        # Too few cohorts for an out-of-time split; fall back to a single holdout.
        n_test_cohorts = min(1, len(cohorts) - 1)
        n_val_cohorts = min(1, max(len(cohorts) - 1 - n_test_cohorts, 0))

    test_cohorts = set(cohorts[len(cohorts) - n_test_cohorts :]) if n_test_cohorts else set()
    remaining = cohorts[: len(cohorts) - n_test_cohorts] if n_test_cohorts else cohorts
    val_cohorts = set(remaining[len(remaining) - n_val_cohorts :]) if n_val_cohorts else set()
    train_cohorts = set(remaining[: len(remaining) - n_val_cohorts]) if n_val_cohorts else set(remaining)

    train = frame[frame[date_col].isin(train_cohorts)].copy()
    val = frame[frame[date_col].isin(val_cohorts)].copy()
    test = frame[frame[date_col].isin(test_cohorts)].copy()
    return train, val, test


def ks_statistic(y_true, y_score) -> float:
    """Kolmogorov-Smirnov separation between good/bad score distributions.

    KS is the regulator-favoured discrimination metric for credit scorecards:
    the maximum vertical gap between the cumulative score distributions of
    defaulters and non-defaulters. Reported alongside AUC/Gini.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    order = np.argsort(y_score)
    y_sorted = y_true[order]
    cum_bad = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
    cum_good = np.cumsum(1 - y_sorted) / max((1 - y_sorted).sum(), 1)
    return float(np.max(np.abs(cum_bad - cum_good)))
