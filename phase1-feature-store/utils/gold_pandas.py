"""Pandas implementation of gold layer (Windows + Python 3.13 PySpark workaround)."""

import os
from datetime import datetime, timedelta

import pandas as pd

from utils.data_processing_gold_table import (
    KEY_COLUMNS,
    MAX_OHE_CARDINALITY,
    PII_COLUMNS,
    _parse_partition_date,
    _partition_path,
)
from utils.parquet_io import list_partition_paths, read_pandas_partition, save_pandas_partition


def _read_silver(silver_directory, silver_prefix, snapshot_date_str):
    path = _partition_path(silver_directory, silver_prefix, snapshot_date_str)
    df = read_pandas_partition(path)
    print("loaded silver from:", path, "row count:", len(df))
    return df


def _read_all_partitions(base_directory, prefix):
    paths = list_partition_paths(base_directory, prefix)
    if not paths:
        raise FileNotFoundError(
            f"No non-empty partitions found under {base_directory} for {prefix}. "
            "Re-run gold for that table."
        )
    return pd.concat([read_pandas_partition(path) for path in paths], ignore_index=True)


def _median_impute(df, column_name):
    if column_name in df.columns:
        df[column_name] = df[column_name].fillna(df[column_name].median())
    return df


def _impute_numeric_medians(df, exclude_columns):
    for column_name in df.columns:
        if column_name in exclude_columns:
            continue
        if pd.api.types.is_numeric_dtype(df[column_name]):
            df[column_name] = df[column_name].fillna(df[column_name].median())
    return df


def _detect_categorical_columns(df, exclude_columns):
    categoricals = []
    for column_name in df.columns:
        if column_name in exclude_columns:
            continue
        if df[column_name].dtype == object or str(df[column_name].dtype) == "string":
            distinct_count = df[column_name].nunique(dropna=True)
            if 0 < distinct_count <= MAX_OHE_CARDINALITY:
                categoricals.append(column_name)
    return categoricals


def _one_hot_encode(df, categorical_columns):
    if not categorical_columns:
        return df
    for column_name in categorical_columns:
        df[column_name] = df[column_name].astype(str)
    return pd.get_dummies(
        df, columns=categorical_columns, prefix=categorical_columns, drop_first=True
    )


def _aggregate_clickstream(df, as_of_date_str):
    fe_cols = [column_name for column_name in df.columns if column_name.startswith("fe_")]
    grouped = df.groupby("Customer_ID", as_index=False)
    out = grouped.agg(
        cs_row_count=("snapshot_date", "count"),
        cs_distinct_snapshot_count=("snapshot_date", "nunique"),
        cs_source_max_snapshot_date=("snapshot_date", "max"),
        **{f"{fe_col}_avg": (fe_col, "mean") for fe_col in fe_cols},
        **{f"{fe_col}_sum": (fe_col, "sum") for fe_col in fe_cols},
    )
    out["snapshot_date"] = as_of_date_str
    return out


def _latest_pit_features(source_df, customer_col, snapshot_col, as_of_date, strict_before, snapshot_alias):
    source_df = source_df.copy()
    source_df[snapshot_col] = pd.to_datetime(source_df[snapshot_col])
    as_of_ts = pd.Timestamp(as_of_date)
    if strict_before:
        filtered = source_df[source_df[snapshot_col] < as_of_ts]
    else:
        filtered = source_df[source_df[snapshot_col] <= as_of_ts]
    idx = filtered.groupby(customer_col)[snapshot_col].idxmax()
    result = filtered.loc[idx].copy()
    if snapshot_alias:
        result = result.rename(columns={snapshot_col: snapshot_alias})
    return result


def _prefix_columns(df, prefix, skip_columns):
    rename_map = {
        column_name: f"{prefix}{column_name}"
        for column_name in df.columns
        if column_name not in skip_columns
    }
    return df.rename(columns=rename_map)


def _pit_join_labels_to_g1(labels_df, g1_df, snapshot_alias):
    labels = labels_df.rename(columns={"snapshot_date": "label_snapshot_date"})
    g1 = g1_df.rename(columns={"snapshot_date": "g1_snapshot_date"})
    labels["label_snapshot_date"] = pd.to_datetime(labels["label_snapshot_date"])
    g1["g1_snapshot_date"] = pd.to_datetime(g1["g1_snapshot_date"])

    merged = labels.merge(g1, on="Customer_ID", how="left")
    merged = merged[merged["g1_snapshot_date"] < merged["label_snapshot_date"]]
    merged = merged.sort_values("g1_snapshot_date", ascending=False)
    merged = merged.groupby("loan_id", as_index=False).first()
    return merged.rename(columns={"g1_snapshot_date": snapshot_alias})


def process_labels_gold_table(
    snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd=30, mob=6
):
    _ = spark
    df = _read_silver(silver_loan_daily_directory, "silver_loan_daily", snapshot_date_str)
    df = df[df["mob"] == mob].copy()
    df["label"] = (df["dpd"] >= dpd).astype(int)
    df["label_def"] = f"{dpd}dpd_{mob}mob"
    out = df[["loan_id", "Customer_ID", "label", "label_def", "snapshot_date"]]
    path = _partition_path(gold_label_store_directory, "gold_label_store", snapshot_date_str)
    save_pandas_partition(out, path)
    print("saved to:", path)


def process_g1_attributes_table(
    snapshot_date_str, silver_attributes_directory, gold_g1_attributes_directory, spark
):
    _ = spark
    df = _read_silver(silver_attributes_directory, "silver_attributes", snapshot_date_str)
    df = df.drop(columns=[column for column in PII_COLUMNS if column in df.columns], errors="ignore")
    df = _median_impute(df, "Age")
    df = _one_hot_encode(df, _detect_categorical_columns(df, set(KEY_COLUMNS)))
    path = _partition_path(gold_g1_attributes_directory, "g1_attributes", snapshot_date_str)
    save_pandas_partition(df, path)
    print("saved to:", path)


def process_g1_financials_table(
    snapshot_date_str, silver_financials_directory, gold_g1_financials_directory, spark
):
    _ = spark
    df = _read_silver(silver_financials_directory, "silver_financials", snapshot_date_str)
    if "Type_of_Loan" in df.columns:
        df = df.drop(columns=["Type_of_Loan"])
    df = _impute_numeric_medians(df, set(KEY_COLUMNS))
    df = _one_hot_encode(df, _detect_categorical_columns(df, set(KEY_COLUMNS)))
    path = _partition_path(gold_g1_financials_directory, "g1_financials", snapshot_date_str)
    save_pandas_partition(df, path)
    print("saved to:", path)


def process_g1_clickstream_table(
    snapshot_date_str,
    silver_clickstream_directory,
    gold_g1_clickstream_directory,
    spark,
    clickstream_window_days=30,
):
    _ = spark
    as_of_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d").date()
    window_start = as_of_date - timedelta(days=clickstream_window_days)
    prefix = "silver_clickstream"
    paths = list_partition_paths(silver_clickstream_directory, prefix)
    selected = [
        path
        for path in paths
        if window_start < _parse_partition_date(path, prefix) <= as_of_date
    ]
    if not selected:
        raise FileNotFoundError(
            f"No silver clickstream partitions in ({window_start}, {as_of_date}]"
        )
    df = pd.concat([read_pandas_partition(path) for path in selected], ignore_index=True)
    print(
        f"silver clickstream window for {snapshot_date_str}: "
        f"{len(selected)} partition(s), row count: {len(df)}"
    )
    df = _impute_numeric_medians(df, set(KEY_COLUMNS))
    df = _aggregate_clickstream(df, snapshot_date_str)
    path = _partition_path(gold_g1_clickstream_directory, "g1_clickstream", snapshot_date_str)
    save_pandas_partition(df, path)
    print("saved to:", path)


def process_g2_feature_store(
    snapshot_date_str,
    clickstream_window_days,
    gold_g1_clickstream_directory,
    gold_g1_attributes_directory,
    gold_g1_financials_directory,
    gold_g2_feature_store_directory,
    spark,
):
    _ = clickstream_window_days, spark
    as_of_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d").date()
    cs_features = read_pandas_partition(
        _partition_path(gold_g1_clickstream_directory, "g1_clickstream", snapshot_date_str)
    )
    cs_features["cs_as_of_snapshot_date"] = cs_features["snapshot_date"]

    attributes_all = _read_all_partitions(gold_g1_attributes_directory, "g1_attributes")
    financials_all = _read_all_partitions(gold_g1_financials_directory, "g1_financials")

    attr_features = _latest_pit_features(
        attributes_all, "Customer_ID", "snapshot_date", as_of_date, False, "att_snapshot_date"
    )
    fin_features = _latest_pit_features(
        financials_all, "Customer_ID", "snapshot_date", as_of_date, False, "fin_snapshot_date"
    )
    attr_features = _prefix_columns(attr_features, "att_", {"Customer_ID", "att_snapshot_date"})
    fin_features = _prefix_columns(fin_features, "fin_", {"Customer_ID", "fin_snapshot_date"})

    features = cs_features.merge(attr_features, on="Customer_ID", how="left").merge(
        fin_features, on="Customer_ID", how="left"
    )
    features["snapshot_date"] = snapshot_date_str
    for date_col in ["cs_as_of_snapshot_date", "att_snapshot_date", "fin_snapshot_date"]:
        features[date_col] = pd.to_datetime(features[date_col], errors="coerce")
    features["feature_snapshot_date"] = features[
        ["cs_as_of_snapshot_date", "att_snapshot_date", "fin_snapshot_date"]
    ].max(axis=1)
    features = features.drop(columns=["cs_as_of_snapshot_date"])

    path = _partition_path(gold_g2_feature_store_directory, "g2_feature_store", snapshot_date_str)
    save_pandas_partition(features, path)
    print("saved to:", path)


def process_g2_training_table(
    clickstream_window_days,
    gold_g1_clickstream_directory,
    gold_g1_attributes_directory,
    gold_g1_financials_directory,
    gold_label_store_directory,
    gold_g2_training_table_directory,
    spark,
):
    _ = clickstream_window_days, spark
    labels = _read_all_partitions(gold_label_store_directory, "gold_label_store")
    clickstream_all = _read_all_partitions(gold_g1_clickstream_directory, "g1_clickstream")
    attributes_all = _read_all_partitions(gold_g1_attributes_directory, "g1_attributes")
    financials_all = _read_all_partitions(gold_g1_financials_directory, "g1_financials")

    cs_features = _pit_join_labels_to_g1(labels, clickstream_all, "cs_snapshot_date")
    attr_features = _prefix_columns(
        _pit_join_labels_to_g1(labels, attributes_all, "att_snapshot_date"),
        "att_",
        {"loan_id", "Customer_ID", "label", "label_def", "label_snapshot_date", "att_snapshot_date"},
    )
    fin_features = _prefix_columns(
        _pit_join_labels_to_g1(labels, financials_all, "fin_snapshot_date"),
        "fin_",
        {"loan_id", "Customer_ID", "label", "label_def", "label_snapshot_date", "fin_snapshot_date"},
    )

    training = cs_features.merge(
        attr_features.drop(columns=["Customer_ID", "label", "label_def", "label_snapshot_date"]),
        on="loan_id",
        how="left",
    ).merge(
        fin_features.drop(columns=["Customer_ID", "label", "label_def", "label_snapshot_date"]),
        on="loan_id",
        how="left",
    )

    for date_col in ["cs_snapshot_date", "att_snapshot_date", "fin_snapshot_date"]:
        training[date_col] = pd.to_datetime(training[date_col])
    training["feature_snapshot_date"] = training[
        ["cs_snapshot_date", "att_snapshot_date", "fin_snapshot_date"]
    ].max(axis=1)
    training = training[training["feature_snapshot_date"] < training["label_snapshot_date"]]
    training = training.sort_values("feature_snapshot_date", ascending=False)
    training = training.groupby(["loan_id", "label_snapshot_date"], as_index=False).first()

    path = os.path.join(gold_g2_training_table_directory, "g2_training_table.parquet")
    save_pandas_partition(training, path)
    print("saved to:", path)
