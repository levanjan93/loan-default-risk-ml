import glob
import os
import shutil
import sys
from datetime import datetime, timedelta

import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col
from pyspark.sql.types import DateType, IntegerType, StringType
from pyspark.sql.window import Window

from utils.parquet_io import read_partition_pandas, save_partition_pandas

# ---------------------------------------------------------------------------
# I/O helpers (Windows: pyarrow; Linux/Docker: Spark native)
# ---------------------------------------------------------------------------

KEY_COLUMNS = ["Customer_ID", "snapshot_date"]
PII_COLUMNS = ["SSN", "Name"]
MAX_OHE_CARDINALITY = 50


def _partition_path(base_directory, prefix, snapshot_date_str, ext="parquet"):
    name = prefix + "_" + snapshot_date_str.replace("-", "_") + "." + ext
    return os.path.join(base_directory, name)


def _parse_partition_date(path, prefix):
    """Parse YYYY-MM-DD from '{prefix}_YYYY_MM_DD.parquet'."""
    basename = os.path.basename(path)
    date_token = basename[len(prefix) + 1 : -len(".parquet")]
    return datetime.strptime(date_token, "%Y_%m_%d").date()


def _cast_snapshot_date(df):
    if "snapshot_date" not in df.columns:
        return df
    dtype = dict(df.dtypes).get("snapshot_date", "")
    if dtype == "date":
        return df
    if dtype in ("int", "integer", "bigint", "smallint"):
        return df.withColumn(
            "snapshot_date",
            F.date_add(F.lit("1970-01-01"), col("snapshot_date").cast(IntegerType())),
        )
    if dtype == "string":
        return df.withColumn("snapshot_date", F.to_date(col("snapshot_date")))
    if "timestamp" in dtype:
        return df.withColumn("snapshot_date", F.to_date(col("snapshot_date")))
    return df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))


def _standardize_columns(df):
    if "Customer_ID" in df.columns:
        df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    if "loan_id" in df.columns:
        df = df.withColumn("loan_id", col("loan_id").cast(StringType()))
    if "label_def" in df.columns:
        df = df.withColumn("label_def", col("label_def").cast(StringType()))
    return _cast_snapshot_date(df)


def _read_parquet(spark, path):
    if sys.platform == "win32":
        return _standardize_columns(read_partition_pandas(spark, path))
    return _standardize_columns(spark.read.parquet(path))


def _ml_vectors_to_arrays(df):
    for field in df.schema.fields:
        if field.dataType.simpleString() == "vector":
            df = df.withColumn(field.name, vector_to_array(col(field.name)))
    return df


def _flatten_array_columns(df):
    """Array columns can crash PySpark Python workers on Windows."""
    for field in df.schema.fields:
        dtype = field.dataType.simpleString()
        if dtype == "vector":
            df = df.withColumn(field.name, vector_to_array(col(field.name)))
            dtype = "array<double>"
        if dtype.startswith("array"):
            df = df.withColumn(field.name, F.concat_ws(",", col(field.name)))
    return df


def _save_parquet(df, path):
    df = _flatten_array_columns(_ml_vectors_to_arrays(_standardize_columns(df)))
    if sys.platform == "win32":
        save_partition_pandas(df, path)
    else:
        df.write.mode("overwrite").parquet(path)
    print("saved to:", path)


def _read_silver(spark, silver_directory, silver_prefix, snapshot_date_str):
    path = _partition_path(silver_directory, silver_prefix, snapshot_date_str)
    df = _read_parquet(spark, path)
    row_count = _partition_row_count(path) if sys.platform == "win32" else df.count()
    print("loaded silver from:", path, "row count:", row_count)
    return df


def _partition_row_count(path):
    import pyarrow.parquet as pq

    files = glob.glob(os.path.join(path, "*.parquet"))
    return sum(pq.ParquetFile(file_path).metadata.num_rows for file_path in files)


def _list_partition_paths(base_directory, prefix):
    pattern = os.path.join(base_directory, f"{prefix}_*.parquet")
    paths = sorted(glob.glob(pattern))
    valid_paths = []
    for path in paths:
        if glob.glob(os.path.join(path, "*.parquet")) and _partition_row_count(path) > 0:
            valid_paths.append(path)
    return valid_paths


def _read_all_partitions(spark, base_directory, prefix):
    paths = _list_partition_paths(base_directory, prefix)
    if not paths:
        raise FileNotFoundError(
            f"No non-empty partitions found under {base_directory} for {prefix}. "
            "Re-run gold for that table."
        )
    df = _read_parquet(spark, paths[0])
    for path in paths[1:]:
        df = df.unionByName(_read_parquet(spark, path), allowMissingColumns=True)
    return df


def _read_silver_clickstream_window(spark, silver_clickstream_directory, as_of_date_str, window_days):
    """Load silver clickstream partitions falling in (as_of - window_days, as_of]."""
    as_of_date = datetime.strptime(as_of_date_str, "%Y-%m-%d").date()
    window_start = as_of_date - timedelta(days=window_days)
    prefix = "silver_clickstream"

    paths = _list_partition_paths(silver_clickstream_directory, prefix)
    selected = [
        path
        for path in paths
        if window_start < _parse_partition_date(path, prefix) <= as_of_date
    ]
    if not selected:
        raise FileNotFoundError(
            f"No silver clickstream partitions in ({window_start}, {as_of_date}]"
        )

    df = _read_parquet(spark, selected[0])
    for path in selected[1:]:
        df = df.unionByName(_read_parquet(spark, path), allowMissingColumns=True)
    print(
        f"silver clickstream window for {as_of_date_str}: "
        f"{len(selected)} partition(s), row count: {df.count()}"
    )
    return df


def _median_impute_column(df, column_name):
    if column_name not in df.columns:
        return df
    non_null = df.filter(col(column_name).isNotNull())
    if non_null.count() == 0:
        return df
    median_value = non_null.approxQuantile(column_name, [0.5], 0.01)[0]
    return df.fillna({column_name: median_value})


def _impute_numeric_medians(df, exclude_columns):
    numeric_types = ("int", "bigint", "double", "float", "decimal", "long", "short")
    for field in df.schema.fields:
        if field.name in exclude_columns:
            continue
        if field.dataType.simpleString().split("(")[0] in numeric_types:
            df = _median_impute_column(df, field.name)
    return df


def _detect_categorical_columns(df, exclude_columns):
    categoricals = []
    for field in df.schema.fields:
        if field.name in exclude_columns:
            continue
        if isinstance(field.dataType, StringType):
            distinct_count = df.select(field.name).distinct().count()
            if 0 < distinct_count <= MAX_OHE_CARDINALITY:
                categoricals.append(field.name)
    return categoricals


def _one_hot_encode(df, categorical_columns):
    if not categorical_columns:
        return df

    indexers = [
        StringIndexer(
            inputCol=column_name,
            outputCol=f"{column_name}_idx",
            handleInvalid="keep",
        )
        for column_name in categorical_columns
    ]
    encoder = OneHotEncoder(
        inputCols=[f"{c}_idx" for c in categorical_columns],
        outputCols=[f"{c}_ohe" for c in categorical_columns],
        dropLast=True,
    )
    model = Pipeline(stages=indexers + [encoder]).fit(df)
    df = model.transform(df)
    drop_columns = list(categorical_columns) + [f"{c}_idx" for c in categorical_columns]
    return df.drop(*drop_columns)


def _aggregate_silver_clickstream(df, as_of_date_str):
    """Aggregate silver clickstream to one row per Customer_ID for as-of snapshot_date."""
    fe_cols = [c for c in df.columns if c.startswith("fe_")]
    agg_exprs = [
        F.count("*").alias("cs_row_count"),
        F.countDistinct("snapshot_date").alias("cs_distinct_snapshot_count"),
        F.max("snapshot_date").alias("cs_source_max_snapshot_date"),
    ]
    for fe_col in fe_cols:
        agg_exprs.append(F.avg(col(fe_col)).alias(f"{fe_col}_avg"))
        agg_exprs.append(F.sum(col(fe_col)).alias(f"{fe_col}_sum"))

    return (
        df.groupBy("Customer_ID")
        .agg(*agg_exprs)
        .withColumn("snapshot_date", F.lit(as_of_date_str).cast(DateType()))
    )


def _latest_pit_features(
    source_df, customer_col, snapshot_col, as_of_date, strict_before=False, snapshot_alias=None,
):
    as_of_lit = F.lit(as_of_date).cast(DateType())
    if strict_before:
        filtered = source_df.filter(col(snapshot_col) < as_of_lit)
    else:
        filtered = source_df.filter(col(snapshot_col) <= as_of_lit)

    window_spec = Window.partitionBy(customer_col).orderBy(col(snapshot_col).desc())
    result = (
        filtered.withColumn("_pit_rank", F.row_number().over(window_spec))
        .filter(col("_pit_rank") == 1)
        .drop("_pit_rank")
    )
    if snapshot_alias:
        result = result.withColumnRenamed(snapshot_col, snapshot_alias)
    else:
        result = result.drop(snapshot_col)
    return result


def _prefix_columns(df, prefix, skip_columns):
    for column_name in df.columns:
        if column_name not in skip_columns:
            df = df.withColumnRenamed(column_name, f"{prefix}{column_name}")
    return df


def _pit_join_labels_to_g1(labels_df, g1_df, snapshot_alias):
    """Point-in-time join: latest g1 row strictly before each label snapshot_date."""
    feature_cols = [c for c in g1_df.columns if c not in ("Customer_ID", "snapshot_date")]
    joined = labels_df.alias("lab").join(
        g1_df.alias("g1"),
        (col("lab.Customer_ID") == col("g1.Customer_ID"))
        & (col("g1.snapshot_date") < col("lab.snapshot_date")),
    )
    window_spec = Window.partitionBy("loan_id").orderBy(col("g1.snapshot_date").desc())
    pit = joined.withColumn("_pit_rank", F.row_number().over(window_spec)).filter(col("_pit_rank") == 1)

    select_exprs = [
        col("lab.loan_id").alias("loan_id"),
        col("lab.Customer_ID").alias("Customer_ID"),
        col("lab.label").alias("label"),
        col("lab.label_def").alias("label_def"),
        col("lab.snapshot_date").alias("label_snapshot_date"),
        col("g1.snapshot_date").alias(snapshot_alias),
    ] + [col(f"g1.{c}").alias(c) for c in feature_cols]
    return pit.select(*select_exprs)


# ---------------------------------------------------------------------------
# Label store (Lab 2)
# ---------------------------------------------------------------------------

def process_labels_gold_table(
    snapshot_date_str,
    silver_loan_daily_directory,
    gold_label_store_directory,
    spark,
    dpd=30,
    mob=6,
):
    df = _read_silver(spark, silver_loan_daily_directory, "silver_loan_daily", snapshot_date_str)
    df = df.filter(col("mob") == mob)
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(f"{dpd}dpd_{mob}mob").cast(StringType()))
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")
    path = _partition_path(gold_label_store_directory, "gold_label_store", snapshot_date_str)
    return _save_parquet(df, path)


# ---------------------------------------------------------------------------
# Gold Layer 1
# ---------------------------------------------------------------------------

def process_g1_attributes_table(
    snapshot_date_str, silver_attributes_directory, gold_g1_attributes_directory, spark
):
    df = _read_silver(spark, silver_attributes_directory, "silver_attributes", snapshot_date_str)
    drop_cols = [c for c in PII_COLUMNS if c in df.columns]
    df = df.drop(*drop_cols)

    df = _median_impute_column(df, "Age")
    df = _one_hot_encode(df, _detect_categorical_columns(df, set(KEY_COLUMNS)))

    path = _partition_path(gold_g1_attributes_directory, "g1_attributes", snapshot_date_str)
    return _save_parquet(df, path)


def process_g1_financials_table(
    snapshot_date_str, silver_financials_directory, gold_g1_financials_directory, spark
):
    df = _read_silver(spark, silver_financials_directory, "silver_financials", snapshot_date_str)

    if "Type_of_Loan" in df.columns:
        df = df.drop("Type_of_Loan")

    df = _impute_numeric_medians(df, exclude_columns=set(KEY_COLUMNS))
    df = _one_hot_encode(df, _detect_categorical_columns(df, set(KEY_COLUMNS)))

    path = _partition_path(gold_g1_financials_directory, "g1_financials", snapshot_date_str)
    return _save_parquet(df, path)


def process_g1_clickstream_table(
    snapshot_date_str,
    silver_clickstream_directory,
    gold_g1_clickstream_directory,
    spark,
    clickstream_window_days=30,
):
    """
    Gold layer 1 clickstream: rolling-window aggregation from silver, one row per Customer_ID.
    snapshot_date on output = as-of date (end of window).
    """
    df = _read_silver_clickstream_window(
        spark, silver_clickstream_directory, snapshot_date_str, clickstream_window_days
    )
    df = _impute_numeric_medians(df, exclude_columns=set(KEY_COLUMNS))
    df = _aggregate_silver_clickstream(df, snapshot_date_str)

    path = _partition_path(gold_g1_clickstream_directory, "g1_clickstream", snapshot_date_str)
    return _save_parquet(df, path)


# ---------------------------------------------------------------------------
# Gold Layer 2
# ---------------------------------------------------------------------------

def _build_g2_features(
    spark,
    snapshot_date_str,
    gold_g1_clickstream_directory,
    gold_g1_attributes_directory,
    gold_g1_financials_directory,
    strict_before=False,
):
    as_of_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d").date()

    cs_path = _partition_path(gold_g1_clickstream_directory, "g1_clickstream", snapshot_date_str)
    cs_features = _read_parquet(spark, cs_path)
    cs_features = cs_features.withColumn("cs_as_of_snapshot_date", col("snapshot_date"))

    attributes_all = _read_all_partitions(spark, gold_g1_attributes_directory, "g1_attributes")
    financials_all = _read_all_partitions(spark, gold_g1_financials_directory, "g1_financials")

    attr_features = _latest_pit_features(
        attributes_all,
        "Customer_ID",
        "snapshot_date",
        as_of_date,
        strict_before=strict_before,
        snapshot_alias="att_snapshot_date",
    )
    fin_features = _latest_pit_features(
        financials_all,
        "Customer_ID",
        "snapshot_date",
        as_of_date,
        strict_before=strict_before,
        snapshot_alias="fin_snapshot_date",
    )

    attr_features = _prefix_columns(attr_features, "att_", {"Customer_ID", "att_snapshot_date"})
    fin_features = _prefix_columns(fin_features, "fin_", {"Customer_ID", "fin_snapshot_date"})

    features = (
        cs_features.join(attr_features, on="Customer_ID", how="left")
        .join(fin_features, on="Customer_ID", how="left")
        .withColumn("snapshot_date", F.lit(snapshot_date_str).cast(DateType()))
        .withColumn(
            "feature_snapshot_date",
            F.greatest(
                col("cs_as_of_snapshot_date"),
                col("att_snapshot_date"),
                col("fin_snapshot_date"),
            ),
        )
        .drop("cs_as_of_snapshot_date")
    )
    return features


def process_g2_feature_store(
    snapshot_date_str,
    clickstream_window_days,
    gold_g1_clickstream_directory,
    gold_g1_attributes_directory,
    gold_g1_financials_directory,
    gold_g2_feature_store_directory,
    spark,
):
    # clickstream_window_days applied in g1; g2 reads pre-aggregated partition for snapshot_date_str
    _ = clickstream_window_days
    features = _build_g2_features(
        spark,
        snapshot_date_str,
        gold_g1_clickstream_directory,
        gold_g1_attributes_directory,
        gold_g1_financials_directory,
        strict_before=False,
    )
    path = _partition_path(gold_g2_feature_store_directory, "g2_feature_store", snapshot_date_str)
    return _save_parquet(features, path)


def process_g2_training_table(
    clickstream_window_days,
    gold_g1_clickstream_directory,
    gold_g1_attributes_directory,
    gold_g1_financials_directory,
    gold_label_store_directory,
    gold_g2_training_table_directory,
    spark,
):
    _ = clickstream_window_days
    labels = _read_all_partitions(spark, gold_label_store_directory, "gold_label_store")
    clickstream_all = _read_all_partitions(spark, gold_g1_clickstream_directory, "g1_clickstream")
    attributes_all = _read_all_partitions(spark, gold_g1_attributes_directory, "g1_attributes")
    financials_all = _read_all_partitions(spark, gold_g1_financials_directory, "g1_financials")

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

    training = (
        cs_features.join(
            attr_features.drop("Customer_ID", "label", "label_def", "label_snapshot_date"),
            on="loan_id",
            how="left",
        )
        .join(
            fin_features.drop("Customer_ID", "label", "label_def", "label_snapshot_date"),
            on="loan_id",
            how="left",
        )
    )

    training = training.withColumn(
        "feature_snapshot_date",
        F.greatest(
            col("cs_snapshot_date"),
            col("att_snapshot_date"),
            col("fin_snapshot_date"),
        ),
    )
    training = training.filter(col("feature_snapshot_date") < col("label_snapshot_date"))

    guard_window = Window.partitionBy("loan_id", "label_snapshot_date").orderBy(
        col("feature_snapshot_date").desc()
    )
    training = (
        training.withColumn("_guard_rank", F.row_number().over(guard_window))
        .filter(col("_guard_rank") == 1)
        .drop("_guard_rank")
    )
    training = _flatten_array_columns(training)

    path = os.path.join(gold_g2_training_table_directory, "g2_training_table.parquet")
    return _save_parquet(training, path)
