from datetime import datetime

from pyspark.sql.functions import col


def _ingest_bronze_partition(
    snapshot_date_str,
    csv_file_path,
    bronze_directory,
    partition_prefix,
    spark,
):
    """Ingest one snapshot_date partition from a raw CSV into the bronze datamart."""
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    df = (
        spark.read.csv(csv_file_path, header=True, inferSchema=True)
        .filter(col("snapshot_date") == snapshot_date)
    )
    print(snapshot_date_str, "row count:", df.count())

    partition_name = partition_prefix + "_" + snapshot_date_str.replace("-", "_") + ".csv"
    filepath = bronze_directory + partition_name
    df.toPandas().to_csv(filepath, index=False)
    print("saved to:", filepath)

    return df


def process_bronze_lms_table(snapshot_date_str, bronze_lms_directory, spark):
    """Bronze ingest for label-store source: lms_loan_daily.csv (same logic as Lab 2)."""
    return _ingest_bronze_partition(
        snapshot_date_str,
        "data/lms_loan_daily.csv",
        bronze_lms_directory,
        "bronze_loan_daily",
        spark,
    )


def process_bronze_clickstream_table(snapshot_date_str, bronze_clickstream_directory, spark):
    """Bronze ingest for feature-store source: feature_clickstream.csv."""
    return _ingest_bronze_partition(
        snapshot_date_str,
        "data/feature_clickstream.csv",
        bronze_clickstream_directory,
        "bronze_clickstream",
        spark,
    )


def process_bronze_attributes_table(snapshot_date_str, bronze_attributes_directory, spark):
    """Bronze ingest for feature-store source: features_attributes.csv."""
    return _ingest_bronze_partition(
        snapshot_date_str,
        "data/features_attributes.csv",
        bronze_attributes_directory,
        "bronze_attributes",
        spark,
    )


def process_bronze_financials_table(snapshot_date_str, bronze_financials_directory, spark):
    """Bronze ingest for feature-store source: features_financials.csv."""
    return _ingest_bronze_partition(
        snapshot_date_str,
        "data/features_financials.csv",
        bronze_financials_directory,
        "bronze_financials",
        spark,
    )


def process_bronze_table(snapshot_date_str, bronze_lms_directory, spark):
    """Backward-compatible alias used by Lab 2 silver/gold scripts for LMS bronze."""
    return process_bronze_lms_table(snapshot_date_str, bronze_lms_directory, spark)
