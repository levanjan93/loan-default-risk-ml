import argparse
import os
import sys
from datetime import datetime

import pyspark

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import utils.data_processing_bronze_table as bronze
import utils.data_processing_gold_table as gold
import utils.data_processing_silver_table as silver


def use_pandas_gold_backend():
    """PySpark Python workers are unreliable on Windows with Python 3.13+."""
    return sys.platform == "win32" and sys.version_info >= (3, 13)


def generate_first_of_month_dates(start_date_str, end_date_str):
    """Return the first day of each month between start_date and end_date (inclusive)."""
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    first_of_month_dates = []
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates


def parse_args():
    parser = argparse.ArgumentParser(description="Medallion datamart pipeline (phase 1)")
    parser.add_argument(
        "--snapshotdate",
        default=None,
        help="As-of date for g2_feature_store (YYYY-MM-DD). If omitted, g2 runs for all months.",
    )
    parser.add_argument(
        "--clickstream-window-days",
        type=int,
        default=30,
        help="Lookback window (days) for clickstream aggregation in gold layer 1.",
    )
    parser.add_argument(
        "--training",
        action="store_true",
        help="Build g2_training_table (point-in-time join with label store).",
    )
    parser.add_argument(
        "--skip-bronze-silver",
        action="store_true",
        help="Skip bronze/silver and only run gold (requires existing silver tables).",
    )
    parser.add_argument(
        "--training-only",
        action="store_true",
        help="Build g2_training_table only (requires existing gold g1 + label store).",
    )
    parser.add_argument(
        "--gold-g2-only",
        action="store_true",
        help="Run only g2 feature store (+ training if --training); requires existing gold g1.",
    )
    return parser.parse_args()


args = parse_args()

start_date_str = "2023-01-01"
end_date_str = "2024-12-01"
dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
g2_dates = [args.snapshotdate] if args.snapshotdate else dates_str_lst

print("Monthly snapshot dates:", dates_str_lst)
print("Gold layer 2 as-of dates:", g2_dates)
print("Clickstream window (days):", args.clickstream_window_days)
if use_pandas_gold_backend():
    print("Gold backend: pandas (Windows + Python 3.13 compatibility mode)")
else:
    print("Gold backend: PySpark")

BRONZE_DIRS = {
    "lms": "datamart/bronze/lms/",
    "clickstream": "datamart/bronze/clickstream/",
    "attributes": "datamart/bronze/attributes/",
    "financials": "datamart/bronze/financials/",
}
SILVER_DIRS = {
    "lms": "datamart/silver/loan_daily/",
    "clickstream": "datamart/silver/clickstream/",
    "attributes": "datamart/silver/attributes/",
    "financials": "datamart/silver/financials/",
}
GOLD_DIRS = {
    "label_store": "datamart/gold/label_store/",
    "g1_attributes": "datamart/gold/g1_attributes/",
    "g1_financials": "datamart/gold/g1_financials/",
    "g1_clickstream": "datamart/gold/g1_clickstream/",
    "g2_feature_store": "datamart/gold/g2_feature_store/",
    "g2_training_table": "datamart/gold/g2_training_table/",
}

for directory in list(BRONZE_DIRS.values()) + list(SILVER_DIRS.values()) + list(GOLD_DIRS.values()):
    os.makedirs(directory, exist_ok=True)

gold_backend = gold
if use_pandas_gold_backend():
    import utils.gold_pandas as gold_backend

spark = None
if not use_pandas_gold_backend() or not args.skip_bronze_silver:
    spark_master = "local[1]" if sys.platform == "win32" else "local[*]"
    spark = (
        pyspark.sql.SparkSession.builder
        .appName("loan_default_medallion_phase1")
        .master(spark_master)
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

if args.training_only:
    print("\n=== Gold layer 2 training table ===")
    gold_backend.process_g2_training_table(
        args.clickstream_window_days,
        GOLD_DIRS["g1_clickstream"],
        GOLD_DIRS["g1_attributes"],
        GOLD_DIRS["g1_financials"],
        GOLD_DIRS["label_store"],
        GOLD_DIRS["g2_training_table"],
        spark,
    )
elif args.gold_g2_only:
    for date_str in g2_dates:
        print(f"\n=== Gold layer 2 feature store: {date_str} ===")
        gold_backend.process_g2_feature_store(
            date_str,
            args.clickstream_window_days,
            GOLD_DIRS["g1_clickstream"],
            GOLD_DIRS["g1_attributes"],
            GOLD_DIRS["g1_financials"],
            GOLD_DIRS["g2_feature_store"],
            spark,
        )
    if args.training:
        print("\n=== Gold layer 2 training table ===")
        gold_backend.process_g2_training_table(
            args.clickstream_window_days,
            GOLD_DIRS["g1_clickstream"],
            GOLD_DIRS["g1_attributes"],
            GOLD_DIRS["g1_financials"],
            GOLD_DIRS["label_store"],
            GOLD_DIRS["g2_training_table"],
            spark,
        )
else:
    if not args.skip_bronze_silver:
        for date_str in dates_str_lst:
            print(f"\n=== Bronze ingest: {date_str} ===")
            bronze.process_bronze_lms_table(date_str, BRONZE_DIRS["lms"], spark)
            bronze.process_bronze_clickstream_table(date_str, BRONZE_DIRS["clickstream"], spark)
            bronze.process_bronze_attributes_table(date_str, BRONZE_DIRS["attributes"], spark)
            bronze.process_bronze_financials_table(date_str, BRONZE_DIRS["financials"], spark)

            print(f"\n=== Silver transform: {date_str} ===")
            silver.process_silver_lms_table(date_str, BRONZE_DIRS["lms"], SILVER_DIRS["lms"], spark)
            silver.process_silver_clickstream_table(
                date_str, BRONZE_DIRS["clickstream"], SILVER_DIRS["clickstream"], spark
            )
            silver.process_silver_attributes_table(
                date_str, BRONZE_DIRS["attributes"], SILVER_DIRS["attributes"], spark
            )
            silver.process_silver_financials_table(
                date_str, BRONZE_DIRS["financials"], SILVER_DIRS["financials"], spark
            )

    for date_str in dates_str_lst:
        print(f"\n=== Gold label store: {date_str} ===")
        gold_backend.process_labels_gold_table(
            date_str, SILVER_DIRS["lms"], GOLD_DIRS["label_store"], spark, dpd=30, mob=6
        )

        print(f"\n=== Gold layer 1: {date_str} ===")
        gold_backend.process_g1_attributes_table(
            date_str, SILVER_DIRS["attributes"], GOLD_DIRS["g1_attributes"], spark
        )
        gold_backend.process_g1_financials_table(
            date_str, SILVER_DIRS["financials"], GOLD_DIRS["g1_financials"], spark
        )
        gold_backend.process_g1_clickstream_table(
            date_str,
            SILVER_DIRS["clickstream"],
            GOLD_DIRS["g1_clickstream"],
            spark,
            args.clickstream_window_days,
        )

    for date_str in g2_dates:
        print(f"\n=== Gold layer 2 feature store: {date_str} ===")
        gold_backend.process_g2_feature_store(
            date_str,
            args.clickstream_window_days,
            GOLD_DIRS["g1_clickstream"],
            GOLD_DIRS["g1_attributes"],
            GOLD_DIRS["g1_financials"],
            GOLD_DIRS["g2_feature_store"],
            spark,
        )

    if args.training:
        print("\n=== Gold layer 2 training table ===")
        gold_backend.process_g2_training_table(
            args.clickstream_window_days,
            GOLD_DIRS["g1_clickstream"],
            GOLD_DIRS["g1_attributes"],
            GOLD_DIRS["g1_financials"],
            GOLD_DIRS["label_store"],
            GOLD_DIRS["g2_training_table"],
            spark,
        )

if spark is not None:
    spark.stop()
print("\nPipeline complete.")
