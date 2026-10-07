"""Run medallion pipeline for a single monthly snapshot."""

from __future__ import annotations

import argparse

from utils.config import (
    BRONZE_DIRS,
    CLICKSTREAM_WINDOW_DAYS,
    DEFAULT_DPD_THRESHOLD_DAYS,
    DEFAULT_MOB_INSTALLMENT,
    GOLD_DIRS,
    SILVER_DIRS,
)
from utils.datamart import (
    data_processing_bronze_table as bronze,
    data_processing_silver_table as silver,
)
from utils.datamart.spark_session import create_spark, ensure_datamart_dirs, get_gold_backend


def run_medallion_snapshot(snapshot_date: str) -> None:
    ensure_datamart_dirs()
    spark = create_spark()
    gold = get_gold_backend()

    try:
        print(f"\n=== Bronze: {snapshot_date} ===")
        bronze.process_bronze_lms_table(snapshot_date, BRONZE_DIRS["lms"], spark)
        bronze.process_bronze_clickstream_table(snapshot_date, BRONZE_DIRS["clickstream"], spark)
        bronze.process_bronze_attributes_table(snapshot_date, BRONZE_DIRS["attributes"], spark)
        bronze.process_bronze_financials_table(snapshot_date, BRONZE_DIRS["financials"], spark)

        print(f"\n=== Silver: {snapshot_date} ===")
        silver.process_silver_lms_table(
            snapshot_date, BRONZE_DIRS["lms"], SILVER_DIRS["lms"], spark
        )
        silver.process_silver_clickstream_table(
            snapshot_date, BRONZE_DIRS["clickstream"], SILVER_DIRS["clickstream"], spark
        )
        silver.process_silver_attributes_table(
            snapshot_date, BRONZE_DIRS["attributes"], SILVER_DIRS["attributes"], spark
        )
        silver.process_silver_financials_table(
            snapshot_date, BRONZE_DIRS["financials"], SILVER_DIRS["financials"], spark
        )

        print(f"\n=== Gold: {snapshot_date} ===")
        gold.process_labels_gold_table(
            snapshot_date,
            SILVER_DIRS["lms"],
            GOLD_DIRS["label_store"],
            spark,
            dpd=DEFAULT_DPD_THRESHOLD_DAYS,
            mob=DEFAULT_MOB_INSTALLMENT,
        )
        gold.process_g1_attributes_table(
            snapshot_date, SILVER_DIRS["attributes"], GOLD_DIRS["g1_attributes"], spark
        )
        gold.process_g1_financials_table(
            snapshot_date, SILVER_DIRS["financials"], GOLD_DIRS["g1_financials"], spark
        )
        gold.process_g1_clickstream_table(
            snapshot_date,
            SILVER_DIRS["clickstream"],
            GOLD_DIRS["g1_clickstream"],
            spark,
            CLICKSTREAM_WINDOW_DAYS,
        )
        gold.process_g2_feature_store(
            snapshot_date,
            CLICKSTREAM_WINDOW_DAYS,
            GOLD_DIRS["g1_clickstream"],
            GOLD_DIRS["g1_attributes"],
            GOLD_DIRS["g1_financials"],
            GOLD_DIRS["g2_feature_store"],
            spark,
        )
    finally:
        spark.stop()

    print(f"\nMedallion snapshot complete: {snapshot_date}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run medallion snapshot for one as-of date")
    parser.add_argument("--snapshotdate", required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    run_medallion_snapshot(args.snapshotdate)


if __name__ == "__main__":
    main()
