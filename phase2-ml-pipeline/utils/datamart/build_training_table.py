"""Rebuild gold g2_training_table from all label + g1 partitions."""

from __future__ import annotations

import argparse

from utils.config import GOLD_DIRS
from utils.datamart.spark_session import create_spark, ensure_datamart_dirs, get_gold_backend


def build_training_table() -> None:
    ensure_datamart_dirs()
    spark = create_spark()
    gold = get_gold_backend()
    try:
        # Reuse the inference feature store, anchored to each loan's application
        # month, and attach the matured label -> leakage-safe, no train/serve skew.
        gold.process_g2_training_table(
            GOLD_DIRS["g2_feature_store"],
            GOLD_DIRS["label_store"],
            GOLD_DIRS["g2_training_table"],
            spark,
        )
    finally:
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build g2 training table")
    args = parser.parse_args()
    build_training_table()


if __name__ == "__main__":
    main()
