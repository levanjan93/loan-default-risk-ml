"""Spark session and gold-layer backend selection (PySpark vs pandas)."""

from __future__ import annotations

import os
import sys

import pyspark


def use_pandas_gold_backend() -> bool:
    """Use pandas gold on Windows (avoids PySpark worker / winutils issues)."""
    return sys.platform == "win32"


def create_spark():
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    spark_master = "local[1]" if sys.platform == "win32" else "local[*]"
    spark = (
        pyspark.sql.SparkSession.builder.appName("loan_default_medallion")
        .master(spark_master)
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def get_gold_backend():
    if use_pandas_gold_backend():
        import utils.datamart.gold_pandas as gold_backend
    else:
        import utils.datamart.data_processing_gold_table as gold_backend
    return gold_backend


def ensure_datamart_dirs() -> None:
    from utils.config import BRONZE_DIRS, GOLD_DIRS, SILVER_DIRS

    for directory in (
        *BRONZE_DIRS.values(),
        *SILVER_DIRS.values(),
        *GOLD_DIRS.values(),
    ):
        os.makedirs(directory, exist_ok=True)
