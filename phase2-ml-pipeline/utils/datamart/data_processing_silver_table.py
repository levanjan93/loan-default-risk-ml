import os
import shutil
import sys

import pyspark.sql.functions as F
from pyspark.sql.functions import col, trim
from pyspark.sql.types import (
    DateType,
    DoubleType,
    FloatType,
    IntegerType,
    StringType,
)

from utils.datamart.parquet_io import save_pandas_partition, save_partition_pandas

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _bronze_csv_path(bronze_directory, bronze_prefix, snapshot_date_str):
    partition_name = bronze_prefix + "_" + snapshot_date_str.replace("-", "_") + ".csv"
    return bronze_directory + partition_name


def _silver_parquet_path(silver_directory, silver_prefix, snapshot_date_str):
    partition_name = silver_prefix + "_" + snapshot_date_str.replace("-", "_") + ".parquet"
    return silver_directory + partition_name


def _load_bronze_csv(spark, bronze_directory, bronze_prefix, snapshot_date_str):
    filepath = _bronze_csv_path(bronze_directory, bronze_prefix, snapshot_date_str)
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print("loaded from:", filepath, "row count:", df.count())
    return df


def _save_silver_parquet_pandas(df, filepath):
    """Write parquet without Hadoop winutils (Windows-friendly)."""
    save_pandas_partition(df.toPandas(), filepath)


def _save_silver_parquet(df, silver_directory, silver_prefix, snapshot_date_str):
    filepath = _silver_parquet_path(silver_directory, silver_prefix, snapshot_date_str)
    if sys.platform == "win32":
        # Spark parquet write needs winutils on Windows; Docker/Linux uses Spark native write.
        _save_silver_parquet_pandas(df, filepath)
    else:
        df.write.mode("overwrite").parquet(filepath)
    print("saved to:", filepath)
    return df


def _cast_columns(df, column_type_map):
    for column, new_type in column_type_map.items():
        if column in df.columns:
            df = df.withColumn(column, col(column).cast(new_type))
    return df


def _enforce_key_integrity(df, key_columns):
    """Drop rows with null/blank keys; trim string keys."""
    for key_col in key_columns:
        if key_col not in df.columns:
            continue
        df = df.withColumn(key_col, trim(col(key_col).cast(StringType())))
        df = df.filter(col(key_col).isNotNull() & (col(key_col) != ""))
    return df.dropna(subset=key_columns)


def _dedupe_keys(df, key_columns):
    return df.dropDuplicates(key_columns)


def _null_if_in(column_name, invalid_values):
    condition = col(column_name).isin(list(invalid_values))
    return F.when(condition, F.lit(None)).otherwise(col(column_name))


def _null_if_not_regex(column_name, pattern):
    return F.when(col(column_name).rlike(pattern), col(column_name)).otherwise(F.lit(None))


def _cast_numeric_string(column_name, numeric_type=DoubleType()):
    """Strip non-numeric characters, then cast; unparseable values become null."""
    cleaned = F.regexp_replace(trim(col(column_name).cast(StringType())), r"[^0-9.\-]", "")
    return F.when(
        (cleaned == "") | (cleaned == "-") | (cleaned == "."),
        F.lit(None),
    ).otherwise(cleaned.cast(numeric_type))


# ---------------------------------------------------------------------------
# Label store (LMS) — same logic as Lab 2
# ---------------------------------------------------------------------------

def process_silver_lms_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    df = _load_bronze_csv(spark, bronze_lms_directory, "bronze_loan_daily", snapshot_date_str)

    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }
    df = _cast_columns(df, column_type_map)

    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))
    df = df.withColumn(
        "installments_missed",
        F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType()),
    ).fillna({"installments_missed": 0})
    df = df.withColumn(
        "first_missed_date",
        F.when(
            col("installments_missed") > 0,
            F.add_months(col("snapshot_date"), -1 * col("installments_missed")),
        ).cast(DateType()),
    )
    df = df.withColumn(
        "dpd",
        F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date")))
        .otherwise(0)
        .cast(IntegerType()),
    )

    return _save_silver_parquet(df, silver_loan_daily_directory, "silver_loan_daily", snapshot_date_str)


def process_silver_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    """Backward-compatible alias for Lab 2 gold scripts."""
    return process_silver_lms_table(
        snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark
    )


# ---------------------------------------------------------------------------
# Feature store — clickstream
# ---------------------------------------------------------------------------

def process_silver_clickstream_table(
    snapshot_date_str, bronze_clickstream_directory, silver_clickstream_directory, spark
):
    df = _load_bronze_csv(spark, bronze_clickstream_directory, "bronze_clickstream", snapshot_date_str)

    key_columns = ["Customer_ID", "snapshot_date"]
    df = _enforce_key_integrity(df, key_columns)
    df = _dedupe_keys(df, key_columns)

    fe_columns = [f"fe_{i}" for i in range(1, 21)]
    column_type_map = {c: IntegerType() for c in fe_columns}
    column_type_map["Customer_ID"] = StringType()
    column_type_map["snapshot_date"] = DateType()
    df = _cast_columns(df, column_type_map)

    return _save_silver_parquet(df, silver_clickstream_directory, "silver_clickstream", snapshot_date_str)


# ---------------------------------------------------------------------------
# Feature store — attributes
# ---------------------------------------------------------------------------

def process_silver_attributes_table(
    snapshot_date_str, bronze_attributes_directory, silver_attributes_directory, spark
):
    df = _load_bronze_csv(spark, bronze_attributes_directory, "bronze_attributes", snapshot_date_str)

    key_columns = ["Customer_ID", "snapshot_date"]
    df = _enforce_key_integrity(df, key_columns)
    df = _dedupe_keys(df, key_columns)

    column_type_map = {
        "Customer_ID": StringType(),
        "Name": StringType(),
        "Age": IntegerType(),
        "SSN": StringType(),
        "Occupation": StringType(),
        "snapshot_date": DateType(),
    }
    df = _cast_columns(df, column_type_map)

    df = df.withColumn(
        "SSN",
        F.when(col("SSN").rlike(r"^\d{3}-\d{2}-\d{4}$"), col("SSN")).otherwise(F.lit(None)),
    )
    df = df.withColumn(
        "Age",
        F.when((col("Age") > 0) & (col("Age") < 120), col("Age")).otherwise(F.lit(None)),
    )
    df = df.withColumn(
        "Occupation",
        _null_if_in("Occupation", ["_______"]),
    )

    return _save_silver_parquet(df, silver_attributes_directory, "silver_attributes", snapshot_date_str)


# ---------------------------------------------------------------------------
# Feature store — financials
# ---------------------------------------------------------------------------

def process_silver_financials_table(
    snapshot_date_str, bronze_financials_directory, silver_financials_directory, spark
):
    df = _load_bronze_csv(spark, bronze_financials_directory, "bronze_financials", snapshot_date_str)

    key_columns = ["Customer_ID", "snapshot_date"]
    df = _enforce_key_integrity(df, key_columns)
    df = _dedupe_keys(df, key_columns)

    # Cast string/numeric columns to target types (invalid casts become null in Spark)
    column_type_map = {
        "Customer_ID": StringType(),
        "Annual_Income": StringType(),
        "Monthly_Inhand_Salary": DoubleType(),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": IntegerType(),
        "Num_of_Loan": StringType(),
        "Type_of_Loan": StringType(),
        "Delay_from_due_date": IntegerType(),
        "Num_of_Delayed_Payment": StringType(),
        "Changed_Credit_Limit": StringType(),
        "Num_Credit_Inquiries": DoubleType(),
        "Credit_Mix": StringType(),
        "Outstanding_Debt": StringType(),
        "Credit_Utilization_Ratio": DoubleType(),
        "Credit_History_Age": StringType(),
        "Payment_of_Min_Amount": StringType(),
        "Total_EMI_per_month": DoubleType(),
        "Amount_invested_monthly": StringType(),
        "Payment_Behaviour": StringType(),
        "Monthly_Balance": StringType(),
        "snapshot_date": DateType(),
    }
    df = _cast_columns(df, column_type_map)

    # Gibberish / sentinel → null (rules from EDA)
    df = df.withColumn("Credit_Mix", _null_if_in("Credit_Mix", ["_", ""]))
    df = df.withColumn("Payment_of_Min_Amount", _null_if_in("Payment_of_Min_Amount", ["NM", ""]))
    df = df.withColumn("Payment_Behaviour", _null_if_in("Payment_Behaviour", ["!@9#%8", ""]))
    df = df.withColumn("Type_of_Loan", _null_if_in("Type_of_Loan", ["NULL", ""]))

    df = df.withColumn("Credit_History_Age", trim(col("Credit_History_Age")))
    df = df.withColumn(
        "Credit_History_Age",
        _null_if_not_regex("Credit_History_Age", r"^\d+ Years and \d+ Months$"),
    )

    # Clean numeric strings then cast
    for numeric_col in [
        "Annual_Income",
        "Changed_Credit_Limit",
        "Outstanding_Debt",
        "Amount_invested_monthly",
        "Monthly_Balance",
    ]:
        df = df.withColumn(numeric_col, _cast_numeric_string(numeric_col, DoubleType()))

    df = df.withColumn("Num_of_Loan", _cast_numeric_string("Num_of_Loan", IntegerType()))
    df = df.withColumn(
        "Num_of_Delayed_Payment",
        _cast_numeric_string("Num_of_Delayed_Payment", IntegerType()),
    )

    # Plausible range checks for numeric fields
    df = df.withColumn(
        "Num_Bank_Accounts",
        F.when((col("Num_Bank_Accounts") >= 0) & (col("Num_Bank_Accounts") <= 20), col("Num_Bank_Accounts"))
        .otherwise(F.lit(None)),
    )
    df = df.withColumn(
        "Num_Credit_Card",
        F.when((col("Num_Credit_Card") >= 0) & (col("Num_Credit_Card") <= 20), col("Num_Credit_Card"))
        .otherwise(F.lit(None)),
    )
    df = df.withColumn(
        "Interest_Rate",
        F.when((col("Interest_Rate") >= 0) & (col("Interest_Rate") <= 100), col("Interest_Rate"))
        .otherwise(F.lit(None)),
    )
    df = df.withColumn(
        "Delay_from_due_date",
        F.when((col("Delay_from_due_date") >= 0) & (col("Delay_from_due_date") <= 365), col("Delay_from_due_date"))
        .otherwise(F.lit(None)),
    )
    df = df.withColumn(
        "Credit_Utilization_Ratio",
        F.when(
            (col("Credit_Utilization_Ratio") >= 0) & (col("Credit_Utilization_Ratio") <= 100),
            col("Credit_Utilization_Ratio"),
        ).otherwise(F.lit(None)),
    )

    return _save_silver_parquet(df, silver_financials_directory, "silver_financials", snapshot_date_str)
