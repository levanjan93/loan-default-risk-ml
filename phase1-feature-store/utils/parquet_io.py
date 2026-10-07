"""Windows-safe parquet I/O (Spark cannot read pyarrow TIMESTAMP(NANOS))."""

import glob
import os
import shutil

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType


def _date_like_columns(columns):
    return [c for c in columns if "date" in c.lower()]


def normalize_dates_for_parquet(pdf: pd.DataFrame) -> pd.DataFrame:
    pdf = pdf.copy()
    for col_name in _date_like_columns(pdf.columns):
        pdf[col_name] = pd.to_datetime(pdf[col_name], errors="coerce").dt.strftime("%Y-%m-%d")
    return pdf


def _normalize_arrow_table(table: pa.Table) -> pa.Table:
    """Cast Arrow types that Spark cannot read into plain strings."""
    columns = {}
    for name in table.column_names:
        column = table[name]
        field_type = column.type
        if pa.types.is_timestamp(field_type) or pa.types.is_date(field_type):
            columns[name] = pc.cast(column, pa.string())
        elif pa.types.is_null(field_type):
            columns[name] = pa.array([""] * table.num_rows, type=pa.string())
        else:
            columns[name] = column
    return pa.table(columns)


def _prepare_pandas_for_spark(pdf: pd.DataFrame) -> pd.DataFrame:
    """Coerce pandas dtypes so Spark can load rows with an explicit schema."""
    pdf = pdf.copy()
    for col_name in _date_like_columns(pdf.columns):
        parsed = pd.to_datetime(pdf[col_name], errors="coerce")
        pdf[col_name] = parsed.dt.strftime("%Y-%m-%d").where(parsed.notna(), None)

    for col_name in pdf.columns:
        if pdf[col_name].isna().all():
            pdf[col_name] = ""
        elif str(pdf[col_name].dtype) == "float32":
            pdf[col_name] = pdf[col_name].astype("float64")
        elif str(pdf[col_name].dtype) in ("int16", "int32", "uint32"):
            pdf[col_name] = pdf[col_name].astype("int64")
        elif pdf[col_name].dtype == object or str(pdf[col_name].dtype) == "string":
            pdf[col_name] = pdf[col_name].map(
                lambda value: None
                if value is None or (isinstance(value, float) and pd.isna(value))
                else str(value)
            )
        if pdf[col_name].map(lambda value: isinstance(value, list)).any():
            pdf[col_name] = pdf[col_name].map(
                lambda value: ",".join(map(str, value)) if isinstance(value, list) else value
            )
    return pdf


def _pandas_to_spark_schema(pdf: pd.DataFrame) -> StructType:
    fields = []
    for col_name in pdf.columns:
        series = pdf[col_name]
        if col_name in _date_like_columns(pdf.columns) or pd.api.types.is_object_dtype(series):
            fields.append(StructField(col_name, StringType(), True))
        elif pd.api.types.is_integer_dtype(series):
            fields.append(StructField(col_name, LongType(), True))
        elif pd.api.types.is_float_dtype(series):
            fields.append(StructField(col_name, DoubleType(), True))
        else:
            fields.append(StructField(col_name, StringType(), True))
    return StructType(fields)


def _pandas_to_spark(spark, pdf: pd.DataFrame):
    pdf = _prepare_pandas_for_spark(pdf)
    schema = _pandas_to_spark_schema(pdf)
    records = []
    for row in pdf.itertuples(index=False, name=None):
        record = {}
        for idx, col_name in enumerate(pdf.columns):
            value = row[idx]
            if value is None or (isinstance(value, float) and pd.isna(value)):
                record[col_name] = None
            elif col_name in _date_like_columns(pdf.columns) or isinstance(value, str):
                record[col_name] = str(value) if value != "" else None
            else:
                record[col_name] = value
        records.append(record)
    return spark.createDataFrame(records, schema=schema)


def save_partition_pandas(df_spark, path: str) -> None:
    """Write Spark DataFrame to parquet without Hadoop winutils (Windows)."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

    temp_csv_dir = path + "_csv_export"
    if os.path.exists(temp_csv_dir):
        shutil.rmtree(temp_csv_dir)
    df_spark.coalesce(1).write.mode("overwrite").option("header", True).csv(temp_csv_dir)

    csv_files = sorted(glob.glob(os.path.join(temp_csv_dir, "part-*.csv")))
    pdf = pd.concat([pd.read_csv(csv_file) for csv_file in csv_files], ignore_index=True)
    shutil.rmtree(temp_csv_dir)

    pdf = normalize_dates_for_parquet(pdf)
    for col_name in pdf.columns:
        if pdf[col_name].isna().all():
            pdf[col_name] = ""
    pdf.to_parquet(
        os.path.join(path, "part-00000.parquet"),
        engine="pyarrow",
        index=False,
    )


def read_partition_pandas(spark, path: str):
    return _pandas_to_spark(spark, read_pandas_partition(path))


def read_pandas_partition(path: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(path, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files in {path}")
    tables = [_normalize_arrow_table(pq.read_table(file_path)) for file_path in files]
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    return _prepare_pandas_for_spark(table.to_pandas())


def save_pandas_partition(pdf: pd.DataFrame, path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    pdf = normalize_dates_for_parquet(pdf)
    for col_name in pdf.columns:
        if pdf[col_name].isna().all():
            pdf[col_name] = ""
    pdf.to_parquet(
        os.path.join(path, "part-00000.parquet"),
        engine="pyarrow",
        index=False,
    )


def list_partition_paths(base_directory: str, prefix: str):
    pattern = os.path.join(base_directory, f"{prefix}_*.parquet")
    paths = sorted(glob.glob(pattern))
    valid_paths = []
    for path in paths:
        if glob.glob(os.path.join(path, "*.parquet")):
            row_count = sum(
                pq.ParquetFile(file_path).metadata.num_rows
                for file_path in glob.glob(os.path.join(path, "*.parquet"))
            )
            if row_count > 0:
                valid_paths.append(path)
    return valid_paths
