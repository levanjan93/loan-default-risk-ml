"""Shared gold-layer feature engineering definitions (pandas + Spark)."""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

STANDARD_LOAN_TYPES = [
    "auto_loan",
    "credit_builder_loan",
    "debt_consolidation_loan",
    "home_equity_loan",
    "mortgage_loan",
    "payday_loan",
    "personal_loan",
    "student_loan",
]

PAYMENT_BEHAVIOUR_REGEX = re.compile(
    r"(Low|High)_spent_(Small|Medium|Large)_value", re.IGNORECASE
)
CREDIT_HISTORY_REGEX = re.compile(
    r"(?P<years>\d+)\s+Years?\s+and\s+(?P<months>\d+)\s+Months?", re.IGNORECASE
)

# Columns dropped after engineering (raw strings / PII / intermediates)
FINANCIAL_DROP_COLUMNS = [
    "Type_of_Loan",
    "Credit_History_Age",
    "Payment_Behaviour",
]
ATTRIBUTE_DROP_COLUMNS = ["Age"]  # replaced by Age_group OHE; raw Age kept until after binning

KNOWN_NUMERIC_COLUMNS = {
    "Age",
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "credit_history_months",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance",
    "log_Annual_Income",
    "debt_to_income",
    "Debt_to_Salary",
    "emi_to_salary",
    "balance_to_debt",
    "investment_rate",
    "inq_per_loan",
    "has_credit_limit_change",
    "Num_Fin_Pdts",
    "Loans_per_Credit_Item",
    "Repayment_Ability",
    "Loan_Extent",
    "num_loan_types",
    *{f"lt_{loan_type}" for loan_type in STANDARD_LOAN_TYPES},
    *[f"fe_{i}" for i in range(1, 21)],
    *[f"fe_{i}_avg" for i in range(1, 21)],
    *[f"click_{i}m" for i in range(1, 7)],
    "cs_row_count",
    "cs_distinct_snapshot_count",
}


def tokenize_loan_types(type_of_loan) -> list[str]:
    if type_of_loan is None or (isinstance(type_of_loan, float) and np.isnan(type_of_loan)):
        return []
    text = str(type_of_loan).strip()
    if not text or text.upper() in {"NULL", "NAN"}:
        return []
    tokens = re.split(r", |, and ", text)
    cleaned = []
    for token in tokens:
        norm = re.sub(r"[^a-z0-9]+", "_", token.strip().lower()).strip("_")
        if norm and norm not in {"null", "not_specified", "nan"}:
            cleaned.append(norm)
    return cleaned


def parse_credit_history_months(value) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    match = CREDIT_HISTORY_REGEX.search(str(value))
    if not match:
        return None
    years = int(match.group("years"))
    months = int(match.group("months"))
    return float(years * 12 + months)


def split_payment_behaviour_series(series: pd.Series) -> pd.DataFrame:
    spent = pd.Series(index=series.index, dtype=object)
    value = pd.Series(index=series.index, dtype=object)
    for idx, raw in series.items():
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            continue
        match = PAYMENT_BEHAVIOUR_REGEX.search(str(raw))
        if match:
            spent.at[idx] = match.group(1).title()
            value.at[idx] = match.group(2).title()
    return pd.DataFrame(
        {
            "Payment_Behaviour_Spent": spent,
            "Payment_Behaviour_Value": value,
        }
    )


def engineer_loan_type_columns(df: pd.DataFrame, type_col: str = "Type_of_Loan") -> pd.DataFrame:
    out = df.copy()
    counts = out[type_col].apply(tokenize_loan_types).apply(Counter)
    out["num_loan_types"] = counts.apply(lambda counter: int(sum(counter.values())))
    for loan_type in STANDARD_LOAN_TYPES:
        out[f"lt_{loan_type}"] = counts.apply(lambda counter, lt=loan_type: int(counter.get(lt, 0)))
    return out


def engineer_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "Credit_History_Age" in out.columns:
        out["credit_history_months"] = out["Credit_History_Age"].apply(parse_credit_history_months)

    if "Payment_Behaviour" in out.columns:
        split_cols = split_payment_behaviour_series(out["Payment_Behaviour"])
        out = pd.concat([out, split_cols], axis=1)

    if "Type_of_Loan" in out.columns:
        out = engineer_loan_type_columns(out)

    if "Delay_from_due_date" in out.columns:
        out["Delay_from_due_date"] = pd.to_numeric(out["Delay_from_due_date"], errors="coerce")

    # Safe numeric helpers
    def _num(col_name: str) -> pd.Series:
        if col_name not in out.columns:
            return pd.Series(0.0, index=out.index)
        return pd.to_numeric(out[col_name], errors="coerce").fillna(0.0)

    annual_income = _num("Annual_Income")
    monthly_salary = _num("Monthly_Inhand_Salary")
    outstanding_debt = _num("Outstanding_Debt")
    total_emi = _num("Total_EMI_per_month")
    amount_invested = _num("Amount_invested_monthly")
    monthly_balance = _num("Monthly_Balance")
    num_bank = _num("Num_Bank_Accounts")
    num_card = _num("Num_Credit_Card")
    num_loan = _num("Num_of_Loan")
    num_inq = _num("Num_Credit_Inquiries")
    changed_limit = _num("Changed_Credit_Limit")
    delay_due = _num("Delay_from_due_date")

    out["log_Annual_Income"] = np.log1p(annual_income.clip(lower=0))
    out["debt_to_income"] = outstanding_debt / (annual_income + 1.0)
    out["Debt_to_Salary"] = outstanding_debt / (monthly_salary + 1.0)
    out["emi_to_salary"] = total_emi / (monthly_salary + 1.0)
    out["balance_to_debt"] = (monthly_balance + 1.0) / (outstanding_debt + 1.0)
    out["investment_rate"] = amount_invested / (monthly_salary + 1.0)
    out["inq_per_loan"] = num_inq / (num_loan + 1.0)
    out["has_credit_limit_change"] = (changed_limit != 0).astype(int)
    out["Num_Fin_Pdts"] = num_bank + num_card + num_loan
    out["Loans_per_Credit_Item"] = num_loan / (num_bank + num_card + 1.0)
    out["Repayment_Ability"] = monthly_salary - total_emi
    out["Loan_Extent"] = delay_due * num_loan

    drop_cols = [c for c in FINANCIAL_DROP_COLUMNS if c in out.columns]
    return out.drop(columns=drop_cols)


def engineer_age_group(df: pd.DataFrame, age_col: str = "Age") -> pd.DataFrame:
    out = df.copy()
    age = pd.to_numeric(out[age_col], errors="coerce")
    out["Age_group"] = pd.cut(
        age,
        bins=[-np.inf, 25, 40, 60, np.inf],
        labels=["<25", "25-39", "40-59", "60+"],
    ).astype(object)
    return out


def fill_categorical_nulls(df: pd.DataFrame, columns: list[str], fill_value: str = "Unknown") -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = out[column].fillna(fill_value).replace("", fill_value)
    return out


def month_lag_dates(as_of_date_str: str, lag_months: int = 6) -> list[tuple[int, str]]:
    as_of = datetime.strptime(as_of_date_str, "%Y-%m-%d").date()
    lags = []
    for months_ago in range(1, lag_months + 1):
        lag_date = as_of - relativedelta(months=months_ago)
        lags.append((months_ago, lag_date.strftime("%Y-%m-%d")))
    return lags


def aggregate_monthly_click_lags_pandas(
    silver_clickstream_directory: str,
    as_of_date_str: str,
    list_partition_paths,
    read_pandas_partition,
    partition_path,
    lag_months: int = 6,
) -> pd.DataFrame:
    """Pivot mean fe_1 for each of the prior *lag_months* monthly snapshots."""
    prefix = "silver_clickstream"
    frames = []
    for months_ago, lag_date in month_lag_dates(as_of_date_str, lag_months):
        path = partition_path(silver_clickstream_directory, prefix, lag_date)
        try:
            part = read_pandas_partition(path)
        except Exception:
            continue
        if part.empty:
            continue
        agg = (
            part.groupby("Customer_ID", as_index=False)["fe_1"]
            .mean()
            .rename(columns={"fe_1": f"click_{months_ago}m"})
        )
        frames.append(agg)

    if not frames:
        return pd.DataFrame(columns=["Customer_ID"] + [f"click_{i}m" for i in range(1, lag_months + 1)])

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="Customer_ID", how="outer")

    for months_ago in range(1, lag_months + 1):
        col = f"click_{months_ago}m"
        if col not in merged.columns:
            merged[col] = np.nan
    return merged
