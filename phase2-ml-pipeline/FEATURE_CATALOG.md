# Gold Layer Feature Catalog

This document describes every model-ready feature produced by the medallion pipeline (`g1_*` → `g2_feature_store` → `g2_training_table`).

**Total model features (intended): 93** — prefixed as they appear in `g2_training_table` / `g2_feature_store`.

> After changing feature engineering, rebuild **all** gold partitions so old schema columns (e.g. `att_Age`, `fin_Credit_History_Age`) do not leak in via `unionByName`:
>
> ```bash
> python -m utils.run_local_backfill --start 2023-01-01 --end 2024-06-01 --force-retrain
> ```

---

## Architecture overview

| Gold table | Grain | Role |
|------------|-------|------|
| `label_store` | `loan_id` × `snapshot_date` | 30 DPD @ MOB 6 default label |
| `g1_attributes` | `Customer_ID` × `snapshot_date` | Demographics |
| `g1_financials` | `Customer_ID` × `snapshot_date` | Credit & affordability |
| `g1_clickstream` | `Customer_ID` × `snapshot_date` | Engagement behaviour |
| `g2_feature_store` | `Customer_ID` × `snapshot_date` | PIT-joined inference features |
| `g2_training_table` | `loan_id` × `label_snapshot_date` | PIT-joined labelled training rows |

Prefix convention in G2:
- Clickstream features: **no prefix**
- Attribute features: `att_`
- Financial features: `fin_`

---

## 1. Clickstream features (28)

Built in `g1_clickstream` from a **30-day rolling window** plus **6-month monthly lags** on `fe_1`.

### Activity metadata (2)

| Feature | Description |
|---------|-------------|
| `cs_row_count` | Number of clickstream events in the 30-day window |
| `cs_distinct_snapshot_count` | Distinct snapshot dates contributing to the window |

### Monthly engagement lags (6)

Mean `fe_1` per customer for each of the prior 6 monthly silver partitions.

| Feature | Description |
|---------|-------------|
| `click_1m` | Mean `fe_1` from 1 month before as-of date |
| `click_2m` | Mean `fe_1` from 2 months before as-of date |
| `click_3m` | Mean `fe_1` from 3 months before as-of date |
| `click_4m` | Mean `fe_1` from 4 months before as-of date |
| `click_5m` | Mean `fe_1` from 5 months before as-of date |
| `click_6m` | Mean `fe_1` from 6 months before as-of date |

### Rolling-window aggregates (20)

For each `fe_1` … `fe_20` in the 30-day window:

| Pattern | Description |
|---------|-------------|
| `fe_{i}_avg` | Mean value of feature *i* |

> Sum columns removed because `fe_{i}_sum = fe_{i}_avg × cs_row_count`.

---

## 2. Attribute features (18)

Built in `g1_attributes`. Raw `Age` is binned; `Occupation` and `Age_group` are one-hot encoded (`drop_first=True`).

### Age group dummies (3)

| Feature | Category |
|---------|----------|
| `att_Age_group_<25` | Age &lt; 25 |
| `att_Age_group_40-59` | Age 40–59 |
| `att_Age_group_Unknown` | Missing / invalid age |

*Reference category (dropped): `60+`*

### Occupation dummies (15)

| Feature |
|---------|
| `att_Occupation_Architect` |
| `att_Occupation_Developer` |
| `att_Occupation_Doctor` |
| `att_Occupation_Engineer` |
| `att_Occupation_Entrepreneur` |
| `att_Occupation_Journalist` |
| `att_Occupation_Lawyer` |
| `att_Occupation_Manager` |
| `att_Occupation_Mechanic` |
| `att_Occupation_Media_Manager` |
| `att_Occupation_Musician` |
| `att_Occupation_Scientist` |
| `att_Occupation_Teacher` |
| `att_Occupation_Unknown` |
| `att_Occupation_Writer` |

*Reference category (dropped): first alphabetically among remaining levels in each snapshot.*

---

## 3. Financial features (47)

Built in `g1_financials`. Raw `Type_of_Loan`, `Credit_History_Age`, and combined `Payment_Behaviour` are parsed/dropped after engineering.

### Raw numeric (15)

| Feature | Source |
|---------|--------|
| `fin_Annual_Income` | Annual income |
| `fin_Monthly_Inhand_Salary` | Monthly in-hand salary |
| `fin_Num_Bank_Accounts` | Bank account count |
| `fin_Num_Credit_Card` | Credit card count |
| `fin_Interest_Rate` | Interest rate |
| `fin_Num_of_Loan` | Number of loans |
| `fin_Delay_from_due_date` | Days delayed from due date |
| `fin_Num_of_Delayed_Payment` | Count of delayed payments |
| `fin_Changed_Credit_Limit` | Credit limit change amount |
| `fin_Num_Credit_Inquiries` | Credit inquiry count |
| `fin_Outstanding_Debt` | Outstanding debt |
| `fin_Credit_Utilization_Ratio` | Utilization ratio |
| `fin_Total_EMI_per_month` | Total EMI per month |
| `fin_Amount_invested_monthly` | Monthly investment amount |
| `fin_Monthly_Balance` | Monthly balance |

### Engineered numeric (14)

| Feature | Formula / meaning |
|---------|-------------------|
| `fin_credit_history_months` | `years×12 + months` parsed from `Credit_History_Age` |
| `fin_num_loan_types` | Distinct loan types in `Type_of_Loan` |
| `fin_log_Annual_Income` | `log1p(Annual_Income)` |
| `fin_debt_to_income` | `Outstanding_Debt / (Annual_Income + 1)` |
| `fin_Debt_to_Salary` | `Outstanding_Debt / (Monthly_Inhand_Salary + 1)` |
| `fin_emi_to_salary` | `Total_EMI_per_month / (Monthly_Inhand_Salary + 1)` |
| `fin_balance_to_debt` | `(Monthly_Balance + 1) / (Outstanding_Debt + 1)` |
| `fin_investment_rate` | `Amount_invested_monthly / (Monthly_Inhand_Salary + 1)` |
| `fin_inq_per_loan` | `Num_Credit_Inquiries / (Num_of_Loan + 1)` |
| `fin_has_credit_limit_change` | 1 if `Changed_Credit_Limit ≠ 0`, else 0 |
| `fin_Num_Fin_Pdts` | `Num_Bank_Accounts + Num_Credit_Card + Num_of_Loan` |
| `fin_Loans_per_Credit_Item` | `Num_of_Loan / (Num_Bank_Accounts + Num_Credit_Card + 1)` |
| `fin_Repayment_Ability` | `Monthly_Inhand_Salary − Total_EMI_per_month` |
| `fin_Loan_Extent` | `Delay_from_due_date × Num_of_Loan` |

### Loan-type count features (8)

Count of each loan type token parsed from `Type_of_Loan`:

| Feature |
|---------|
| `fin_lt_auto_loan` |
| `fin_lt_credit_builder_loan` |
| `fin_lt_debt_consolidation_loan` |
| `fin_lt_home_equity_loan` |
| `fin_lt_mortgage_loan` |
| `fin_lt_payday_loan` |
| `fin_lt_personal_loan` |
| `fin_lt_student_loan` |

### Categorical dummies (10)

`Payment_Behaviour` is split into **Spent** (Low/High) and **Value** (Small/Medium/Large) before encoding.

| Feature | Group |
|---------|-------|
| `fin_Credit_Mix_Good` | Credit mix |
| `fin_Credit_Mix_Standard` | Credit mix |
| `fin_Credit_Mix_Unknown` | Credit mix |
| `fin_Payment_of_Min_Amount_Yes` | Min payment |
| `fin_Payment_of_Min_Amount_Unknown` | Min payment |
| `fin_Payment_Behaviour_Spent_Low` | Spend level |
| `fin_Payment_Behaviour_Spent_Unknown` | Spend level |
| `fin_Payment_Behaviour_Value_Small` | Payment size |
| `fin_Payment_Behaviour_Value_Medium` | Payment size |
| `fin_Payment_Behaviour_Value_Unknown` | Payment size |

*Reference categories (dropped): `Bad` (Credit_Mix), `No` (Payment_of_Min_Amount), `High` (Spent), `Large` (Value).*

---

## 4. Columns excluded from modelling

These appear in gold tables but are **not** passed to sklearn:

| Column | Reason |
|--------|--------|
| `loan_id`, `Customer_ID` | Identifiers |
| `snapshot_date`, `*_snapshot_date`, `feature_snapshot_date`, `label_snapshot_date` | Temporal keys |
| `label`, `label_def`, `default_flag` | Target / metadata |
| `cs_source_max_snapshot_date` | Date metadata |
| `att_Age`, `fin_Credit_History_Age` | Legacy raw fields (excluded after schema migration) |

---

## 5. Imputation and leakage

### What changed

Previously, the gold layer applied **partition-level median imputation** before train/test split. That can leak information because validation and test rows influence the imputation statistic.

**Current approach:**
- Gold layer stores **nulls** for missing numerics (categoricals filled with `"Unknown"` only for OHE stability).
- `utils/ml/train.py` fits `SimpleImputer(strategy="median")` **inside the sklearn `Pipeline` on the training fold only**, then applies those statistics to validation / inference.

### Does imputation cause train/test contamination?

| Stage | Leakage risk | Status |
|-------|--------------|--------|
| Gold-layer median impute (old) | **Yes** — statistics computed on all rows in partition | **Removed** |
| sklearn `SimpleImputer` in `Pipeline` | **No** — fit on `X_train` only | **Correct** |
| `StandardScaler` in `Pipeline` | **No** — fit on `X_train` only | **Correct** |
| OHE category levels from full partition | **Mild** — new categories in test map to `"Unknown"` / `handleInvalid=keep` | Acceptable for this assignment |

**Bottom line:** Your modelling path is safe **as long as you rely on the sklearn pipeline imputer** (which you do). The gold layer no longer pre-fills numeric medians. For production, persist the fitted pipeline (`model_store/best_model.joblib`) so inference uses **training-time** imputation statistics, not batch statistics from the scoring cohort.

---

## 6. Source files

| File | Purpose |
|------|---------|
| `utils/datamart/gold_feature_engineering.py` | Shared feature definitions & pandas helpers |
| `utils/datamart/gold_pandas.py` | Gold layer (Windows backend) |
| `utils/datamart/data_processing_gold_table.py` | Gold layer (Spark / Docker backend) |
| `utils/ml/dataset.py` | `select_feature_columns()` for training |
| `utils/ml/train.py` | Leakage-safe imputation + scaling in Pipeline |

---

## 7. Feature count summary

| Group | Count |
|-------|------:|
| Clickstream | 28 |
| Attributes (`att_`) | 18 |
| Financials (`fin_`) | 47 |
| **Total model features** | **93** |

Use feature importance analysis downstream to prune; this catalog prioritizes **breadth** over minimalism so you can compare against the top student submissions.
