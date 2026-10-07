# Phase 1 — Medallion feature store

Builds a **monthly snapshot datamart** for cash-loan default modelling: bronze ingestion from four raw CSVs, silver cleansing, gold layer-1 aggregates, and gold layer-2 **point-in-time feature store** (`g2_feature_store`) plus optional **`g2_training_table`** (features joined to the default label at application).

The same medallion patterns are integrated into the Airflow pipeline in [`../phase2-ml-pipeline/`](../phase2-ml-pipeline/).

---

## Data sources (`data/`)

| File | Content |
|------|---------|
| `lms_loan_daily.csv` | Loan master & performance (LMS) |
| `features_financials.csv` | Bureau / financial attributes |
| `features_attributes.csv` | Customer demographics |
| `feature_clickstream.csv` | Digital engagement events |

---

## Run the pipeline

```bash
python -m pip install -r requirements.txt
```

**Full history (2023-01 → 2024-12), one month:**

```bash
python main.py --snapshotdate 2024-07-01 --training
```

**Gold layer 2 only** (silver already built):

```bash
python main.py --snapshotdate 2024-07-01 --training --skip-bronze-silver
```

**Training table only:**

```bash
python main.py --training-only
```

On **Windows with Python 3.13+**, gold layer 2 automatically uses the **Pandas backend** (PySpark worker issues); Linux/Docker uses PySpark as in the course labs.

---

## Key modules

| Module | Role |
|--------|------|
| `utils/data_processing_bronze_table.py` | Monthly bronze snapshots |
| `utils/data_processing_silver_table.py` | Typed, cleaned silver Parquet |
| `utils/data_processing_gold_table.py` | Gold L1 + L2 feature engineering |
| `utils/gold_pandas.py` | Pandas implementation of gold L2 |
| `main.py` | CLI entrypoint |

Exploratory work: `data_processing_main.ipynb`.

---

## Docker (optional)

```bash
docker compose build
docker compose up
```

JupyterLab is exposed per `docker-compose.yaml` for notebook-driven development.
