# Phase 2 — Production ML pipeline

End-to-end, monitored machine-learning pipeline that predicts **probability of default at loan application** (`30 DPD @ MOB 6`) for a cash-loan book. Orchestrated in **Apache Airflow**, runnable in **Docker**.

Part of the portfolio monorepo — overview and results: [`../README.md`](../README.md). Phase 1 feature store: [`../phase1-feature-store/`](../phase1-feature-store/).

**Presentation assets:** `docs/DECK_OUTLINE.md`, `outputs/charts/`, `notebooks/03_model_monitoring.ipynb`.

---

## Prerequisites

- **Docker Desktop** installed and running (Linux engine / WSL2 on Windows)
- Ports **8080** free
- Repo root contains: `data/` (4 CSVs), `datamart/`, `dags/`, `utils/`, `Dockerfile`, `docker-compose.yaml`, `requirements.txt`

> `docker compose` and `docker-compose` are equivalent on current Docker installs.

---

## Quick start (recommended)

From the repo root:

```bash
docker compose build
docker compose up
```

When the webserver is ready, look in the terminal for a banner:

```text
============================================================
  Airflow is ready — open in your browser:
  http://localhost:8080
  Login: admin / admin
  DAG:   loan_default_ml_pipeline
============================================================
```

There is no auto-clickable hyperlink — copy **`http://localhost:8080`** into Chrome/Edge, or type `localhost:8080` in the address bar.

1. Open **http://localhost:8080**
2. Sign in: **`admin`** / **`admin`** (created automatically on first init; Airflow 2.x requires login)
3. Confirm DAG **`loan_default_ml_pipeline`** appears
4. Click **▶ Trigger DAG** (play icon) — no date picker needed; runs snapshot **`2024-07-01`** by default
5. Watch **Grid** — all 8 tasks should turn **success** (~2–4 min)

For a **specific month**, use **Backfill** (start = end = e.g. `2024-06-01`) or CLI:

```bash
docker compose exec airflow-scheduler airflow dags trigger loan_default_ml_pipeline --exec-date 2024-06-01
```

Run detached (optional):

```bash
docker compose up -d
docker compose down          # stop
docker compose down -v       # stop and wipe Airflow DB volume (fresh init)
```

---

## Airflow DAG

**DAG id:** `loan_default_ml_pipeline`  
**Schedule:** monthly (`0 0 1 * *`), 2023-01-01 → 2024-12-01  
**Catchup:** `False` (does not auto-queue 24 runs on startup; use manual trigger or backfill)

```
start
  → run_medallion_snapshot     # bronze → silver → gold for {{ ds }}
  → build_training_table       # rebuild g2_training_table
  → train_model                # champion to model_store/ (retrain Jan/Apr/Jul/Oct)
  → backfill_inference         # score all history ≤ {{ ds }}
  → run_inference              # score current month
  → run_monitoring             # metrics + charts
  → end
```

### Trigger one run (UI)

1. DAGs → **loan_default_ml_pipeline** → ▶ **Trigger DAG**
2. Set logical / execution date to a month with data, e.g. **`2024-07-01`**

### Backfill (UI or CLI)

Assignment requires the pipeline to be **backfillable** across time; it does not require auto-catchup on startup.

**UI:** DAG → **Backfill** → choose start/end (e.g. `2024-05-01` to `2024-06-01`)

**CLI:**

```bash
docker compose exec airflow-scheduler airflow dags backfill loan_default_ml_pipeline \
  -s 2024-05-01 -e 2024-06-01
```

---

## Verify outputs

After a successful run for snapshot `YYYY-MM-DD`:

| Path | Purpose |
|---|---|
| `model_store/best_model.joblib` | Champion model artefact |
| `model_store/model_metadata.json` | Metrics, threshold, feature list |
| `datamart/gold/predictions/snapshot_date=YYYY-MM-DD/predictions.parquet` | Batch scores (gold table) |
| `datamart/gold/monitoring/snapshot_date=YYYY-MM-DD/monitoring_metrics.parquet` | Performance & stability (gold table) |
| `outputs/charts/*.png` | Monitoring visualisations |
| `outputs/charts/monitoring_summary.json` | Latest metric snapshot for deck |

Quick check (host Python):

```bash
python -c "import pandas as pd; print(pd.read_parquet('datamart/gold/predictions/snapshot_date=2024-07-01/predictions.parquet').head())"
```

---

## What the pipeline does

1. **Datamart (medallion)** — Assignment 1 bronze → silver → gold; point-in-time `g2_feature_store` and labelled `g2_training_table`. Feature definitions: `FEATURE_CATALOG.md`.
2. **Train & select** — `utils/ml/train.py` loads tuned params from `model_store/best_params.json` and the promoted 36-feature LightGBM champion from `model_store/champion_config.json`; calibrates scores to true PD; persists to `model_store/`.
3. **Inference** — `utils/ml/inference.py` writes monthly partitions under `datamart/gold/predictions/`.
4. **Monitoring** — `utils/ml/monitor.py` writes `datamart/gold/monitoring/` and renders deck charts to `outputs/charts/` (AUC/Gini/KS/Brier, actual vs predicted default rate, score & feature PSI, approval rate, bad-rate among approved).

### Model governance (in DAG)

- **Retrain:** `--force-retrain` on **Jan, Apr, Jul, Oct** runs
- **Otherwise:** reuse existing champion if `model_store/best_model.joblib` exists
- **Drift / performance SOP:** PSI bands and refresh rules documented in deck (`docs/DECK_OUTLINE.md`)

---

## Run locally (no Airflow)

Useful for development and notebook work:

```bash
python -m pip install -r requirements.txt
python -m utils.run_local_backfill --start 2023-01-01 --end 2024-12-01 --force-retrain
```

Or run individual steps:

```bash
set PYTHONPATH=.          # PowerShell: $env:PYTHONPATH = (Get-Location).Path
python utils/datamart/run_snapshot.py --snapshotdate 2024-07-01
python utils/datamart/build_training_table.py
python utils/ml/train.py --snapshotdate 2024-07-01 --force-retrain
python utils/ml/inference.py --snapshotdate 2024-07-01
python utils/ml/monitor.py --snapshotdate 2024-07-01
```

---

## Notebooks (research → production)

```bash
python -m pip install -r requirements-dev.txt
```

| Notebook | Role |
|---|---|
| `notebooks/01_eda.ipynb` | EDA, label maturity, split design |
| `notebooks/02b_model_tuning_random_dev_oot.ipynb` | Tuning + SHAP feature reduction; exports `champion_config.json` |
| `notebooks/03_model_monitoring.ipynb` | Monitoring analysis aligned with production gold tables |

---

## Repository layout

```
dags/                      Airflow DAG
utils/datamart/            medallion bronze/silver/gold processing
utils/ml/                  train / inference / monitor / dataset helpers
model_store/               champion artefact + metadata + tuned params
datamart/                  bronze, silver, gold (predictions + monitoring)
data/                      raw CSV sources
outputs/charts/            monitoring plots for the deck
notebooks/                 EDA, tuning, monitoring
docs/DECK_OUTLINE.md       Task 2 slideument plan
FEATURE_CATALOG.md         feature dictionary
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `dockerDesktopLinuxEngine` not found | Start Docker Desktop; wait ~30s; retry |
| Build fails on `numpy` / Python version | Confirm `Dockerfile` uses `FROM apache/airflow:2.7.3-python3.10` |
| `libgomp.so.1` error when loading model | Rebuild image: `docker compose build --no-cache` (needs `libgomp1` in Dockerfile) |
| `train_model` instant, `run_inference` fails | Delete `model_store/best_model.joblib`; re-trigger with a **Jul/Oct/Jan/Apr** date |
| Login fails after many rebuilds | Reset volume: `docker compose down -v`, then `docker compose up` |
| Empty predictions for recent months | Ensure datamart has history; run medallion for that month first |

---

## Key design choices

| Decision | Rationale |
|---|---|
| LightGBM champion (36 SHAP features), LogReg benchmark | Strong tabular performance at ~9k rows; interpretable benchmark for governance |
| ROC-AUC / Gini primary metric | Standard scorecard discrimination; KS, PR-AUC, Brier as guardrails |
| Isotonic calibration | Scores usable as true PDs for expected-loss and pricing |
| Random dev + temporal OOT split, PIT joins | Reduces temporal and train-test leakage |
| Cost-based threshold (FN ≫ FP) | Missed default costs more than wrongful decline |
| PSI monitoring + quarterly retrain | Label-free drift detection; production refresh SOP |

---

## Deck / slide outline

`docs/DECK_OUTLINE.md` — slide-by-slide plan: architecture, design rationale, monitoring visuals, governance SOP.
