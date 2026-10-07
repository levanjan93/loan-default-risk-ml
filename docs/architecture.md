# Architecture & design notes

## Problem framing

- **Outcome:** binary default within the course label window (30+ DPD at MOB 6).
- **Use case:** application-time score and approve/decline policy with expected-loss interpretation (calibrated PD).

## Phase 1 — Why medallion + PIT?

| Layer | Purpose |
|-------|---------|
| **Bronze** | Immutable monthly extracts from LMS, financials, attributes, clickstream |
| **Silver** | Schema enforcement, deduplication, typing |
| **Gold L1** | Reusable aggregates (e.g. clickstream windows) |
| **Gold L2** | **Point-in-time** `g2_feature_store` — features as known at `snapshot_date` |
| **Training table** | Join features to labels without temporal leakage |

## Phase 2 — Production loop

The Airflow DAG `loan_default_ml_pipeline` (monthly):

1. `run_medallion_snapshot` — refresh bronze/silver/gold for `{{ ds }}`
2. `build_training_table` — rebuild `g2_training_table`
3. `train_model` — champion LightGBM + isotonic calibration → `model_store/`
4. `backfill_inference` / `run_inference` — partition scores under `datamart/gold/predictions/`
5. `run_monitoring` — performance, stability (PSI), policy KPIs → `datamart/gold/monitoring/` + charts

**Retrain policy:** forced refresh on Jan / Apr / Jul / Oct; otherwise reuse champion artefact if present.

## Modelling choices

| Decision | Rationale |
|----------|-----------|
| LightGBM + 36 SHAP features | Strong tabular performance at ~9k applications; reduced overfit vs full feature set |
| Random dev + temporal OOT | Discrimination on hold-out random split; generalization checked on future cohorts |
| Isotonic calibration | Scores approximate true PD for pricing / EL |
| Cost ratio FN:FP = 5:1 | Missing a default is costlier than a false decline → higher threshold (~0.67) |
| PSI on score + top features | Label-free drift detection between snapshots |

## Monitoring KPIs

- **Discrimination:** ROC-AUC, Gini, KS, Brier
- **Calibration:** actual vs predicted default rate
- **Stability:** score PSI, feature PSI (top SHAP drivers)
- **Policy:** approval rate, bad rate among approved

Detailed slide outline for presentations: `phase2-ml-pipeline/docs/DECK_OUTLINE.md`.
