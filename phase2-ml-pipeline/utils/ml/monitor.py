"""Monitor model performance and stability over time, persisting a gold table.

Three monitoring lenses (standard MLOps for a deployed credit model):

1. **Performance** (needs matured labels): ROC-AUC / Gini, KS, Brier
   (calibration), and actual vs predicted default rate.
2. **Stability / drift** (available immediately, no labels): population
   stability index (PSI) of the score distribution and of key input features
   versus the training-baseline month.
3. **Business KPIs**: approval rate and the realised bad-rate among approved
   loans at the production decision threshold.

PSI governance bands (industry convention): < 0.10 stable, 0.10-0.25 watch,
> 0.25 significant shift -> investigate / retrain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import brier_score_loss, roc_auc_score

from utils.config import (
    CHARTS_DIR,
    GOLD_MONITORING_DIR,
    GOLD_PREDICTIONS_DIR,
    PSI_ALERT_THRESHOLD,
    PSI_WARN_THRESHOLD,
)
from utils.ml.dataset import (
    attach_loan_ids,
    ks_statistic,
    list_g2_feature_snapshots,
    load_g2_feature_store,
    load_training_frame,
)
from utils.ml.importance import load_monitored_features


def _load_predictions(as_of: pd.Timestamp) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not GOLD_PREDICTIONS_DIR.exists():
        return pd.DataFrame()

    for partition in sorted(GOLD_PREDICTIONS_DIR.glob("snapshot_date=*")):
        date_str = partition.name.split("=", 1)[1]
        if pd.to_datetime(date_str) > as_of:
            continue
        files = list(partition.glob("*.parquet"))
        if files:
            frames.append(pd.read_parquet(files[0]))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _psi(expected: pd.Series, actual: pd.Series, bins=10, quantile=False) -> float:
    """Population Stability Index between a baseline (expected) and current sample."""
    expected = pd.to_numeric(expected, errors="coerce").dropna()
    actual = pd.to_numeric(actual, errors="coerce").dropna()
    if len(expected) < 10 or len(actual) < 10:
        return float("nan")

    if quantile:
        breakpoints = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        if len(breakpoints) < 3:
            return float("nan")
        breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    else:
        breakpoints = np.linspace(0, 1, bins + 1)

    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    psi = 0.0
    for e, a in zip(expected_pct, actual_pct):
        e, a = max(e, 1e-4), max(a, 1e-4)  # avoid div/0 and log(0)
        psi += (a - e) * np.log(a / e)
    return float(psi)


def _psi_band(value: float) -> str:
    if np.isnan(value):
        return "n/a"
    if value >= PSI_ALERT_THRESHOLD:
        return "alert"
    if value >= PSI_WARN_THRESHOLD:
        return "watch"
    return "stable"


def _metric_row(snapshot_date, name, value, n, band=None) -> dict:
    row = {"snapshot_date": snapshot_date, "metric_name": name, "metric_value": float(value), "n_observations": int(n)}
    if band is not None:
        row["band"] = band
    return row


def _feature_psi(baseline: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """PSI on model inputs; uses a bounded null-rate proxy when coverage collapses."""
    baseline = pd.to_numeric(baseline, errors="coerce")
    current = pd.to_numeric(current, errors="coerce")
    null_shift = abs(float(current.isna().mean()) - float(baseline.isna().mean()))
    if null_shift >= 0.5 or current.notna().sum() < 10:
        return null_shift * PSI_ALERT_THRESHOLD

    fill = baseline.median()
    if pd.isna(fill):
        return float("nan")
    return _psi(baseline.fillna(fill), current.fillna(fill), bins=bins, quantile=True)


def _applicant_features(snapshot_date: str) -> pd.DataFrame:
    """Feature matrix for loans applying in *snapshot_date* (matches inference population)."""
    features = load_g2_feature_store(snapshot_date)
    return attach_loan_ids(features, snapshot_date)


def run_monitoring(snapshot_date: str) -> Path:
    as_of = pd.to_datetime(snapshot_date)
    partition_dir = GOLD_MONITORING_DIR / f"snapshot_date={snapshot_date}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    output_path = partition_dir / "monitoring_metrics.parquet"
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    predictions = _load_predictions(as_of)
    labeled = load_training_frame(snapshot_date)
    rows: list[dict] = []

    # ---- 1. Performance on loans whose labels have matured ----
    if not predictions.empty and not labeled.empty:
        merged = predictions.merge(labeled[["loan_id", "default_flag"]], on="loan_id", how="inner")
        eval_rows = merged.dropna(subset=["default_flag"])
        if len(eval_rows) >= 10 and eval_rows["default_flag"].nunique() == 2:
            y_true = eval_rows["default_flag"].astype(int)
            y_score = eval_rows["default_probability"]
            auc = float(roc_auc_score(y_true, y_score))
            n = len(eval_rows)
            rows += [
                _metric_row(snapshot_date, "roc_auc", auc, n),
                _metric_row(snapshot_date, "gini", 2 * auc - 1, n),
                _metric_row(snapshot_date, "ks", ks_statistic(y_true, y_score), n),
                _metric_row(snapshot_date, "brier_score", brier_score_loss(y_true, y_score), n),
                _metric_row(snapshot_date, "actual_default_rate", y_true.mean(), n),
                _metric_row(snapshot_date, "predicted_default_rate", (y_score >= 0.5).mean(), n),
            ]
            # Business KPIs at the production decision threshold.
            if "default_prediction" in eval_rows.columns:
                approved = eval_rows["default_prediction"] == 0
                approval_rate = float(approved.mean())
                bad_among_approved = float(y_true[approved].mean()) if approved.any() else float("nan")
                rows += [
                    _metric_row(snapshot_date, "approval_rate", approval_rate, n),
                    _metric_row(snapshot_date, "bad_rate_among_approved", bad_among_approved, n),
                ]

    # ---- 2a. Score-distribution stability vs the first scored month (baseline) ----
    if not predictions.empty:
        predictions["_month"] = pd.to_datetime(predictions["snapshot_date"]).dt.to_period("M")
        baseline_month = predictions["_month"].min()
        current_month = as_of.to_period("M")
        baseline = predictions.loc[predictions["_month"] == baseline_month, "default_probability"]
        current = predictions.loc[predictions["_month"] == current_month, "default_probability"]
        if not current.empty and current_month != baseline_month:
            score_psi = _psi(baseline, current, bins=10, quantile=False)
            rows.append(_metric_row(snapshot_date, "score_psi", score_psi, len(current), _psi_band(score_psi)))

    # ---- 2b. Input-feature drift on application cohort vs baseline month ----
    feature_snaps = [s for s in list_g2_feature_snapshots() if pd.to_datetime(s) <= as_of]
    monitored_features = load_monitored_features()
    if feature_snaps:
        baseline_snap = feature_snaps[0]
        baseline_apps = _applicant_features(baseline_snap)
        current_apps = _applicant_features(snapshot_date)
        if (
            not baseline_apps.empty
            and not current_apps.empty
            and baseline_snap != snapshot_date
        ):
            for feat in monitored_features:
                if feat in baseline_apps.columns and feat in current_apps.columns:
                    val = _feature_psi(baseline_apps[feat], current_apps[feat])
                    rows.append(
                        _metric_row(
                            snapshot_date,
                            f"feature_psi__{feat}",
                            val,
                            len(current_apps),
                            _psi_band(val),
                        )
                    )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_parquet(output_path, index=False)
    print(f"Saved monitoring metrics ({len(metrics_df)} rows) -> {output_path}")

    _save_monitoring_charts(snapshot_date)
    return output_path


def _save_monitoring_charts(snapshot_date: str) -> None:
    """Write cumulative monitoring charts for the deck."""
    metric_files = sorted(GOLD_MONITORING_DIR.glob("snapshot_date=*/monitoring_metrics.parquet"))
    if not metric_files:
        return
    history = pd.concat([pd.read_parquet(p) for p in metric_files], ignore_index=True)
    if history.empty:
        return

    history["snapshot_date"] = pd.to_datetime(history["snapshot_date"])
    sns.set_theme(style="whitegrid")

    # Single-series line charts.
    for metric in ["roc_auc", "gini", "ks", "brier_score", "score_psi", "approval_rate", "bad_rate_among_approved"]:
        subset = history[history["metric_name"] == metric].sort_values("snapshot_date")
        if subset.empty:
            continue
        plt.figure(figsize=(8, 4))
        sns.lineplot(data=subset, x="snapshot_date", y="metric_value", marker="o")
        if metric == "score_psi":
            plt.axhline(PSI_WARN_THRESHOLD, color="orange", ls="--", lw=1, label="watch (0.10)")
            plt.axhline(PSI_ALERT_THRESHOLD, color="red", ls="--", lw=1, label="alert (0.25)")
            plt.legend()
        plt.title(f"Model monitoring: {metric}")
        plt.xlabel("Snapshot date")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / f"{metric}_over_time.png", dpi=150)
        plt.close()

    # Actual vs predicted default rate (calibration drift) on one axis.
    act = history[history["metric_name"] == "actual_default_rate"].sort_values("snapshot_date")
    pred = history[history["metric_name"] == "predicted_default_rate"].sort_values("snapshot_date")
    if not act.empty and not pred.empty:
        plt.figure(figsize=(8, 4))
        plt.plot(act["snapshot_date"], act["metric_value"], marker="o", label="actual default rate")
        plt.plot(pred["snapshot_date"], pred["metric_value"], marker="s", label="predicted default rate")
        plt.title("Actual vs predicted default rate over time")
        plt.xlabel("Snapshot date")
        plt.ylabel("default rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / "actual_vs_predicted_default_rate.png", dpi=150)
        plt.close()

    # Feature drift heatmap (PSI by feature over time).
    feat = history[history["metric_name"].str.startswith("feature_psi__")].copy()
    if not feat.empty:
        feat["feature"] = feat["metric_name"].str.replace("feature_psi__", "", regex=False)
        pivot = feat.pivot_table(index="feature", columns="snapshot_date", values="metric_value")
        if not pivot.empty:
            plt.figure(figsize=(min(12, 2 + pivot.shape[1]), 4))
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="rocket_r", vmin=0, vmax=0.5,
                        cbar_kws={"label": "PSI"})
            plt.title("Input feature drift (PSI) vs baseline")
            plt.xlabel("Snapshot date")
            plt.ylabel("")
            plt.tight_layout()
            plt.savefig(CHARTS_DIR / "feature_psi_heatmap.png", dpi=150)
            plt.close()

    latest = history[history["snapshot_date"] == history["snapshot_date"].max()]
    summary = {
        "latest_snapshot_date": snapshot_date,
        "metrics_available": sorted(history["metric_name"].unique().tolist()),
        "latest_values": {
            r["metric_name"]: round(float(r["metric_value"]), 4) for _, r in latest.iterrows()
        },
        "chart_directory": str(CHARTS_DIR),
    }
    (CHARTS_DIR / "monitoring_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved monitoring charts -> {CHARTS_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor model performance and stability")
    parser.add_argument("--snapshotdate", required=True, help="Monitoring snapshot YYYY-MM-DD")
    args = parser.parse_args()
    run_monitoring(args.snapshotdate)


if __name__ == "__main__":
    main()
