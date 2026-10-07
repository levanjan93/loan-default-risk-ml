"""Run batch inference for loans on a given snapshot date."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from utils.config import BEST_MODEL_FILENAME, GOLD_PREDICTIONS_DIR, MODEL_STORE_DIR
from utils.ml.dataset import attach_loan_ids, coerce_feature_matrix, load_g2_feature_store


def run_inference(snapshot_date: str) -> Path:
    model_path = MODEL_STORE_DIR / BEST_MODEL_FILENAME
    partition_dir = GOLD_PREDICTIONS_DIR / f"snapshot_date={snapshot_date}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    output_path = partition_dir / "predictions.parquet"

    empty_cols = [
        "loan_id",
        "Customer_ID",
        "snapshot_date",
        "default_probability",
        "default_prediction",
        "decision_threshold",
        "model_name",
    ]

    if not model_path.exists():
        print(f"No trained model at {model_path}; skipping inference for {snapshot_date}.")
        pd.DataFrame(columns=empty_cols).to_parquet(output_path, index=False)
        return output_path

    features = load_g2_feature_store(snapshot_date)
    scored = attach_loan_ids(features, snapshot_date)
    if scored.empty:
        pd.DataFrame(columns=empty_cols).to_parquet(output_path, index=False)
        print(f"No scorable loans for {snapshot_date}; wrote empty predictions partition.")
        return output_path

    artefact = joblib.load(model_path)
    pipeline = artefact["pipeline"]
    feature_cols = artefact["feature_columns"]
    threshold = float(artefact.get("threshold", 0.5))

    missing_cols = [c for c in feature_cols if c not in scored.columns]
    for col in missing_cols:
        scored[col] = pd.NA

    X = coerce_feature_matrix(scored, feature_cols)
    proba = pipeline.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    out = scored[["loan_id", "Customer_ID", "snapshot_date"]].copy()
    out["default_probability"] = proba
    out["default_prediction"] = pred
    out["decision_threshold"] = threshold
    out["model_name"] = artefact["selected_model"]

    out.to_parquet(output_path, index=False)
    print(f"Saved predictions ({len(out)} rows) -> {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch model inference")
    parser.add_argument("--snapshotdate", required=True, help="Inference snapshot YYYY-MM-DD")
    args = parser.parse_args()
    run_inference(args.snapshotdate)


if __name__ == "__main__":
    main()
