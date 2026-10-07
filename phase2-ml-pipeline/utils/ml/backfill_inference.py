"""Backfill predictions for historical g2 feature store partitions."""

from __future__ import annotations

import argparse
from datetime import datetime

import pandas as pd

from utils.config import BEST_MODEL_FILENAME, GOLD_PREDICTIONS_DIR, MODEL_STORE_DIR
from utils.ml.dataset import list_g2_feature_snapshots
from utils.ml.inference import run_inference


def backfill_inference(as_of: str, force: bool = False) -> None:
    if not (MODEL_STORE_DIR / BEST_MODEL_FILENAME).exists():
        print("No model available; skipping inference backfill.")
        return

    as_of_ts = datetime.strptime(as_of, "%Y-%m-%d")
    for snapshot in list_g2_feature_snapshots():
        snapshot_ts = datetime.strptime(snapshot, "%Y-%m-%d")
        if snapshot_ts > as_of_ts:
            continue
        pred_path = GOLD_PREDICTIONS_DIR / f"snapshot_date={snapshot}" / "predictions.parquet"
        if not force and pred_path.exists():
            try:
                existing = pd.read_parquet(pred_path)
                if len(existing) > 0:
                    continue
            except Exception:
                pass
        run_inference(snapshot)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill historical inference partitions")
    parser.add_argument("--snapshotdate", required=True, help="Backfill through YYYY-MM-DD")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-score all snapshots even when predictions already exist",
    )
    args = parser.parse_args()
    backfill_inference(args.snapshotdate, force=args.force)


if __name__ == "__main__":
    main()
