"""Run the full pipeline locally (outside Docker) for quick smoke tests."""

from __future__ import annotations

import argparse
from datetime import datetime

from dateutil.relativedelta import relativedelta

from utils.datamart.run_snapshot import run_medallion_snapshot
from utils.ml.inference import run_inference
from utils.ml.monitor import run_monitoring
from utils.ml.train import train_models


def month_range(start: str, end: str) -> list[str]:
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    dates: list[str] = []
    current = start_dt
    while current <= end_dt:
        dates.append(current.strftime("%Y-%m-%d"))
        current += relativedelta(months=1)
    return dates


def backfill_inference_through(end_snapshot: str) -> None:
    from utils.ml.backfill_inference import backfill_inference

    backfill_inference(end_snapshot)


def run_backfill(start: str, end: str, force_retrain: bool = False) -> None:
    for snapshot in month_range(start, end):
        print(f"\n=== Processing {snapshot} ===")
        run_medallion_snapshot(snapshot)

    # Training table + model once all gold partitions through *end* exist.
    print(f"\n=== Training through {end} ===")
    train_models(end, force_retrain=force_retrain)
    backfill_inference_through(end)

    for snapshot in month_range(start, end):
        run_inference(snapshot)
        run_monitoring(snapshot)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local pipeline smoke test / backfill")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2024-06-01")
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()
    run_backfill(args.start, args.end, force_retrain=args.force_retrain)


if __name__ == "__main__":
    main()
