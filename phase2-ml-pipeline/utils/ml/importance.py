"""Resolve champion-model feature importances for monitoring."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from utils.config import (
    BEST_MODEL_FILENAME,
    MODEL_METADATA_FILENAME,
    MODEL_STORE_DIR,
    TOP_MONITORED_FEATURES,
)


def _base_estimator(pipeline):
    """Return the fitted tree/linear estimator inside a saved artefact pipeline."""
    if hasattr(pipeline, "calibrated_classifiers_"):
        pipeline = pipeline.calibrated_classifiers_[0].estimator
    if hasattr(pipeline, "named_steps"):
        return pipeline.named_steps.get("model") or pipeline.steps[-1][1]
    return pipeline


def feature_importance_ranking(
    model_path: Path | None = None,
    feature_columns: list[str] | None = None,
) -> pd.Series:
    """Rank model inputs by split-based tree importance (LightGBM / sklearn GBM)."""
    model_path = model_path or (MODEL_STORE_DIR / BEST_MODEL_FILENAME)
    artefact = joblib.load(model_path)
    cols = feature_columns or artefact["feature_columns"]
    estimator = _base_estimator(artefact["pipeline"])
    if not hasattr(estimator, "feature_importances_"):
        raise TypeError(f"Estimator {type(estimator).__name__} has no feature_importances_")
    return pd.Series(estimator.feature_importances_, index=cols).sort_values(ascending=False)


def load_monitored_features(n: int = TOP_MONITORED_FEATURES) -> list[str]:
    """Top-*n* champion features to track with input-drift (PSI / CSI)."""
    metadata_path = MODEL_STORE_DIR / MODEL_METADATA_FILENAME
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        ranked = meta.get("monitored_features") or meta.get("feature_importance_ranking")
        if ranked:
            return list(ranked)[:n]

    ranking = feature_importance_ranking()
    return ranking.head(n).index.tolist()
