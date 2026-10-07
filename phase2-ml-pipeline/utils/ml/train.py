"""Train candidate models, select the best, and persist it to the model store.

Design (credit-risk / industry practice)
----------------------------------------
* **Train–Val–Test–OOT split** — matches ``notebooks/02b_model_tuning_random_dev_oot.ipynb``:
  random stratified train/val/test on development cohorts, with the most recent
  application cohorts sealed as temporal OOT. See ``utils.ml.dataset.random_dev_oot_split``.
* **Model family** — for ~6k rows of wide tabular credit data the field
  standard is a gradient-boosted tree challenger benchmarked against a
  regularised logistic-regression scorecard (interpretable, regulator-friendly)
  and gradient-boosted tree challengers. Deep learning is inappropriate at this volume.
* **Primary metric** — ROC-AUC (Gini = 2*AUC - 1), complemented by KS
  (regulatory), PR-AUC (imbalance) and Brier (calibration).
* **Calibration** — the winning model is wrapped in ``CalibratedClassifierCV``
  so its outputs are usable as true probabilities of default (PD) for
  expected-loss, pricing and provisioning.
* **Decision threshold** — chosen on validation to minimise expected business
  cost (a missed default costs far more than a wrongful rejection), not the
  naive 0.5 cut-off.

The reviewable hyper-parameter search lives in ``notebooks/02b_model_tuning_random_dev_oot.ipynb``;
that notebook exports ``model_store/best_params.json`` and, for the promoted
36-feature LightGBM champion, ``model_store/champion_config.json``. When the
latter is present this script trains that model directly instead of running a
full bake-off on all features.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.config import (
    BEST_MODEL_FILENAME,
    BEST_PARAMS_FILENAME,
    CALIBRATION_METHOD,
    CHAMPION_CONFIG_FILENAME,
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    DEV_TEST_FRAC,
    DEV_VAL_FRAC,
    MODEL_METADATA_FILENAME,
    MODEL_STORE_DIR,
    OOT_COHORTS,
    RANDOM_STATE,
    SELECTION_METRIC,
    TOP_MONITORED_FEATURES,
)
from utils.datamart.build_training_table import build_training_table
from utils.ml.dataset import (
    coerce_feature_matrix,
    ks_statistic,
    load_training_frame,
    random_dev_oot_split,
    select_feature_columns,
)
from utils.ml.importance import feature_importance_ranking


def _load_tuned_params() -> dict:
    """Hyper-parameters exported by the tuning notebook, if present."""
    path = MODEL_STORE_DIR / BEST_PARAMS_FILENAME
    if path.exists():
        try:
            params = json.loads(path.read_text(encoding="utf-8"))
            print(f"Loaded tuned hyper-parameters from {path}")
            return params
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Could not parse {path}: {exc}; using defaults.")
    return {}


def _load_champion_config() -> dict:
    """Feature-reduced champion exported by notebooks/02b (Section 12), if present."""
    path = MODEL_STORE_DIR / CHAMPION_CONFIG_FILENAME
    if not path.exists():
        return {}
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
        print(f"Loaded champion config from {path}")
        return config
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Could not parse {path}: {exc}; ignoring champion config.")
        return {}


def _resolve_feature_columns(df: pd.DataFrame, champion_cfg: dict) -> tuple[list[str], bool]:
    """Return model feature columns and whether a promoted champion config is active."""
    shortlist = champion_cfg.get("feature_columns") or champion_cfg.get("feature_shortlist")
    if not shortlist:
        return select_feature_columns(df), False

    available = set(df.columns)
    missing = [col for col in shortlist if col not in available]
    if missing:
        raise RuntimeError(
            "Champion config references columns missing from the training table: "
            + ", ".join(missing)
        )
    return list(shortlist), True


def _merge_tuned_params(tuned: dict, champion_cfg: dict, model_name: str) -> dict:
    merged = {k: dict(v) for k, v in tuned.items()}
    champion_params = champion_cfg.get("best_params") or {}
    if champion_params:
        merged[model_name] = {**merged.get(model_name, {}), **champion_params}
    return merged


def _numeric_preprocessor(feature_cols: list[str], scale: bool) -> ColumnTransformer:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(transformers=[("num", Pipeline(steps=steps), feature_cols)])


def _build_candidates(feature_cols: list[str], tuned: dict) -> dict[str, Pipeline]:
    lr_params = {"C": 1.0, **tuned.get("logistic_regression", {})}
    hgb_params = {
        "learning_rate": 0.05,
        "max_depth": 3,
        "max_iter": 400,
        "l2_regularization": 1.0,
        **tuned.get("hist_gradient_boosting", {}),
    }
    lgbm_params = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 3,
        "num_leaves": 31,
        "min_child_samples": 20,
        "reg_lambda": 1.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        **tuned.get("lightgbm", {}),
    }

    candidates = {
        # Interpretable, regulator-friendly scorecard baseline. Scaling matters
        # for the linear model; median-imputation keeps it leakage-safe.
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", _numeric_preprocessor(feature_cols, scale=True)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        **lr_params,
                    ),
                ),
            ]
        ),
        # Gradient-boosted trees: the industry workhorse for tabular credit risk.
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("preprocessor", _numeric_preprocessor(feature_cols, scale=False)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        random_state=RANDOM_STATE,
                        early_stopping=True,
                        validation_fraction=0.15,
                        **hgb_params,
                    ),
                ),
            ]
        ),
    }

    # Optional dev/notebook challenger — not in the production Docker image.
    try:
        from lightgbm import LGBMClassifier

        candidates["lightgbm"] = Pipeline(
            steps=[
                ("preprocessor", _numeric_preprocessor(feature_cols, scale=False)),
                (
                    "model",
                    LGBMClassifier(
                        objective="binary",
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        verbose=-1,
                        **lgbm_params,
                    ),
                ),
            ]
        )
    except ImportError:
        pass

    return candidates


def _evaluate(name: str, y_true: pd.Series, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    auc = float(roc_auc_score(y_true, y_score))
    return {
        "model_name": name,
        "roc_auc": auc,
        "gini": float(2 * auc - 1),
        "ks": ks_statistic(y_true, y_score),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "positive_rate": float(np.mean(y_true)),
        "n": int(len(y_true)),
    }


def _best_cost_threshold(y_true: pd.Series, y_score: np.ndarray) -> tuple[float, float]:
    """Threshold minimising expected business cost on the validation fold."""
    y_true = np.asarray(y_true).astype(int)
    grid = np.linspace(0.05, 0.95, 181)
    best_thr, best_cost = 0.5, np.inf
    for thr in grid:
        y_pred = (y_score >= thr).astype(int)
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        cost = COST_FALSE_NEGATIVE * fn + COST_FALSE_POSITIVE * fp
        if cost < best_cost:
            best_cost, best_thr = cost, float(thr)
    return best_thr, float(best_cost)


def train_models(as_of: str, force_retrain: bool = True) -> dict:
    MODEL_STORE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_STORE_DIR / BEST_MODEL_FILENAME
    metadata_path = MODEL_STORE_DIR / MODEL_METADATA_FILENAME

    if model_path.exists() and not force_retrain:
        print(f"Model already exists at {model_path}; skipping retrain.")
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    print("Rebuilding g2 training table from gold partitions...")
    build_training_table()

    df = load_training_frame(as_of)
    if df.empty or df["default_flag"].nunique() < 2:
        print(
            "Insufficient labeled training data for this snapshot; skipping train step. "
            "Backfill earlier datamart partitions first."
        )
        return {}

    champion_cfg = _load_champion_config()
    feature_cols, champion_promoted = _resolve_feature_columns(df, champion_cfg)
    if not feature_cols:
        raise RuntimeError("No numeric feature columns found for training.")

    tuned = _load_tuned_params()
    promoted_model = None
    if champion_promoted:
        promoted_model = champion_cfg.get("selected_model") or champion_cfg.get("champion_family")
        if not promoted_model:
            raise RuntimeError("Champion config must include `selected_model` or `champion_family`.")
        tuned = _merge_tuned_params(tuned, champion_cfg, promoted_model)
        print(
            f"Promoted champion from {CHAMPION_CONFIG_FILENAME}: "
            f"{promoted_model} on {len(feature_cols)} features"
        )

    # ---- Train–Val–Test–OOT split (same as 02b) ----
    train_df, val_df, test_df, oot_df = random_dev_oot_split(
        df,
        n_oot_cohorts=OOT_COHORTS,
        test_size=DEV_TEST_FRAC,
        val_size=DEV_VAL_FRAC,
        random_state=RANDOM_STATE,
    )

    def _xy(frame: pd.DataFrame):
        return coerce_feature_matrix(frame, feature_cols), frame["default_flag"].astype(int)

    X_train, y_train = _xy(train_df)
    X_val, y_val = _xy(val_df)

    print(
        f"Split -> train={len(X_train)} | val={len(X_val)} | test={len(test_df)} | "
        f"oot={len(oot_df)} (features={len(feature_cols)}, bad_rate={df['default_flag'].mean():.3f})"
    )

    # ---- Candidate bake-off on the validation cohort (skipped when champion is promoted) ----
    results = []
    best_name, best_auc, best_pipeline = None, -np.inf, None
    candidates = _build_candidates(feature_cols, tuned)

    if champion_promoted:
        if promoted_model not in candidates:
            raise RuntimeError(
                f"Promoted champion `{promoted_model}` is unavailable "
                "(install lightgbm via requirements-dev.txt if needed)."
            )
        best_name = promoted_model
        best_pipeline = candidates[best_name]
        best_pipeline.fit(X_train, y_train)
        val_score = best_pipeline.predict_proba(X_val)[:, 1]
        metrics = _evaluate(best_name, y_val, val_score)
        metrics["split"] = "validation"
        results.append(metrics)
        best_auc = metrics[SELECTION_METRIC]
        print(
            f"{best_name:>22}: AUC={metrics['roc_auc']:.4f} Gini={metrics['gini']:.4f} "
            f"KS={metrics['ks']:.4f} PR-AUC={metrics['pr_auc']:.4f} Brier={metrics['brier']:.4f}"
        )
        print(f"Using promoted `{best_name}` ({len(feature_cols)} features) from champion config")
    else:
        for name, pipeline in candidates.items():
            pipeline.fit(X_train, y_train)
            val_score = pipeline.predict_proba(X_val)[:, 1]
            metrics = _evaluate(name, y_val, val_score)
            metrics["split"] = "validation"
            results.append(metrics)
            print(
                f"{name:>22}: AUC={metrics['roc_auc']:.4f} Gini={metrics['gini']:.4f} "
                f"KS={metrics['ks']:.4f} PR-AUC={metrics['pr_auc']:.4f} Brier={metrics['brier']:.4f}"
            )
            if metrics[SELECTION_METRIC] > best_auc:
                best_auc, best_name, best_pipeline = metrics[SELECTION_METRIC], name, pipeline
        print(f"Selected `{best_name}` by {SELECTION_METRIC}={best_auc:.4f}")

    # ---- Decision threshold (on validation) ----
    val_score = best_pipeline.predict_proba(X_val)[:, 1]
    if champion_promoted and champion_cfg.get("decision_threshold") is not None:
        threshold = float(champion_cfg["decision_threshold"])
        exp_cost = float("nan")
        print(f"Using champion decision_threshold={threshold:.3f} from {CHAMPION_CONFIG_FILENAME}")
    else:
        threshold, exp_cost = _best_cost_threshold(y_val, val_score)
        print(f"Cost-optimal threshold={threshold:.3f} (FN:FP cost = {COST_FALSE_NEGATIVE}:{COST_FALSE_POSITIVE})")

    # ---- Refit on train+val and calibrate so scores read as true PDs ----
    X_fit = pd.concat([X_train, X_val], axis=0)
    y_fit = pd.concat([y_train, y_val], axis=0)
    base_pipeline = _build_candidates(feature_cols, tuned)[best_name]
    n_pos = int(y_fit.sum())
    calibrated = None
    if n_pos >= 30:
        try:
            calibrated = CalibratedClassifierCV(
                base_pipeline, method=CALIBRATION_METHOD, cv=3
            )
            calibrated.fit(X_fit, y_fit)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Calibration failed ({exc}); using uncalibrated model.")
            calibrated = None
    if calibrated is None:
        base_pipeline.fit(X_fit, y_fit)
    final_model = calibrated if calibrated is not None else base_pipeline

    # ---- Random test + temporal OOT evaluation (mirrors 02b Sections 6 & 11) ----
    random_test_metrics = {}
    if not test_df.empty and test_df["default_flag"].nunique() == 2:
        X_test, y_test = _xy(test_df)
        test_score = final_model.predict_proba(X_test)[:, 1]
        random_test_metrics = _evaluate(f"{best_name}_calibrated", y_test, test_score, threshold)
        random_test_metrics["split"] = "test"
        results.append(random_test_metrics)
        print(
            f"Random test -> AUC={random_test_metrics['roc_auc']:.4f} Gini={random_test_metrics['gini']:.4f} "
            f"KS={random_test_metrics['ks']:.4f} Brier={random_test_metrics['brier']:.4f}"
        )

    oot_metrics = {}
    if not oot_df.empty and oot_df["default_flag"].nunique() == 2:
        X_oot, y_oot = _xy(oot_df)
        oot_score = final_model.predict_proba(X_oot)[:, 1]
        oot_metrics = _evaluate(f"{best_name}_calibrated", y_oot, oot_score, threshold)
        oot_metrics["split"] = "oot"
        results.append(oot_metrics)
        print(
            f"OOT -> AUC={oot_metrics['roc_auc']:.4f} Gini={oot_metrics['gini']:.4f} "
            f"KS={oot_metrics['ks']:.4f} Brier={oot_metrics['brier']:.4f}"
        )

    artefact = {
        "pipeline": final_model,
        "feature_columns": feature_cols,
        "trained_as_of": as_of,
        "selected_model": best_name,
        "threshold": threshold,
        "calibrated": calibrated is not None,
        "calibration_method": CALIBRATION_METHOD if calibrated is not None else None,
    }
    joblib.dump(artefact, model_path)

    importance_ranking = feature_importance_ranking(model_path, feature_cols)
    monitored_features = importance_ranking.head(TOP_MONITORED_FEATURES).index.tolist()

    metadata = {
        "selected_model": best_name,
        "calibrated": calibrated is not None,
        "trained_as_of": as_of,
        "trained_at_utc": datetime.utcnow().isoformat(),
        "selection_metric": SELECTION_METRIC,
        "decision_threshold": threshold,
        "expected_validation_cost": None if np.isnan(exp_cost) else exp_cost,
        "cost_ratio_fn_fp": [COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE],
        "n_features": len(feature_cols),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(test_df)),
        "n_oot": int(len(oot_df)),
        "split_strategy": "random_dev_oot",
        "oot_cohorts": OOT_COHORTS,
        "dev_test_frac": DEV_TEST_FRAC,
        "dev_val_frac": DEV_VAL_FRAC,
        "overall_bad_rate": float(df["default_flag"].mean()),
        "tuned_params_used": bool(tuned),
        "champion_config_used": champion_promoted,
        "champion_config_source": champion_cfg.get("source_notebook"),
        "shap_keep_fraction": champion_cfg.get("shap_keep_fraction"),
        "candidate_results": results,
        "random_test_metrics": random_test_metrics,
        "oot_metrics": oot_metrics,
        "oot_test_metrics": oot_metrics,  # backward-compat alias
        "feature_columns": feature_cols,
        "feature_importance_ranking": importance_ranking.index.tolist(),
        "monitored_features": monitored_features,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved best model ({best_name}, calibrated={calibrated is not None}) -> {model_path}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and select best ML model")
    parser.add_argument("--snapshotdate", required=True, help="Train using data available as of YYYY-MM-DD")
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Overwrite existing model in model_store",
    )
    args = parser.parse_args()
    train_models(args.snapshotdate, force_retrain=args.force_retrain)


if __name__ == "__main__":
    main()
