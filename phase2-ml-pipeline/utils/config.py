"""Shared paths and constants for the Assignment 2 ML pipeline."""

import os
from pathlib import Path

# Project root inside Docker is /opt/airflow; locally it resolves to repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _datamart_path(path: Path) -> str:
    """Assignment 1 modules concatenate filenames; require trailing separator."""
    return os.path.join(str(path), "")


DATA_DIR = PROJECT_ROOT / "data"
DATAMART_DIR = PROJECT_ROOT / "datamart"
MODEL_STORE_DIR = PROJECT_ROOT / "model_store"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

BRONZE_DIR = DATAMART_DIR / "bronze"
SILVER_DIR = DATAMART_DIR / "silver"
GOLD_DIR = DATAMART_DIR / "gold"

# Assignment 1 medallion layout (string paths for PySpark modules)
BRONZE_DIRS = {
    "lms": _datamart_path(BRONZE_DIR / "lms"),
    "clickstream": _datamart_path(BRONZE_DIR / "clickstream"),
    "attributes": _datamart_path(BRONZE_DIR / "attributes"),
    "financials": _datamart_path(BRONZE_DIR / "financials"),
}
SILVER_DIRS = {
    "lms": _datamart_path(SILVER_DIR / "loan_daily"),
    "clickstream": _datamart_path(SILVER_DIR / "clickstream"),
    "attributes": _datamart_path(SILVER_DIR / "attributes"),
    "financials": _datamart_path(SILVER_DIR / "financials"),
}
GOLD_DIRS = {
    "label_store": _datamart_path(GOLD_DIR / "label_store"),
    "g1_attributes": _datamart_path(GOLD_DIR / "g1_attributes"),
    "g1_financials": _datamart_path(GOLD_DIR / "g1_financials"),
    "g1_clickstream": _datamart_path(GOLD_DIR / "g1_clickstream"),
    "g2_feature_store": _datamart_path(GOLD_DIR / "g2_feature_store"),
    "g2_training_table": _datamart_path(GOLD_DIR / "g2_training_table"),
}

GOLD_PREDICTIONS_DIR = GOLD_DIR / "predictions"
GOLD_MONITORING_DIR = GOLD_DIR / "monitoring"
CHARTS_DIR = OUTPUTS_DIR / "charts"

RAW_FILES = {
    "clickstream": DATA_DIR / "feature_clickstream.csv",
    "attributes": DATA_DIR / "features_attributes.csv",
    "financials": DATA_DIR / "features_financials.csv",
    "loans": DATA_DIR / "lms_loan_daily.csv",
}

# Business rules from Assignment 1 datamart
DEFAULT_MOB_INSTALLMENT = 6
DEFAULT_DPD_THRESHOLD_DAYS = 30
CLICKSTREAM_WINDOW_DAYS = 30
CLICKSTREAM_LAG_MONTHS = 6

MODEL_CANDIDATES = (
    "logistic_regression",
    "hist_gradient_boosting",
    "lightgbm",  # included when lightgbm is installed (requirements-dev.txt)
)
BEST_MODEL_FILENAME = "best_model.joblib"
MODEL_METADATA_FILENAME = "model_metadata.json"
BEST_PARAMS_FILENAME = "best_params.json"  # exported by the tuning notebook
CHAMPION_CONFIG_FILENAME = "champion_config.json"  # feature-reduced champion from 02b

TRAIN_TEST_SPLIT_RATIO = 0.2
RANDOM_STATE = 42

# Split strategy aligned with notebooks/02b_model_tuning_random_dev_oot.ipynb:
# random stratified train/val/test on development cohorts + temporal OOT holdout.
OOT_COHORTS = 3
DEV_TEST_FRAC = 0.15
DEV_VAL_FRAC = 0.15

# Legacy temporal split (used by notebooks/02_model_tuning.ipynb).
N_VAL_COHORTS = 2
N_TEST_COHORTS = 2

# Primary model-selection metric. ROC-AUC (Gini = 2*AUC-1) is the industry
# standard discrimination metric for application credit scorecards.
SELECTION_METRIC = "roc_auc"

# Business cost ratio used to pick the decision threshold.
# Approving a defaulter (false negative) loses ~LGD * exposure; rejecting a good
# applicant (false positive) only loses the interest margin. We weight a missed
# default 5x a wrongful rejection (tunable; revisit with Finance).
COST_FALSE_NEGATIVE = 5.0
COST_FALSE_POSITIVE = 1.0

# Probability calibration so predicted scores can be read as true PDs
# (required for expected-loss, pricing and provisioning).
CALIBRATION_METHOD = "isotonic"

# Governance / monitoring thresholds (PSI convention).
PSI_WARN_THRESHOLD = 0.10
PSI_ALERT_THRESHOLD = 0.25

# Number of champion-model inputs tracked for input drift (PSI / CSI).
TOP_MONITORED_FEATURES = 7
