"""Central paths and modeling constants.

Everything that another module might otherwise hardcode lives here, so changing
where data sits or how cost tiers are cut is a one-file edit rather than a grep.
"""

from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
CLAIMS_FILE = DATA_RAW / "Patient_Claim_Data.xlsx"
INCOME_FILE = DATA_RAW / "Median_Income.xlsx"

EXPORTED_MODEL_DIR = PROJECT_ROOT / "exported_model"
CLASSIFIER_DIR = EXPORTED_MODEL_DIR / "tier_classifier"
REGRESSOR_DIR = EXPORTED_MODEL_DIR / "charge_regressor"
METADATA_FILE = EXPORTED_MODEL_DIR / "model_metadata.json"

# The pickled sklearn pipelines inside the MLflow model directories. Serving
# loads these directly rather than going through mlflow.sklearn.load_model: the
# pickle is a plain sklearn Pipeline, so reading it needs only scikit-learn,
# while importing mlflow drags in pyarrow, sqlalchemy, and alembic that a
# read-only inference path never uses. Training still exports MLflow format.
CLASSIFIER_PICKLE = CLASSIFIER_DIR / "model.pkl"
REGRESSOR_PICKLE = REGRESSOR_DIR / "model.pkl"

# CSV rather than parquet so reading the background needs no parquet engine.
SHAP_BACKGROUND_FILE = EXPORTED_MODEL_DIR / "shap_background.csv"

# Rows of training data shipped alongside the model as a SHAP background
# distribution. Small enough to commit, large enough to be representative -- and
# the model-agnostic explainer path is O(background size) per explanation, so
# this doubles as the latency budget for a KNN production model.
SHAP_BACKGROUND_ROWS = 200
SHAP_TOP_FEATURES = 5

MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
MLFLOW_EXPERIMENT = "healthcare-claims-cost"

# The beneficiary key. Rows repeat per beneficiary (one row per outpatient
# diagnosis), so this is the grouping key for the train/test split -- never a
# unique row id. See feature_engineering.split_by_beneficiary.
GROUP_KEY = "BENE_ID"

TARGET_CHARGE = "TOTAL_CHARGE"
TARGET_LOG_CHARGE = "LOG_TOTAL_CHARGE"
TARGET_TIER = "TC_class"

# Cost tiers in dollars. Class 5 (>$1M) is genuinely rare (521 of 46,059 rows,
# 1.1%), which is why model selection weights macro-F1 rather than accuracy.
TIER_BINS = [-np.inf, 1_000, 10_000, 100_000, 1_000_000, np.inf]
TIER_LABELS = [1, 2, 3, 4, 5]
TIER_DESCRIPTIONS = {
    1: "Under $1K - routine, auto-approval candidate",
    2: "$1K-$10K - standard processing",
    3: "$10K-$100K - elevated cost, sample for review",
    4: "$100K-$1M - high cost, route to utilization review",
    5: "Over $1M - catastrophic claim, specialist handling",
}

# High-cardinality ICD-10 diagnosis codes (174 inpatient / 211 outpatient
# distinct values). Frequency encoding keeps these as one numeric column each
# instead of exploding into hundreds of dummies.
FREQUENCY_ENCODED_COLUMNS = ["PRNCPAL_DGNS_CD_inp", "PRNCPAL_DGNS_CD_out"]
CATEGORICAL_COLUMNS = ["CLM_E_POA_IND_SW1"]
NUMERIC_COLUMNS = [
    "Number_of_Claims_inp",
    "Number_of_Claims_out",
    "Median_Income",
]

FEATURE_COLUMNS = FREQUENCY_ENCODED_COLUMNS + CATEGORICAL_COLUMNS + NUMERIC_COLUMNS

RANDOM_STATE = 42
TEST_SIZE = 0.3
