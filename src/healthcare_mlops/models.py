"""Model definitions and pipeline assembly.

Every model is wrapped in a scikit-learn ``Pipeline`` whose first stage is the
preprocessor from ``feature_engineering``. That is deliberate: preprocessing and
estimator serialise as one object, so the serving path loads a single artifact
and there is no second copy of the encoding logic that could drift from what was
trained.

Label convention
----------------
``TC_class`` is 1-5, but XGBoost requires zero-based class labels. Rather than
give one model a different pipeline shape, every classifier trains on ``y - 1``
and callers map the predicted index back through ``config.TIER_LABELS``. The
offset is applied in exactly one place (``to_zero_based`` / ``to_tier_labels``)
so it cannot drift.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from . import config
from .feature_engineering import build_preprocessor


def to_zero_based(y) -> np.ndarray:
    """Convert 1-5 tier labels to the 0-4 form estimators train on.

    Args:
        y: Tier labels in the range 1-5.

    Returns:
        Zero-based integer labels.
    """
    return np.asarray(y, dtype=int) - 1


def to_tier_labels(y) -> np.ndarray:
    """Convert 0-4 predicted indices back to 1-5 tier labels.

    Args:
        y: Zero-based predictions from an estimator.

    Returns:
        Tier labels in the range 1-5.
    """
    return np.asarray(y, dtype=int) + 1


def classifier_candidates() -> dict[str, Pipeline]:
    """Build the four candidate tier classifiers.

    Class weighting matters here because tier 5 is 1.1% of rows. The tree-based
    models take ``class_weight="balanced"`` directly. KNN has no such parameter
    -- distance weighting is the closest available lever, and its weakness on
    this imbalance is itself a finding worth reporting rather than hiding.
    XGBoost is weighted per-sample at fit time (see ``sample_weights``).

    Returns:
        Mapping of model name to an unfitted pipeline.
    """
    estimators = {
        "knn": KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1),
        "decision_tree": DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=16,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            num_class=len(config.TIER_LABELS),
            tree_method="hist",
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        ),
    }
    return {
        name: Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])
        for name, estimator in estimators.items()
    }


def regressor_candidates() -> dict[str, Pipeline]:
    """Build the candidate log-charge regressors.

    Returns:
        Mapping of model name to an unfitted pipeline.
    """
    estimators = {
        "linear_regression": LinearRegression(),
        "ridge_regression": Ridge(alpha=1.0, random_state=config.RANDOM_STATE),
    }
    return {
        name: Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])
        for name, estimator in estimators.items()
    }


def sample_weights(y) -> np.ndarray:
    """Compute inverse-frequency sample weights for tier labels.

    Used for XGBoost, which has no ``class_weight`` parameter, so that all four
    classifiers face the same imbalance correction and the comparison stays fair.

    Args:
        y: Tier labels (either 1-5 or 0-4; only relative frequency matters).

    Returns:
        Per-sample weights normalised to mean 1.
    """
    labels = np.asarray(y)
    classes, counts = np.unique(labels, return_counts=True)
    weight_per_class = dict(zip(classes, len(labels) / (len(classes) * counts), strict=True))
    weights = np.array([weight_per_class[label] for label in labels], dtype=float)
    return weights / weights.mean()
