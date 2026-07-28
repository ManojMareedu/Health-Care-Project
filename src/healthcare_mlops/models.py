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
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRegressor

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

    Linear models alone cannot answer whether the weak fit is a property of the
    features or of the model class, so two tree ensembles are included. If a
    gradient-boosted regressor cannot beat ridge on the same leakage-free split,
    the relationship really is close to linear in log space and the ceiling is
    the feature set, not the estimator.

    Returns:
        Mapping of model name to an unfitted pipeline.
    """
    estimators = {
        "linear_regression": LinearRegression(),
        "ridge_regression": Ridge(alpha=1.0, random_state=config.RANDOM_STATE),
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=300,
            max_depth=16,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        ),
        "xgboost_regressor": XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        ),
    }
    return {
        name: Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])
        for name, estimator in estimators.items()
    }


# Search spaces for the two strongest classifiers. Deliberately small: this is a
# portfolio project on a fixed dataset, and an exhaustive grid would burn compute
# to move a metric that is limited by six features, not by hyperparameters.
CLASSIFIER_SEARCH_SPACES: dict[str, dict[str, list]] = {
    "random_forest": {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [12, 16, 24, None],
        "model__min_samples_leaf": [1, 2, 5, 10],
        "model__max_features": ["sqrt", "log2", 0.5],
    },
    "xgboost": {
        "model__n_estimators": [200, 400, 600],
        "model__max_depth": [4, 6, 8, 10],
        "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
        "model__subsample": [0.7, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.9, 1.0],
        "model__min_child_weight": [1, 3, 5],
    },
}


def tune_classifier(
    name: str,
    x_train: pd.DataFrame,
    y_train,
    groups,
    n_iter: int = 10,
    cv_splits: int = 3,
) -> tuple[Pipeline, dict, float]:
    """Randomised hyperparameter search under grouped cross-validation.

    The folds are grouped by ``BENE_ID``. Using a plain KFold here would put the
    same beneficiary on both sides of every internal split and select
    hyperparameters against a leaky score -- reintroducing, inside the search,
    exactly the defect the outer split was built to remove.

    Scoring is macro-F1 to match the production selection rule, so the search
    optimises for the rare catastrophic tier rather than overall accuracy.

    Args:
        name: Key into ``CLASSIFIER_SEARCH_SPACES``.
        x_train: Training features.
        y_train: Zero-based tier labels.
        groups: Beneficiary ids aligned with ``x_train``.
        n_iter: Parameter settings sampled.
        cv_splits: Number of grouped CV folds.

    Returns:
        ``(best_pipeline, best_params, best_cv_macro_f1)``.
    """
    search = RandomizedSearchCV(
        estimator=classifier_candidates()[name],
        param_distributions=CLASSIFIER_SEARCH_SPACES[name],
        n_iter=n_iter,
        scoring="f1_macro",
        cv=GroupKFold(n_splits=cv_splits),
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    # XGBoost's sample weights are deliberately omitted during the search:
    # sklearn will not resample a fit-time weight vector per fold without
    # metadata routing enabled. Every sampled configuration is therefore
    # penalised identically, so the ranking still holds, and the winning
    # configuration is refitted with weights before export.
    search.fit(x_train, y_train, groups=groups)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


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
