"""Metrics and the documented model-selection rule.

Selection rule
--------------
Accuracy alone is the wrong criterion on this target. Tier 3 is half the rows,
so a model that ignores tiers 1 and 5 entirely can still post a respectable
accuracy while being useless for the actual business question -- which claims
are unusually expensive. Selection therefore uses:

    score = 0.6 * macro_f1 + 0.4 * accuracy

Macro-F1 weights every tier equally regardless of size, so a model must handle
the rare catastrophic tier to win. The 60/40 split keeps accuracy in the picture
without letting it dominate. The rule is stated here, logged to MLflow with each
run, and repeated in the README.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

MACRO_F1_WEIGHT = 0.6
ACCURACY_WEIGHT = 0.4


def classification_metrics(y_true, y_pred, y_proba=None) -> dict[str, float]:
    """Compute tier-classification metrics.

    Args:
        y_true: True tier labels.
        y_pred: Predicted tier labels.
        y_proba: Optional class probability matrix for ROC-AUC.

    Returns:
        Mapping of metric name to value. ``roc_auc_ovr`` is omitted when
        probabilities are unavailable or undefined for the label set.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }
    if y_proba is not None:
        # ValueError is raised when a class is absent from y_true. Omitting the
        # metric is more honest than reporting a silently degraded one.
        with contextlib.suppress(ValueError):
            metrics["roc_auc_ovr"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            )
    metrics["selection_score"] = selection_score(metrics)
    return metrics


def selection_score(metrics: dict[str, float]) -> float:
    """Apply the weighted selection rule to a metrics mapping.

    Args:
        metrics: Mapping containing ``macro_f1`` and ``accuracy``.

    Returns:
        The weighted score used to pick the production classifier.
    """
    return float(MACRO_F1_WEIGHT * metrics["macro_f1"] + ACCURACY_WEIGHT * metrics["accuracy"])


def regression_metrics(y_true_log, y_pred_log) -> dict[str, float]:
    """Compute log-charge regression metrics, in log and dollar space.

    RMSE is reported in log space (where the model was fit) and the back-
    transformed median absolute dollar error is reported alongside it, because a
    log-space RMSE of 1.7 is hard to reason about when the question is "how many
    dollars off is this".

    Args:
        y_true_log: True ``log(TOTAL_CHARGE)`` values.
        y_pred_log: Predicted values in log space.

    Returns:
        Mapping of metric name to value.
    """
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)
    dollars_true = np.exp(y_true_log)
    dollars_pred = np.exp(y_pred_log)
    return {
        "rmse_log": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "mae_log": float(mean_absolute_error(y_true_log, y_pred_log)),
        "r2_log": float(r2_score(y_true_log, y_pred_log)),
        "median_abs_dollar_error": float(np.median(np.abs(dollars_true - dollars_pred))),
    }


def confusion_frame(y_true, y_pred, labels) -> pd.DataFrame:
    """Build a labelled confusion matrix.

    Args:
        y_true: True tier labels.
        y_pred: Predicted tier labels.
        labels: Ordered tier labels.

    Returns:
        A frame indexed by true tier with predicted tiers as columns.
    """
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(
        matrix,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )


def select_best(results: dict[str, dict[str, float]]) -> str:
    """Pick the winning model by the documented rule.

    Args:
        results: Mapping of model name to its metrics.

    Returns:
        Name of the highest-scoring model.
    """
    return max(results, key=lambda name: results[name]["selection_score"])
