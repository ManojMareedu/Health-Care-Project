"""SHAP explanations for individual predictions.

A tier prediction on its own is not actionable -- a reviewer needs to know *why*
a claim was flagged. This module attaches the top contributing features to each
prediction, in both the API response and the dashboard.

Explainer selection
-------------------
``TreeExplainer`` is fast and exact, but only works on tree ensembles. The
production classifier is chosen at training time by a metric rule, so it is not
guaranteed to be a tree -- KNN is a live candidate. Rather than assume, the
explainer is chosen by inspecting the fitted estimator:

* tree-based (decision tree, random forest, XGBoost) -> ``TreeExplainer``
* anything else (KNN) -> the model-agnostic ``KernelExplainer`` path

The model-agnostic path is dramatically slower, which is why it runs against the
small persisted background sample rather than the full training set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from . import config

TREE_ESTIMATORS = (DecisionTreeClassifier, RandomForestClassifier, XGBClassifier)


def is_tree_model(pipeline_obj: Pipeline) -> bool:
    """Report whether the pipeline's estimator supports the fast tree explainer.

    Args:
        pipeline_obj: Fitted pipeline whose final step is named ``model``.

    Returns:
        True when ``TreeExplainer`` can be used.
    """
    return isinstance(pipeline_obj.named_steps["model"], TREE_ESTIMATORS)


def _feature_names(pipeline_obj: Pipeline) -> list[str]:
    """Return readable output feature names from the fitted preprocessor.

    Args:
        pipeline_obj: Fitted pipeline.

    Returns:
        Feature names with the ColumnTransformer prefix stripped.
    """
    raw = pipeline_obj.named_steps["preprocessor"].get_feature_names_out()
    return [name.split("__", 1)[-1] for name in raw]


class PredictionExplainer:
    """Wraps a fitted pipeline with a SHAP explainer chosen to match it.

    Attributes:
        pipeline: The fitted classification pipeline.
        uses_tree_explainer: Whether the fast exact path is in use.
    """

    def __init__(self, pipeline_obj: Pipeline, background: pd.DataFrame | None = None):
        """Build an explainer appropriate to the pipeline's estimator.

        Args:
            pipeline_obj: Fitted pipeline whose final step is named ``model``.
            background: Raw-feature background sample. Required for the
                model-agnostic path; ignored by the tree path.

        Raises:
            ValueError: If a non-tree model is supplied without background data,
                since the model-agnostic explainer cannot be built without it.
        """
        self.pipeline = pipeline_obj
        self.uses_tree_explainer = is_tree_model(pipeline_obj)
        self._names = _feature_names(pipeline_obj)
        preprocessor = pipeline_obj.named_steps["preprocessor"]
        estimator = pipeline_obj.named_steps["model"]

        if self.uses_tree_explainer:
            self._explainer = shap.TreeExplainer(estimator)
            return

        if background is None or background.empty:
            raise ValueError(
                "A non-tree model requires a background sample for the "
                "model-agnostic SHAP explainer"
            )
        transformed = preprocessor.transform(background)
        self._explainer = shap.KernelExplainer(estimator.predict_proba, transformed)

    def shap_values(self, frame: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for raw input rows.

        Args:
            frame: Raw feature rows, matching ``config.FEATURE_COLUMNS``.

        Returns:
            Array of shape ``(n_rows, n_features, n_classes)``.
        """
        transformed = self.pipeline.named_steps["preprocessor"].transform(frame)
        values = self._explainer.shap_values(transformed)
        # shap returns either a list-per-class or a stacked 3-D array depending
        # on explainer and model version; normalise to (rows, features, classes).
        if isinstance(values, list):
            return np.stack(values, axis=-1)
        values = np.asarray(values)
        if values.ndim == 2:
            return values[:, :, np.newaxis]
        return values

    def top_contributions(
        self, frame: pd.DataFrame, predicted_index: int, top_n: int | None = None
    ) -> list[dict[str, float | str]]:
        """Return the features that pushed a single prediction hardest.

        Args:
            frame: A single raw feature row.
            predicted_index: Zero-based index of the predicted class.
            top_n: How many features to return. Defaults to
                ``config.SHAP_TOP_FEATURES``.

        Returns:
            Records ordered by absolute contribution, each carrying the feature
            name, its signed SHAP value, and the direction of its effect.
        """
        top_n = top_n or config.SHAP_TOP_FEATURES
        values = self.shap_values(frame)[0]
        column = min(predicted_index, values.shape[1] - 1)
        contributions = values[:, column]

        order = np.argsort(np.abs(contributions))[::-1][:top_n]
        return [
            {
                "feature": self._names[i],
                "contribution": round(float(contributions[i]), 6),
                "direction": "increases" if contributions[i] > 0 else "decreases",
            }
            for i in order
        ]


def load_background() -> pd.DataFrame | None:
    """Read the background sample shipped with the exported model.

    Returns:
        The background frame, or None when it was not exported.
    """
    if not config.SHAP_BACKGROUND_FILE.exists():
        return None
    return pd.read_csv(config.SHAP_BACKGROUND_FILE)
