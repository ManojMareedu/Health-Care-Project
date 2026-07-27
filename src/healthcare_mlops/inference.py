"""In-process inference against the committed exported models.

This is the single inference path in the system. The FastAPI service wraps it in
HTTP; the Streamlit dashboard calls it directly. Neither reimplements scoring, so
the hosted dashboard cannot drift from what the API would have returned.

Loading directly also means the dashboard has no runtime dependency on any other
service, which is what lets it run on a free host with nothing else deployed.

Why joblib rather than mlflow.sklearn.load_model
------------------------------------------------
Training exports MLflow model directories, and the artefact inside one is a
plain pickled scikit-learn ``Pipeline``. Reading it back needs scikit-learn and
nothing else, whereas importing ``mlflow`` pulls in the tracking client and with
it pyarrow, sqlalchemy, and alembic -- none of which a read-only inference path
touches.

That weight is not merely untidy. On a hosted runner with no prebuilt pyarrow
wheel for its Python version, the install falls back to compiling pyarrow from
source and fails for want of cmake. Loading the pickle directly keeps the serving
dependency set to what serving actually uses, and is verified against the mlflow
loader in the test suite so the two cannot silently diverge.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from . import config, explain, models
from .schemas import ChargePrediction, FeatureContribution, TierPrediction

logger = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    """Everything needed to score a claim and explain the result.

    Attributes:
        classifier: Fitted tier-classification pipeline.
        regressor: Fitted log-charge regression pipeline.
        metadata: Training metadata written at export time.
        explainer: SHAP explainer, or None if one could not be built.
    """

    classifier: Pipeline
    regressor: Pipeline
    metadata: dict
    explainer: explain.PredictionExplainer | None

    @property
    def classifier_name(self) -> str:
        """Name of the production classifier.

        Returns:
            The model name recorded at export time.
        """
        return self.metadata["classifier"]["name"]

    @property
    def regressor_name(self) -> str:
        """Name of the production regressor.

        Returns:
            The model name recorded at export time.
        """
        return self.metadata["regressor"]["name"]


def load_bundle() -> ModelBundle:
    """Load both models, their metadata, and a matching SHAP explainer.

    Returns:
        A populated ``ModelBundle``.

    Raises:
        FileNotFoundError: If the exported model directory is missing.
    """
    if not config.CLASSIFIER_PICKLE.exists():
        raise FileNotFoundError(
            f"No exported model at {config.CLASSIFIER_PICKLE}. "
            "Run: python -m src.healthcare_mlops.train_pipeline"
        )

    classifier = joblib.load(config.CLASSIFIER_PICKLE)
    regressor = joblib.load(config.REGRESSOR_PICKLE)
    metadata = json.loads(config.METADATA_FILE.read_text(encoding="utf-8"))

    explainer: explain.PredictionExplainer | None
    try:
        explainer = explain.PredictionExplainer(classifier, explain.load_background())
    except (ValueError, FileNotFoundError):
        # Predictions stay useful without attributions, so a missing background
        # sample degrades the explanation rather than the whole service.
        logger.warning("SHAP explainer unavailable; predictions will omit attributions")
        explainer = None

    return ModelBundle(
        classifier=classifier, regressor=regressor, metadata=metadata, explainer=explainer
    )


def predict_tier(bundle: ModelBundle, frame: pd.DataFrame) -> TierPrediction:
    """Classify a claim into a cost tier and explain the result.

    Args:
        bundle: Loaded models and explainer.
        frame: One-row feature frame.

    Returns:
        The tier, per-tier probabilities, and top SHAP contributions.
    """
    predicted_index = int(bundle.classifier.predict(frame)[0])
    probabilities = bundle.classifier.predict_proba(frame)[0]
    tier = int(models.to_tier_labels([predicted_index])[0])

    contributions: list[FeatureContribution] = []
    if bundle.explainer is not None:
        contributions = [
            FeatureContribution(**item)
            for item in bundle.explainer.top_contributions(frame, predicted_index)
        ]

    return TierPrediction(
        tier=tier,
        tier_description=config.TIER_DESCRIPTIONS[tier],
        confidence=float(probabilities[predicted_index]),
        probabilities={
            str(int(label)): float(probability)
            for label, probability in zip(config.TIER_LABELS, probabilities, strict=True)
        },
        top_contributions=contributions,
        model_name=bundle.classifier_name,
    )


def predict_charge(bundle: ModelBundle, frame: pd.DataFrame) -> ChargePrediction:
    """Predict a claim's total charge in dollars.

    The model is fit on ``log(TOTAL_CHARGE)``, so this exponentiates back to
    dollars and reports the log-space value alongside it.

    Args:
        bundle: Loaded models and explainer.
        frame: One-row feature frame.

    Returns:
        The predicted charge in both dollar and log space.
    """
    log_charge = float(bundle.regressor.predict(frame)[0])
    return ChargePrediction(
        predicted_charge=float(np.exp(log_charge)),
        predicted_log_charge=log_charge,
        model_name=bundle.regressor_name,
    )
