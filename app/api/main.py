"""FastAPI inference service for claim cost tier and charge prediction.

Two endpoints back the two modeling tasks, plus a health check for
orchestrators and CI smoke tests. Both models and the SHAP explainer are loaded
once at startup rather than per request -- loading a 6MB pipeline on every call
would dominate latency.

Run locally with::

    uvicorn app.api.main:app --reload
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Literal

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.healthcare_mlops import config, explain, models

logger = logging.getLogger(__name__)

# Populated at startup. Kept in a plain dict so the tests can inspect what
# actually loaded rather than mocking module globals.
STATE: dict[str, object] = {}


class ClaimFeatures(BaseModel):
    """One claim's features, matching the columns the pipeline was trained on."""

    PRNCPAL_DGNS_CD_inp: str = Field(
        ..., description="Inpatient principal ICD-10 diagnosis code", examples=["I10"]
    )
    PRNCPAL_DGNS_CD_out: str = Field(
        ..., description="Outpatient principal ICD-10 diagnosis code", examples=["E119"]
    )
    CLM_E_POA_IND_SW1: Literal["Y", "U"] = Field(
        ..., description="Present-on-admission indicator", examples=["Y"]
    )
    Number_of_Claims_inp: int = Field(..., ge=0, description="Inpatient claim count", examples=[3])
    Number_of_Claims_out: int = Field(..., ge=0, description="Outpatient claim count", examples=[7])
    Median_Income: float = Field(
        ..., gt=0, description="State median household income", examples=[60510.0]
    )

    def to_frame(self) -> pd.DataFrame:
        """Convert to the single-row frame the pipeline expects.

        Returns:
            A one-row frame with columns in ``config.FEATURE_COLUMNS`` order.
        """
        return pd.DataFrame([self.model_dump()])[config.FEATURE_COLUMNS]


class FeatureContribution(BaseModel):
    """A single feature's SHAP contribution to a prediction."""

    feature: str
    contribution: float
    direction: str


class TierPrediction(BaseModel):
    """Cost-tier classification response."""

    tier: int = Field(..., description="Predicted cost tier, 1 (lowest) to 5 (highest)")
    tier_description: str
    confidence: float = Field(..., description="Probability assigned to the predicted tier")
    probabilities: dict[str, float]
    top_contributions: list[FeatureContribution]
    model_name: str


class ChargePrediction(BaseModel):
    """Total-charge regression response."""

    predicted_charge: float = Field(..., description="Predicted total charge in dollars")
    predicted_log_charge: float
    model_name: str


class HealthResponse(BaseModel):
    """Service health and loaded-model status."""

    status: str
    classifier_loaded: bool
    regressor_loaded: bool
    explainer_ready: bool
    classifier_name: str | None = None
    regressor_name: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at startup and release them on shutdown.

    Args:
        app: The FastAPI application (unused; required by the lifespan protocol).

    Yields:
        Control back to the running application.
    """
    try:
        STATE["classifier"] = mlflow.sklearn.load_model(str(config.CLASSIFIER_DIR))
        STATE["regressor"] = mlflow.sklearn.load_model(str(config.REGRESSOR_DIR))
        STATE["metadata"] = json.loads(config.METADATA_FILE.read_text(encoding="utf-8"))
        STATE["explainer"] = explain.PredictionExplainer(
            STATE["classifier"], explain.load_background()
        )
        logger.info("Loaded models from %s", config.EXPORTED_MODEL_DIR)
    except Exception:
        # A failed load must not take the process down -- /health has to stay
        # reachable so an orchestrator can report *why* the service is unhealthy.
        logger.exception("Model loading failed; service will report unhealthy")
    yield
    STATE.clear()


app = FastAPI(
    title="Healthcare Claims Cost Intelligence API",
    description=(
        "Predicts the cost tier and total charge of a Medicare-style claim, with "
        "SHAP feature attributions for every tier prediction."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _require(key: str):
    """Fetch a loaded artifact or fail with a clear 503.

    Args:
        key: Key in the module-level ``STATE`` dict.

    Returns:
        The loaded artifact.

    Raises:
        HTTPException: 503 when the artifact never loaded.
    """
    value = STATE.get(key)
    if value is None:
        raise HTTPException(status_code=503, detail=f"{key} is not loaded")
    return value


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report service health and which models are loaded.

    Returns:
        Health status, degraded when any artifact is missing.
    """
    metadata = STATE.get("metadata") or {}
    loaded = all(STATE.get(k) is not None for k in ("classifier", "regressor"))
    return HealthResponse(
        status="healthy" if loaded else "degraded",
        classifier_loaded=STATE.get("classifier") is not None,
        regressor_loaded=STATE.get("regressor") is not None,
        explainer_ready=STATE.get("explainer") is not None,
        classifier_name=metadata.get("classifier", {}).get("name"),
        regressor_name=metadata.get("regressor", {}).get("name"),
    )


@app.get("/model-info")
def model_info() -> dict:
    """Expose the training metadata behind the deployed models.

    Returns:
        The metadata written at export time, including every candidate's metrics.
    """
    return _require("metadata")


@app.post("/predict/tier", response_model=TierPrediction)
def predict_tier(claim: ClaimFeatures) -> TierPrediction:
    """Classify a claim into a cost tier and explain the result.

    Args:
        claim: The claim's features.

    Returns:
        Predicted tier, per-tier probabilities, and top SHAP contributions.
    """
    classifier = _require("classifier")
    metadata = _require("metadata")
    frame = claim.to_frame()

    predicted_index = int(classifier.predict(frame)[0])
    probabilities = classifier.predict_proba(frame)[0]
    tier = int(models.to_tier_labels([predicted_index])[0])

    contributions: list[FeatureContribution] = []
    explainer = STATE.get("explainer")
    if explainer is not None:
        contributions = [
            FeatureContribution(**item)
            for item in explainer.top_contributions(frame, predicted_index)
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
        model_name=metadata["classifier"]["name"],
    )


@app.post("/predict/charge", response_model=ChargePrediction)
def predict_charge(claim: ClaimFeatures) -> ChargePrediction:
    """Predict a claim's total charge in dollars.

    The model is fit on ``log(TOTAL_CHARGE)``, so the response exponentiates back
    to dollars and reports the log-space value alongside it.

    Args:
        claim: The claim's features.

    Returns:
        The predicted charge in both dollar and log space.
    """
    regressor = _require("regressor")
    metadata = _require("metadata")

    log_charge = float(regressor.predict(claim.to_frame())[0])
    return ChargePrediction(
        predicted_charge=float(np.exp(log_charge)),
        predicted_log_charge=log_charge,
        model_name=metadata["regressor"]["name"],
    )
