"""FastAPI inference service for claim cost tier and charge prediction.

This wraps ``src.healthcare_mlops.inference`` in HTTP. The scoring logic itself
lives there and is shared with the dashboard, so the two surfaces cannot return
different answers for the same claim.

The models load once at startup rather than per request -- loading a 6MB pipeline
on every call would dominate latency.

Run locally with::

    uvicorn app.api.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.healthcare_mlops import inference
from src.healthcare_mlops.schemas import (
    ChargePrediction,
    ClaimFeatures,
    HealthResponse,
    TierPrediction,
)

logger = logging.getLogger(__name__)

# Populated at startup. Kept in a plain dict so tests can inspect what actually
# loaded rather than mocking module globals.
STATE: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at startup and release them on shutdown.

    Args:
        app: The FastAPI application (unused; required by the lifespan protocol).

    Yields:
        Control back to the running application.
    """
    try:
        STATE["bundle"] = inference.load_bundle()
        logger.info("Loaded models from %s", inference.config.EXPORTED_MODEL_DIR)
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


def _bundle() -> inference.ModelBundle:
    """Fetch the loaded model bundle or fail with a clear 503.

    Returns:
        The loaded bundle.

    Raises:
        HTTPException: 503 when the models never loaded.
    """
    bundle = STATE.get("bundle")
    if bundle is None:
        raise HTTPException(status_code=503, detail="models are not loaded")
    return bundle


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report service health and which models are loaded.

    Returns:
        Health status, degraded when the models are missing.
    """
    bundle = STATE.get("bundle")
    if bundle is None:
        return HealthResponse(
            status="degraded",
            classifier_loaded=False,
            regressor_loaded=False,
            explainer_ready=False,
        )
    return HealthResponse(
        status="healthy",
        classifier_loaded=bundle.classifier is not None,
        regressor_loaded=bundle.regressor is not None,
        explainer_ready=bundle.explainer is not None,
        classifier_name=bundle.classifier_name,
        regressor_name=bundle.regressor_name,
    )


@app.get("/model-info")
def model_info() -> dict:
    """Expose the training metadata behind the deployed models.

    Returns:
        The metadata written at export time, including every candidate's metrics.
    """
    return _bundle().metadata


@app.post("/predict/tier", response_model=TierPrediction)
def predict_tier(claim: ClaimFeatures) -> TierPrediction:
    """Classify a claim into a cost tier and explain the result.

    Args:
        claim: The claim's features.

    Returns:
        Predicted tier, per-tier probabilities, and top SHAP contributions.
    """
    return inference.predict_tier(_bundle(), claim.to_frame())


@app.post("/predict/charge", response_model=ChargePrediction)
def predict_charge(claim: ClaimFeatures) -> ChargePrediction:
    """Predict a claim's total charge in dollars.

    Args:
        claim: The claim's features.

    Returns:
        The predicted charge in both dollar and log space.
    """
    return inference.predict_charge(_bundle(), claim.to_frame())
