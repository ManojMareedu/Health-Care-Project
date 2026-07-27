"""API smoke tests against the committed exported model."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from src.healthcare_mlops import config

VALID_CLAIM = {
    "PRNCPAL_DGNS_CD_inp": "I10",
    "PRNCPAL_DGNS_CD_out": "E119",
    "CLM_E_POA_IND_SW1": "Y",
    "Number_of_Claims_inp": 3,
    "Number_of_Claims_out": 7,
    "Median_Income": 60510.0,
}

needs_model = pytest.mark.skipif(
    not config.CLASSIFIER_DIR.exists(),
    reason="exported model not present",
)


@pytest.fixture(scope="module")
def client():
    """Provide a client whose lifespan has run, so models are loaded."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_is_reachable(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in {"healthy", "degraded"}


@needs_model
def test_health_reports_loaded_models(client):
    body = client.get("/health").json()
    assert body["status"] == "healthy"
    assert body["classifier_loaded"]
    assert body["regressor_loaded"]


@needs_model
def test_predict_tier_happy_path(client):
    response = client.post("/predict/tier", json=VALID_CLAIM)
    assert response.status_code == 200

    body = response.json()
    assert body["tier"] in config.TIER_LABELS
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["tier_description"]
    assert set(body["probabilities"]) == {str(t) for t in config.TIER_LABELS}
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)


@needs_model
def test_predict_tier_includes_explanations(client):
    contributions = client.post("/predict/tier", json=VALID_CLAIM).json()["top_contributions"]
    assert contributions
    assert len(contributions) <= config.SHAP_TOP_FEATURES
    for item in contributions:
        assert item["direction"] in {"increases", "decreases"}


@needs_model
def test_predict_charge_happy_path(client):
    body = client.post("/predict/charge", json=VALID_CLAIM).json()
    assert body["predicted_charge"] > 0


@needs_model
def test_unseen_diagnosis_code_does_not_error(client):
    """Inference must survive a code the encoder never saw during training."""
    payload = VALID_CLAIM | {"PRNCPAL_DGNS_CD_inp": "ZZZ999"}
    assert client.post("/predict/tier", json=payload).status_code == 200


@pytest.mark.parametrize(
    "payload",
    [
        VALID_CLAIM | {"CLM_E_POA_IND_SW1": "X"},
        VALID_CLAIM | {"Number_of_Claims_inp": -1},
        VALID_CLAIM | {"Median_Income": 0},
        {"PRNCPAL_DGNS_CD_inp": "I10"},
    ],
    ids=["bad_poa", "negative_claims", "zero_income", "missing_fields"],
)
def test_malformed_input_returns_422(client, payload):
    assert client.post("/predict/tier", json=payload).status_code == 422
