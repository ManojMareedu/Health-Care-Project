"""Tests for the shared inference module and the standalone dashboard.

The dashboard is the publicly hosted surface and it loads the model in-process,
so these check the thing that actually ships: scoring works with no API running,
and both surfaces agree because they share one scoring module.
"""

from __future__ import annotations

import pytest

from src.healthcare_mlops import config, inference
from src.healthcare_mlops.schemas import ClaimFeatures

VALID_CLAIM = {
    "PRNCPAL_DGNS_CD_inp": "I10",
    "PRNCPAL_DGNS_CD_out": "E119",
    "CLM_E_POA_IND_SW1": "Y",
    "Number_of_Claims_inp": 3,
    "Number_of_Claims_out": 7,
    "Median_Income": 60510.0,
}

needs_model = pytest.mark.skipif(
    not config.CLASSIFIER_DIR.exists(), reason="exported model not present"
)


@pytest.fixture(scope="module")
def bundle():
    """Load the exported models once for the module."""
    return inference.load_bundle()


@needs_model
def test_bundle_loads_without_any_service_running(bundle):
    """The hosted dashboard has no backend, so loading must be self-contained."""
    assert bundle.classifier is not None
    assert bundle.regressor is not None
    assert bundle.classifier_name
    assert bundle.regressor_name


@needs_model
def test_tier_prediction_is_well_formed(bundle):
    result = inference.predict_tier(bundle, ClaimFeatures(**VALID_CLAIM).to_frame())
    assert result.tier in config.TIER_LABELS
    assert 0.0 <= result.confidence <= 1.0
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-4)
    assert result.top_contributions


@needs_model
def test_charge_prediction_is_positive(bundle):
    result = inference.predict_charge(bundle, ClaimFeatures(**VALID_CLAIM).to_frame())
    assert result.predicted_charge > 0


@needs_model
def test_api_and_dashboard_paths_agree(bundle):
    """Both surfaces call this module, so their answers must be identical."""
    from fastapi.testclient import TestClient

    from app.api.main import app

    direct = inference.predict_tier(bundle, ClaimFeatures(**VALID_CLAIM).to_frame())
    with TestClient(app) as client:
        over_http = client.post("/predict/tier", json=VALID_CLAIM).json()

    assert over_http["tier"] == direct.tier
    assert over_http["confidence"] == pytest.approx(direct.confidence)
    assert over_http["model_name"] == direct.model_name


def test_shared_schema_rejects_bad_input():
    """The dashboard reuses the API's schema, so both reject the same values."""
    from pydantic import ValidationError

    for bad in (
        VALID_CLAIM | {"CLM_E_POA_IND_SW1": "X"},
        VALID_CLAIM | {"Number_of_Claims_inp": -1},
        VALID_CLAIM | {"Median_Income": 0},
    ):
        with pytest.raises(ValidationError):
            ClaimFeatures(**bad)


@needs_model
def test_dashboard_runs_standalone():
    """Boot the real dashboard script with no API reachable."""
    from streamlit.testing.v1 import AppTest

    app_test = AppTest.from_file("app/dashboard/app.py", default_timeout=120).run()
    assert not app_test.exception
    assert not app_test.error
    assert len(app_test.tabs) == 3


@needs_model
def test_dashboard_scores_a_claim():
    """Submitting the form must produce a tier and a charge, not just render."""
    from streamlit.testing.v1 import AppTest

    app_test = AppTest.from_file("app/dashboard/app.py", default_timeout=120).run()
    app_test.button[0].click().run()

    assert not app_test.exception
    assert not app_test.error
    labels = {metric.label for metric in app_test.metric}
    assert {"Confidence", "Predicted charge"} <= labels
