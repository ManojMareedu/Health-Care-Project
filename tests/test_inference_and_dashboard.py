"""Tests for the shared inference module and the standalone dashboard.

The dashboard is the publicly hosted surface and it loads the model in-process,
so these check the thing that actually ships: scoring works with no API running,
and both surfaces agree because they share one scoring module.
"""

from __future__ import annotations

import json

import joblib
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


def test_serving_path_does_not_import_mlflow():
    """Guard the dependency fix that unblocked the hosted deploy.

    Importing mlflow pulls in pyarrow, sqlalchemy, and alembic. On a host whose
    Python has no prebuilt pyarrow wheel, the install tries to compile pyarrow
    and fails for want of cmake -- which is exactly how the first Streamlit Cloud
    deploy died. Nothing on the serving path may reach for mlflow again.
    """
    import ast
    from pathlib import Path

    serving_modules = [
        Path("src/healthcare_mlops/inference.py"),
        Path("src/healthcare_mlops/explain.py"),
        Path("src/healthcare_mlops/schemas.py"),
        Path("src/healthcare_mlops/config.py"),
        Path("app/api/main.py"),
        Path("app/dashboard/app.py"),
    ]
    for module in serving_modules:
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(n.split(".")[0] == "mlflow" for n in names), (
                f"{module} imports mlflow; the serving path must stay mlflow-free"
            )


@needs_model
def test_joblib_loader_matches_mlflow_loader():
    """The exported artefact must score identically however it is loaded.

    Serving reads the pickle directly while training writes MLflow format. If
    those two ever disagree, the deployed model is not the evaluated model.
    """
    mlflow_sklearn = pytest.importorskip("mlflow.sklearn")

    frame = ClaimFeatures(**VALID_CLAIM).to_frame()
    via_joblib = joblib.load(config.CLASSIFIER_PICKLE)
    via_mlflow = mlflow_sklearn.load_model(str(config.CLASSIFIER_DIR))

    assert via_joblib.predict(frame) == via_mlflow.predict(frame)
    assert (via_joblib.predict_proba(frame) == via_mlflow.predict_proba(frame)).all()

    regressor_joblib = joblib.load(config.REGRESSOR_PICKLE)
    regressor_mlflow = mlflow_sklearn.load_model(str(config.REGRESSOR_DIR))
    assert regressor_joblib.predict(frame) == regressor_mlflow.predict(frame)


@needs_model
def test_shap_background_loads_without_a_parquet_engine():
    """The background is CSV so reading it needs no pyarrow/fastparquet."""
    from src.healthcare_mlops import explain

    background = explain.load_background()
    assert background is not None
    assert list(background.columns) == config.FEATURE_COLUMNS
    assert config.SHAP_BACKGROUND_FILE.suffix == ".csv"


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

    app_test = AppTest.from_file("app/dashboard/app.py", default_timeout=180).run()
    app_test.button[0].click().run()

    assert not app_test.exception
    assert not app_test.error
    labels = {metric.label for metric in app_test.metric}
    assert {"Estimated total charge", "Tier confidence"} <= labels
    # The coloured badge is the primary output; a render that lost it would still
    # pass a metrics-only assertion.
    assert any("PREDICTED COST TIER" in str(block.value) for block in app_test.markdown)


@needs_model
def test_dashboard_shows_all_candidate_models():
    """The comparison tab must list every trained candidate, not a stale subset."""
    from streamlit.testing.v1 import AppTest

    app_test = AppTest.from_file("app/dashboard/app.py", default_timeout=180).run()
    rendered = {name for table in app_test.dataframe for name in table.value.index}

    metadata = json.loads(config.METADATA_FILE.read_text(encoding="utf-8"))
    expected = set(metadata["classifier"]["all_candidates"]) | set(
        metadata["regressor"]["all_candidates"]
    )
    assert expected <= rendered
