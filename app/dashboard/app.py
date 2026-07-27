"""Streamlit dashboard for the claims cost intelligence models.

Three views: how the candidate models compared, a live scoring form with SHAP
attributions, and the methodology corrections applied to the legacy pipeline.

Predictions are scored **in-process** against the committed ``exported_model/``
through ``src.healthcare_mlops.inference`` -- the same module the FastAPI service
wraps in HTTP. Sharing that module rather than calling the API over the network
means the two surfaces cannot disagree, and it leaves this app with no runtime
dependency on any other running service, so it deploys to a free static-ish host
on its own.

Run locally with::

    streamlit run app/dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Streamlit Cloud runs this file directly, so the repository root is not
# necessarily on sys.path the way it is under `python -m`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pydantic import ValidationError  # noqa: E402

from src.healthcare_mlops import config, inference  # noqa: E402
from src.healthcare_mlops.schemas import ClaimFeatures  # noqa: E402

TIER_COLORS = {1: "#2E7D32", 2: "#7CB342", 3: "#F9A825", 4: "#EF6C00", 5: "#C62828"}

st.set_page_config(page_title="Claims Cost Intelligence", page_icon="🏥", layout="wide")


@st.cache_resource
def get_bundle() -> inference.ModelBundle:
    """Load the exported models once per session.

    Returns:
        The loaded model bundle.
    """
    return inference.load_bundle()


def render_comparison(metadata: dict) -> None:
    """Show how every candidate model scored on the held-out fold.

    Args:
        metadata: Export metadata carrying all candidates' metrics.
    """
    st.subheader("Model comparison")
    st.caption(
        "Scored on a test fold split by beneficiary, so no patient appears in "
        "both training and test. Selection rule: 0.6 x macro-F1 + 0.4 x accuracy."
    )

    frame = pd.DataFrame(metadata["classifier"]["all_candidates"]).T
    frame.index.name = "model"
    columns = [c for c in ["accuracy", "macro_f1", "roc_auc_ovr", "selection_score"] if c in frame]
    st.dataframe(
        frame[columns].style.format("{:.4f}").highlight_max(axis=0, color="#1B5E20"),
        width="stretch",
    )
    st.success(
        f"Production classifier: **{metadata['classifier']['name']}** "
        f"(selection score {metadata['classifier']['metrics']['selection_score']:.4f})"
    )

    st.subheader("Regression candidates")
    regression = pd.DataFrame(metadata["regressor"]["all_candidates"]).T
    regression.index.name = "model"
    st.dataframe(regression.style.format("{:.4f}"), width="stretch")
    st.info(
        f"Production regressor: **{metadata['regressor']['name']}** (lowest log-space RMSE). "
        "The regression is weak by design of the data, not by oversight -- six features "
        "cannot explain charges spanning $129 to $32.6M."
    )


def render_prediction_form(bundle: inference.ModelBundle) -> None:
    """Render the live scoring form and its results.

    Args:
        bundle: Loaded models and explainer.
    """
    st.subheader("Score a claim")

    with st.form("claim"):
        left, right = st.columns(2)
        with left:
            dgns_inp = st.text_input("Inpatient diagnosis code (ICD-10)", value="I10")
            dgns_out = st.text_input("Outpatient diagnosis code (ICD-10)", value="E119")
            poa = st.selectbox("Present on admission", ["Y", "U"])
        with right:
            claims_inp = st.number_input("Inpatient claim count", min_value=0, value=3)
            claims_out = st.number_input("Outpatient claim count", min_value=0, value=7)
            income = st.number_input("State median income ($)", min_value=1.0, value=60510.0)
        submitted = st.form_submit_button("Predict", width="stretch")

    if not submitted:
        return

    try:
        claim = ClaimFeatures(
            PRNCPAL_DGNS_CD_inp=dgns_inp.strip().upper(),
            PRNCPAL_DGNS_CD_out=dgns_out.strip().upper(),
            CLM_E_POA_IND_SW1=poa,
            Number_of_Claims_inp=int(claims_inp),
            Number_of_Claims_out=int(claims_out),
            Median_Income=float(income),
        )
    except ValidationError as error:
        # Same schema the API enforces, so the dashboard rejects exactly what a
        # POST to /predict/tier would reject.
        st.error(f"Invalid input: {error.errors()[0]['msg']}")
        return

    frame = claim.to_frame()
    tier_result = inference.predict_tier(bundle, frame)
    charge_result = inference.predict_charge(bundle, frame)

    left, right = st.columns([1, 2])
    with left:
        st.markdown(
            f"<div style='background:{TIER_COLORS[tier_result.tier]};color:white;"
            f"padding:20px;border-radius:8px;text-align:center'>"
            f"<div style='font-size:14px;opacity:.85'>PREDICTED COST TIER</div>"
            f"<div style='font-size:52px;font-weight:700'>{tier_result.tier}</div></div>",
            unsafe_allow_html=True,
        )
        st.metric("Confidence", f"{tier_result.confidence:.1%}")
        st.metric("Predicted charge", f"${charge_result.predicted_charge:,.0f}")
    with right:
        st.write(f"**Interpretation:** {tier_result.tier_description}")
        st.bar_chart(pd.Series({f"Tier {k}": v for k, v in tier_result.probabilities.items()}))

    st.subheader("Why this prediction")
    st.caption("SHAP contributions toward the predicted tier, largest effect first.")
    contributions = pd.DataFrame([c.model_dump() for c in tier_result.top_contributions])
    if contributions.empty:
        st.info("Explanations are unavailable because the explainer did not load.")
    else:
        st.dataframe(contributions, width="stretch", hide_index=True)


def render_methodology(metadata: dict) -> None:
    """Explain the corrections applied to the legacy pipeline.

    Args:
        metadata: Export metadata, used for split sizes.
    """
    st.subheader("Corrected methodology")
    st.markdown(
        f"""
This rebuild fixes five defects carried by the original R implementation:

1. **Beneficiary leakage** — `BENE_ID` repeats about 8.5 times per patient. The
   original random row split shared 4,074 of 5,416 beneficiaries between train
   and test. Splitting on the beneficiary brings that to zero
   ({metadata["train_beneficiaries"]:,} train / {metadata["test_beneficiaries"]:,} test).
2. **Encoding leakage** — diagnosis frequencies and the POA dummy were fit on the
   full dataset before splitting. They are now fit on the training fold only.
3. **State-code ambiguity** — both provider state columns agree on every row, so
   the inpatient column drives the income join; a schema check enforces it.
4. **Incomplete POA encoding** — the original produced only a `_Y` column with no
   handling for unseen values; now a one-hot encoder that ignores unknowns.
5. **A silently dropped state** — Wyoming's median income was stored as the text
   `"$60,510 "`. Coercion turned it to `NA` and `na.omit()` removed all 47 Wyoming
   claims, recorded as routine missing-value handling. Parsed properly, all
   46,059 rows survive.
"""
    )


def render_about() -> None:
    """Explain how this deployment relates to the FastAPI service."""
    with st.sidebar:
        st.header("About this demo")
        st.markdown(
            """
This dashboard scores claims **in-process** from the committed
`exported_model/`, so it runs standalone with no backend service.

The project also ships a **FastAPI inference service** (`/predict/tier`,
`/predict/charge`, `/health`) that wraps the same scoring module. It is
Dockerized, covered by the test suite, and smoke-tested in CI — run it with
`docker compose up`. It is not deployed to a public URL.
"""
        )
        st.caption(f"Features: {', '.join(config.FEATURE_COLUMNS)}")


def main() -> None:
    """Compose the dashboard."""
    st.title("Healthcare Claims Cost Intelligence")
    st.caption(
        "Predicting cost tier and total charge for Medicare-style claims, so high-cost "
        "claims reach utilization review before they settle."
    )

    try:
        bundle = get_bundle()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    render_about()

    comparison_tab, predict_tab, method_tab = st.tabs(
        ["Model comparison", "Score a claim", "Methodology"]
    )
    with comparison_tab:
        render_comparison(bundle.metadata)
    with predict_tab:
        render_prediction_form(bundle)
    with method_tab:
        render_methodology(bundle.metadata)


main()
