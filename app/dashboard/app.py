"""Streamlit dashboard for the claims cost intelligence models.

Three views: how the candidate models compared, what drives the production
classifier, and a live prediction form with tier interpretation.

Predictions go through the FastAPI service rather than loading the model
in-process. That keeps one inference path in the system -- if the API is wrong,
the dashboard is wrong in the same way, instead of quietly disagreeing with it.

Run locally with::

    streamlit run app/dashboard/app.py
"""

from __future__ import annotations

import json
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = 15

TIER_COLORS = {1: "#2E7D32", 2: "#7CB342", 3: "#F9A825", 4: "#EF6C00", 5: "#C62828"}

st.set_page_config(page_title="Claims Cost Intelligence", page_icon="🏥", layout="wide")


@st.cache_data(ttl=60)
def fetch_model_info() -> dict | None:
    """Read training metadata from the API.

    Returns:
        The metadata mapping, or None when the API is unreachable.
    """
    try:
        response = requests.get(f"{API_URL}/model-info", timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def api_health() -> dict | None:
    """Read the API health payload.

    Returns:
        The health mapping, or None when the API is unreachable.
    """
    try:
        response = requests.get(f"{API_URL}/health", timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


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

    candidates = metadata["classifier"]["all_candidates"]
    frame = pd.DataFrame(candidates).T
    frame.index.name = "model"
    display_columns = [
        c for c in ["accuracy", "macro_f1", "roc_auc_ovr", "selection_score"] if c in frame
    ]
    st.dataframe(
        frame[display_columns].style.format("{:.4f}").highlight_max(axis=0, color="#1B5E20"),
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
    st.info(f"Production regressor: **{metadata['regressor']['name']}** (lowest log-space RMSE)")


def render_prediction_form(metadata: dict) -> None:
    """Render the live prediction form and results.

    Args:
        metadata: Export metadata, used for tier descriptions.
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

    payload = {
        "PRNCPAL_DGNS_CD_inp": dgns_inp.strip().upper(),
        "PRNCPAL_DGNS_CD_out": dgns_out.strip().upper(),
        "CLM_E_POA_IND_SW1": poa,
        "Number_of_Claims_inp": int(claims_inp),
        "Number_of_Claims_out": int(claims_out),
        "Median_Income": float(income),
    }

    try:
        tier_response = requests.post(
            f"{API_URL}/predict/tier", json=payload, timeout=REQUEST_TIMEOUT
        )
        charge_response = requests.post(
            f"{API_URL}/predict/charge", json=payload, timeout=REQUEST_TIMEOUT
        )
    except requests.RequestException as error:
        st.error(f"Could not reach the API at {API_URL}: {error}")
        return

    if tier_response.status_code == 422:
        st.warning(f"The API rejected this input: {tier_response.json()['detail']}")
        return
    if not tier_response.ok:
        st.error(f"API returned {tier_response.status_code}: {tier_response.text}")
        return

    tier_result = tier_response.json()
    tier = tier_result["tier"]

    left, right = st.columns([1, 2])
    with left:
        st.markdown(
            f"<div style='background:{TIER_COLORS[tier]};color:white;padding:20px;"
            f"border-radius:8px;text-align:center'>"
            f"<div style='font-size:14px;opacity:.85'>PREDICTED COST TIER</div>"
            f"<div style='font-size:52px;font-weight:700'>{tier}</div></div>",
            unsafe_allow_html=True,
        )
        st.metric("Confidence", f"{tier_result['confidence']:.1%}")
        if charge_response.ok:
            st.metric("Predicted charge", f"${charge_response.json()['predicted_charge']:,.0f}")
    with right:
        st.write(f"**Interpretation:** {tier_result['tier_description']}")
        probabilities = pd.Series({f"Tier {k}": v for k, v in tier_result["probabilities"].items()})
        st.bar_chart(probabilities)

    st.subheader("Why this prediction")
    st.caption("SHAP contributions toward the predicted tier, largest effect first.")
    contributions = pd.DataFrame(tier_result["top_contributions"])
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
This rebuild fixes defects carried by the original R implementation:

1. **Beneficiary leakage** — `BENE_ID` repeats about 8.5 times per patient. The
   original random row split shared 4,074 of 5,416 beneficiaries between train
   and test. Splitting on the beneficiary brings that to zero
   ({metadata["train_beneficiaries"]:,} train / {metadata["test_beneficiaries"]:,} test).
2. **Encoding leakage** — diagnosis frequencies and the POA dummy were fit on the
   full dataset before splitting. They are now fit on the training fold only.
3. **State-code ambiguity** — both provider state columns agree on every row, so
   the inpatient column drives the income join; a schema check enforces it.
4. **Incomplete POA encoding** — the original produced only a `_Y` column with no
   handling for unseen values; now a proper one-hot encoder that ignores unknowns.
5. **A silently dropped state** — Wyoming's median income was stored as the text
   `"$60,510 "`. Coercion turned it to `NA` and `na.omit()` removed all 47 Wyoming
   claims, recorded as routine missing-value handling. Parsed properly, all
   46,059 rows survive.
"""
    )


def main() -> None:
    """Compose the dashboard."""
    st.title("Healthcare Claims Cost Intelligence")
    st.caption(
        "Predicting cost tier and total charge for Medicare-style claims, so high-cost "
        "claims reach utilization review before they settle."
    )

    health = api_health()
    if health is None:
        st.error(
            f"The prediction API is not reachable at `{API_URL}`. "
            "Start it with `docker compose up` or `uvicorn app.api.main:app`."
        )
        st.stop()
    if health["status"] != "healthy":
        st.warning(f"API is degraded: {json.dumps(health)}")

    metadata = fetch_model_info()
    if metadata is None:
        st.error("The API is up but did not return model metadata.")
        st.stop()

    comparison_tab, predict_tab, method_tab = st.tabs(
        ["Model comparison", "Score a claim", "Methodology"]
    )
    with comparison_tab:
        render_comparison(metadata)
    with predict_tab:
        render_prediction_form(metadata)
    with method_tab:
        render_methodology(metadata)


main()
