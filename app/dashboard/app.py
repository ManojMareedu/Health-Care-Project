"""Streamlit dashboard for the claims cost intelligence models.

Predictions are scored **in-process** against the committed ``exported_model/``
through ``src.healthcare_mlops.inference`` -- the same module the FastAPI service
wraps in HTTP. Sharing that module rather than calling the API over the network
means the two surfaces cannot disagree, and it leaves this app with no runtime
dependency on any other running service, so it deploys to a free host on its own.

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

# One palette for the whole app: tier colour is the only semantic colour used,
# and it means the same thing on the badge, the probability chart, and the
# tier-reference table.
TIER_COLORS = {1: "#2E7D32", 2: "#7CB342", 3: "#F9A825", 4: "#EF6C00", 5: "#C62828"}
TIER_ACTIONS = {
    1: "Auto-approve",
    2: "Standard processing",
    3: "Sample for review",
    4: "Route to utilization review",
    5: "Specialist handling",
}

# Diagnosis codes seen most often in the training data, offered so a reviewer can
# try the tool without needing to know ICD-10 by heart.
EXAMPLE_DIAGNOSES = ["I10", "E119", "J449", "N179", "Z3400", "C50919", "I509", "U071"]

st.set_page_config(
    page_title="Claims Cost Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_bundle() -> inference.ModelBundle:
    """Load the exported models once per session.

    Returns:
        The loaded model bundle.
    """
    return inference.load_bundle()


def render_header(metadata: dict) -> None:
    """Render the landing block: what this is, who it is for, how good it is.

    Args:
        metadata: Export metadata, used for the headline metrics.
    """
    st.title("Healthcare Claims Cost Intelligence")
    st.markdown(
        "**Flag high-cost claims before they settle.** Built for payer and "
        "hospital-finance teams who settle claims faster than they can review "
        "them: enter a claim's details and get its predicted cost tier, an "
        "estimated total charge, and the factors driving the estimate."
    )

    classifier = metadata["classifier"]["metrics"]
    regressor = metadata["regressor"]["metrics"]
    left, middle, right, far_right = st.columns(4)
    left.metric(
        "Tier accuracy",
        f"{classifier['accuracy']:.1%}",
        help="Share of held-out claims assigned the correct cost tier.",
    )
    middle.metric(
        "Macro-F1",
        f"{classifier['macro_f1']:.3f}",
        help="Averages F1 equally across all five tiers, so the rare "
        "catastrophic tier counts as much as the common ones.",
    )
    right.metric(
        "Charge R²",
        f"{regressor['r2_log']:.3f}",
        help="Variance explained in log-charge space by the regression model.",
    )
    far_right.metric(
        "Median charge error",
        f"${regressor['median_abs_dollar_error']:,.0f}",
        help="Typical dollar gap between predicted and actual total charge.",
    )
    st.caption(
        "Scored on a held-out fold split by beneficiary, so no patient appears in "
        "both training and test. Synthetic CMS-style data — no real patient information."
    )


def render_sidebar(metadata: dict) -> None:
    """Render persistent context about the models and the deployment.

    Args:
        metadata: Export metadata.
    """
    with st.sidebar:
        st.subheader("Production models")
        st.markdown(
            f"**Cost tier:** `{metadata['classifier']['name']}`  \n"
            f"**Total charge:** `{metadata['regressor']['name']}`"
        )
        st.caption(f"Selection rule: {metadata['classifier']['selection_rule']}")

        st.divider()
        st.subheader("Cost tiers")
        for tier, description in config.TIER_DESCRIPTIONS.items():
            band = description.split(" - ")[0]
            st.markdown(
                f"<span style='display:inline-block;width:10px;height:10px;"
                f"border-radius:50%;background:{TIER_COLORS[tier]};"
                f"margin-right:8px'></span>**Tier {tier}** &nbsp;{band}",
                unsafe_allow_html=True,
            )

        st.divider()
        st.caption(
            "This dashboard scores claims in-process from the committed model "
            "artifacts, so it runs standalone. The project also ships a FastAPI "
            "service wrapping the same scoring module, runnable with "
            "`docker compose up`."
        )


def _tier_badge(tier: int, confidence: float) -> str:
    """Build the coloured tier badge markup.

    Args:
        tier: Predicted tier, 1-5.
        confidence: Probability assigned to that tier.

    Returns:
        HTML for the badge.
    """
    return (
        f"<div style='background:{TIER_COLORS[tier]};color:white;padding:18px 20px;"
        f"border-radius:10px;text-align:center;line-height:1.25'>"
        f"<div style='font-size:12px;letter-spacing:.08em;opacity:.85'>"
        f"PREDICTED COST TIER</div>"
        f"<div style='font-size:56px;font-weight:700'>{tier}</div>"
        f"<div style='font-size:14px;opacity:.95'>{TIER_ACTIONS[tier]}</div>"
        f"<div style='font-size:12px;opacity:.8;margin-top:6px'>"
        f"{confidence:.0%} confidence</div></div>"
    )


def render_prediction_form(bundle: inference.ModelBundle) -> None:
    """Render the scoring form and its results.

    Args:
        bundle: Loaded models and explainer.
    """
    st.subheader("Score a claim")
    st.caption(
        "All six fields are required. Defaults describe a typical hypertensive "
        "patient with moderate outpatient activity."
    )

    with st.form("claim"):
        clinical, volume = st.columns(2)
        with clinical:
            st.markdown("**Clinical**")
            dgns_inp = st.selectbox(
                "Inpatient principal diagnosis (ICD-10)",
                EXAMPLE_DIAGNOSES,
                index=0,
                help="Principal diagnosis on the inpatient claim. I10 is essential hypertension.",
                accept_new_options=True,
            )
            dgns_out = st.selectbox(
                "Outpatient principal diagnosis (ICD-10)",
                EXAMPLE_DIAGNOSES,
                index=1,
                help="Principal diagnosis on the outpatient claim. "
                "E119 is type 2 diabetes without complications.",
                accept_new_options=True,
            )
            poa = st.radio(
                "Present on admission",
                ["Y", "U"],
                horizontal=True,
                help="Y = condition present on admission. U = documentation "
                "insufficient to determine.",
            )
        with volume:
            st.markdown("**Utilisation**")
            claims_inp = st.number_input(
                "Inpatient claim count",
                min_value=0,
                max_value=500,
                value=3,
                step=1,
                help="Number of inpatient claims for this beneficiary. Typical range 1-20.",
            )
            claims_out = st.number_input(
                "Outpatient claim count",
                min_value=0,
                max_value=500,
                value=7,
                step=1,
                help="Number of outpatient claims for this beneficiary. Typical range 1-40.",
            )
            income = st.number_input(
                "State median household income (USD)",
                min_value=1.0,
                max_value=250_000.0,
                value=60_510.0,
                step=1_000.0,
                format="%.0f",
                help="Median household income for the provider's state. "
                "Observed range in the data is roughly $45,000-$95,000.",
            )
        submitted = st.form_submit_button("Predict cost tier", width="stretch")

    if not submitted:
        return

    try:
        claim = ClaimFeatures(
            PRNCPAL_DGNS_CD_inp=str(dgns_inp).strip().upper(),
            PRNCPAL_DGNS_CD_out=str(dgns_out).strip().upper(),
            CLM_E_POA_IND_SW1=poa,
            Number_of_Claims_inp=int(claims_inp),
            Number_of_Claims_out=int(claims_out),
            Median_Income=float(income),
        )
    except ValidationError as error:
        # Same schema the API enforces, so the dashboard rejects exactly what a
        # POST to /predict/tier would reject -- shown as readable field errors
        # rather than a traceback.
        st.error("This claim could not be scored. Please correct the following:")
        for issue in error.errors():
            field = issue["loc"][0] if issue["loc"] else "input"
            st.markdown(f"- **{field}** — {issue['msg']}")
        return

    frame = claim.to_frame()
    tier_result = inference.predict_tier(bundle, frame)
    charge_result = inference.predict_charge(bundle, frame)

    st.divider()
    badge, numbers, drivers = st.columns([1, 1, 2])

    with badge:
        st.markdown(_tier_badge(tier_result.tier, tier_result.confidence), unsafe_allow_html=True)

    with numbers:
        st.metric("Estimated total charge", f"${charge_result.predicted_charge:,.0f}")
        st.metric("Tier confidence", f"{tier_result.confidence:.1%}")
        st.caption(tier_result.tier_description)

    with drivers:
        st.markdown("**What drove this prediction**")
        contributions = pd.DataFrame([c.model_dump() for c in tier_result.top_contributions])
        if contributions.empty:
            st.info("Feature attributions are unavailable because the explainer did not load.")
        else:
            contributions["impact"] = contributions["contribution"].abs()
            display = contributions.rename(
                columns={"feature": "Feature", "direction": "Effect on this tier"}
            )
            st.dataframe(
                display[["Feature", "Effect on this tier", "impact"]],
                width="stretch",
                hide_index=True,
                column_config={
                    "impact": st.column_config.ProgressColumn(
                        "Strength",
                        help="Absolute SHAP contribution toward the predicted tier.",
                        min_value=0.0,
                        max_value=float(contributions["impact"].max()),
                        format="%.2f",
                    )
                },
            )
            st.caption(
                "SHAP values: how much each feature pushed this claim toward the tier shown."
            )

    st.markdown("**Probability across all tiers**")
    probabilities = pd.DataFrame(
        {
            "Tier": [f"Tier {tier}" for tier in tier_result.probabilities],
            "Probability": list(tier_result.probabilities.values()),
        }
    )
    st.bar_chart(probabilities, x="Tier", y="Probability", height=220)


def render_comparison(metadata: dict) -> None:
    """Show how every candidate model scored on the held-out fold.

    Args:
        metadata: Export metadata carrying all candidates' metrics.
    """
    st.subheader("Cost tier classification")
    st.caption(
        "Every candidate scored on the same held-out fold, split by beneficiary. "
        f"Production model selected by: {metadata['classifier']['selection_rule']}."
    )

    winner = metadata["classifier"]["name"]
    frame = pd.DataFrame(metadata["classifier"]["all_candidates"]).T
    frame.index.name = "Model"
    columns = [c for c in ["accuracy", "macro_f1", "roc_auc_ovr", "selection_score"] if c in frame]
    display = frame[columns].rename(
        columns={
            "accuracy": "Accuracy",
            "macro_f1": "Macro-F1",
            "roc_auc_ovr": "ROC-AUC",
            "selection_score": "Selection score",
        }
    )
    display.insert(0, "Production", ["✅" if name == winner else "" for name in display.index])
    st.dataframe(
        display,
        width="stretch",
        column_config={
            "Accuracy": st.column_config.NumberColumn(format="%.4f"),
            "Macro-F1": st.column_config.NumberColumn(format="%.4f"),
            "ROC-AUC": st.column_config.NumberColumn(format="%.4f"),
            "Selection score": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    tuning = metadata.get("tuning") or {}
    if tuning:
        with st.expander("Hyperparameter search"):
            st.caption(
                "Randomised search over grouped cross-validation folds (GroupKFold "
                "on beneficiary), scored by macro-F1 so the rare tier counts."
            )
            for name, result in tuning.items():
                st.markdown(f"**{name}** — CV macro-F1 {result['cv_macro_f1']:.4f}")
                st.json(
                    {k.replace("model__", ""): v for k, v in result["best_params"].items()},
                    expanded=False,
                )

    st.subheader("Total charge regression")
    st.caption(
        "Predicting log(total charge). Selected by lowest log-space RMSE. "
        "Linear and tree-based candidates on the identical split."
    )
    regression_winner = metadata["regressor"]["name"]
    regression = pd.DataFrame(metadata["regressor"]["all_candidates"]).T
    regression.index.name = "Model"
    regression_display = regression.rename(
        columns={
            "rmse_log": "RMSE (log)",
            "r2_log": "R²",
            "mae_log": "MAE (log)",
            "median_abs_dollar_error": "Median $ error",
        }
    )
    regression_display.insert(
        0, "Production", ["✅" if n == regression_winner else "" for n in regression_display.index]
    )
    st.dataframe(
        regression_display,
        width="stretch",
        column_config={
            "RMSE (log)": st.column_config.NumberColumn(format="%.4f"),
            "R²": st.column_config.NumberColumn(format="%.4f"),
            "MAE (log)": st.column_config.NumberColumn(format="%.4f"),
            "Median $ error": st.column_config.NumberColumn(format="$%.0f"),
        },
    )
    st.info(
        "The linear models explain about a quarter of the variance; the "
        "gradient-boosted model explains far more on the same features. The "
        "relationship between utilisation and charge is strongly non-linear."
    )


def render_methodology(metadata: dict) -> None:
    """Explain the corrections applied to the legacy pipeline.

    Args:
        metadata: Export metadata, used for split sizes.
    """
    st.subheader("Corrected methodology")
    st.caption(
        "This project began as an academic R analysis. Rebuilding it surfaced "
        "five defects, each verified against the data before being fixed."
    )
    st.markdown(
        f"""
1. **Beneficiary leakage** — `BENE_ID` repeats about 8.5 times per patient, so the
   original random row split shared 4,074 of 5,416 beneficiaries between train and
   test. Splitting on the beneficiary brings that to zero
   ({metadata["train_beneficiaries"]:,} train / {metadata["test_beneficiaries"]:,} test).
2. **Encoding leakage** — diagnosis frequencies and the present-on-admission dummy
   were fit on the full dataset before splitting. They are now fit on the training
   fold only, with unseen codes falling back to the training-set minimum frequency.
3. **State-code ambiguity** — both provider state columns agree on every row, so the
   inpatient column drives the income join, and a schema check enforces that.
4. **Incomplete encoding** — the original produced only a `_Y` column with no handling
   for unseen values; now a one-hot encoder that ignores unknown categories.
5. **A silently dropped state** — Wyoming's median income was stored as the text
   `"$60,510 "`. Numeric coercion turned it to `NA` and a later `na.omit()` removed
   all 47 Wyoming claims, recorded at the time as routine missing-value handling.
   Parsed properly, all 46,059 rows survive.
"""
    )
    st.caption(
        f"Training fold: {metadata['train_rows']:,} rows · "
        f"Held-out fold: {metadata['test_rows']:,} rows"
    )


def main() -> None:
    """Compose the dashboard."""
    try:
        bundle = get_bundle()
    except FileNotFoundError as error:
        st.title("Healthcare Claims Cost Intelligence")
        st.error(str(error))
        st.stop()

    render_header(bundle.metadata)
    render_sidebar(bundle.metadata)

    predict_tab, comparison_tab, method_tab = st.tabs(
        ["Score a claim", "Model performance", "Methodology"]
    )
    with predict_tab:
        render_prediction_form(bundle)
    with comparison_tab:
        render_comparison(bundle.metadata)
    with method_tab:
        render_methodology(bundle.metadata)


main()
