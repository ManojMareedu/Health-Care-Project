"""Request/response schemas shared by the API and the dashboard.

These live outside ``app/api`` so the dashboard can reuse the exact same input
validation without importing FastAPI. Both surfaces then reject the same inputs
for the same reasons, which is the point -- a dashboard that accepted values the
API rejects would be demonstrating something the service cannot actually do.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from . import config


class ClaimFeatures(BaseModel):
    """One claim's features, matching the columns the pipeline was trained on."""

    PRNCPAL_DGNS_CD_inp: str = Field(
        ..., min_length=1, description="Inpatient principal ICD-10 diagnosis code", examples=["I10"]
    )
    PRNCPAL_DGNS_CD_out: str = Field(
        ...,
        min_length=1,
        description="Outpatient principal ICD-10 diagnosis code",
        examples=["E119"],
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
