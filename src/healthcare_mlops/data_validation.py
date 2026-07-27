"""Schema validation for the raw claims extract and the modeling frame.

The legacy pipeline had no validation step at all: a null charge, an
out-of-range state code, or a malformed ICD string would flow straight into
model fitting and surface later as an inscrutable metric. These schemas make
those failures loud and early.

The dataframe-level check ``_states_agree`` is load-bearing rather than
decorative -- it is what licenses the decision documented in
``data_ingestion`` to use ``PRVDR_STATE_CD_inp`` alone.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError

from . import config

# Well-formed ICD-10 code of either system: 3 to 7 alphanumeric characters, no
# punctuation or whitespace. This is the gate -- anything outside it is a
# genuinely malformed string and fails validation.
ICD10_PATTERN = r"^[0-9A-Z]{3,7}$"

# ICD-10-CM *diagnosis* codes specifically start with a letter (the U chapter is
# included -- U07.1 is COVID-19 and appears in this extract). Codes matching
# ICD10_PATTERN but not this one are ICD-10-PCS *procedure* codes, which do not
# belong in a diagnosis column.
ICD10_CM_PATTERN = r"^[A-Z][0-9A-Z]{2,6}$"

# FIPS-style state codes present in the income lookup.
MIN_STATE_CODE = 1
MAX_STATE_CODE = 53


def _states_agree(frame: pd.DataFrame) -> bool:
    """Assert the inpatient and outpatient provider state codes never disagree.

    This encodes the invariant that justifies dropping ``PRVDR_STATE_CD_out``.
    If a future extract violates it, the ``_inp``-only rule is no longer
    information-preserving and the merge strategy must be revisited -- so this
    fails the pipeline rather than warning.

    Args:
        frame: Frame carrying both state code columns.

    Returns:
        True when every row agrees.
    """
    return bool((frame["PRVDR_STATE_CD_inp"] == frame["PRVDR_STATE_CD_out"]).all())


CLAIMS_SCHEMA = pa.DataFrameSchema(
    columns={
        config.GROUP_KEY: pa.Column(int, nullable=False),
        "PRNCPAL_DGNS_CD_inp": pa.Column(str, pa.Check.str_matches(ICD10_PATTERN), nullable=False),
        "PRNCPAL_DGNS_CD_out": pa.Column(str, pa.Check.str_matches(ICD10_PATTERN), nullable=False),
        "CLM_E_POA_IND_SW1": pa.Column(str, pa.Check.isin(["Y", "U"]), nullable=False),
        "PRVDR_STATE_CD_inp": pa.Column(
            int, pa.Check.in_range(MIN_STATE_CODE, MAX_STATE_CODE), nullable=False
        ),
        "PRVDR_STATE_CD_out": pa.Column(
            int, pa.Check.in_range(MIN_STATE_CODE, MAX_STATE_CODE), nullable=False
        ),
        "Number_of_Claims_inp": pa.Column(int, pa.Check.ge(0), nullable=False),
        "Number_of_Claims_out": pa.Column(int, pa.Check.ge(0), nullable=False),
        "CLM_TOT_CHRG_AMT_inp": pa.Column(float, pa.Check.gt(0), nullable=False),
        "CLM_TOT_CHRG_AMT_out": pa.Column(float, pa.Check.gt(0), nullable=False),
    },
    checks=pa.Check(
        _states_agree,
        error=(
            "PRVDR_STATE_CD_inp and PRVDR_STATE_CD_out disagree; the "
            "inpatient-state-only income merge is no longer information-preserving"
        ),
    ),
    strict=False,
    coerce=True,
)

INCOME_SCHEMA = pa.DataFrameSchema(
    columns={
        "PRVDR_STATE_CD": pa.Column(
            int, pa.Check.in_range(MIN_STATE_CODE, MAX_STATE_CODE), unique=True
        ),
        "STATE": pa.Column(str, nullable=False),
        "Median_Income": pa.Column(float, pa.Check.gt(0), nullable=False),
    },
    strict=False,
    coerce=True,
)

MODELING_SCHEMA = pa.DataFrameSchema(
    columns={
        config.TARGET_CHARGE: pa.Column(float, pa.Check.gt(0), nullable=False),
        config.TARGET_LOG_CHARGE: pa.Column(float, nullable=False),
        config.TARGET_TIER: pa.Column(int, pa.Check.isin(config.TIER_LABELS)),
        # A null here means a state code missed the income lookup -- the left
        # join is intentionally allowed to produce it so validation can catch it.
        "Median_Income": pa.Column(float, pa.Check.gt(0), nullable=False),
    },
    strict=False,
    coerce=True,
)


def report_procedure_codes_in_diagnosis(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Find ICD-10-PCS procedure codes sitting in a diagnosis column.

    The legacy extract carries three rows whose ``PRNCPAL_DGNS_CD_out`` holds a
    procedure code (``0FB43ZZ``, ``0FC44ZZ``) rather than a diagnosis code. That
    is a data-quality defect inherited from the upstream CMS join, not a schema
    violation, and three rows out of 46,059 do not justify failing the pipeline.

    So it is reported rather than raised -- the count is logged by the training
    pipeline and the codes are frequency-encoded like any other category. This
    function exists so the contamination stays visible instead of silent, which
    is the failure mode the legacy pipeline had.

    Args:
        frame: Claims frame with both diagnosis columns.

    Returns:
        Mapping of column name to the sorted distinct offending codes. Empty
        lists mean the column is clean.
    """
    findings: dict[str, list[str]] = {}
    for column in config.FREQUENCY_ENCODED_COLUMNS:
        values = frame[column].astype(str)
        offenders = values[~values.str.match(ICD10_CM_PATTERN)]
        findings[column] = sorted(offenders.unique())
    return findings


def validate_claims(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the raw claims extract.

    Args:
        frame: Freshly loaded claims frame.

    Returns:
        The validated (and type-coerced) frame.

    Raises:
        SchemaError: If any column or the cross-column state invariant fails.
    """
    return CLAIMS_SCHEMA.validate(frame, lazy=False)


def validate_income(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the state income lookup.

    Args:
        frame: Freshly loaded income frame.

    Returns:
        The validated frame.

    Raises:
        SchemaError: If a state code is duplicated or out of range.
    """
    return INCOME_SCHEMA.validate(frame, lazy=False)


def validate_modeling_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the post-merge frame that feature engineering will consume.

    Args:
        frame: Output of ``data_ingestion.build_modeling_frame``.

    Returns:
        The validated frame.

    Raises:
        SchemaError: If targets are malformed or the income join left nulls.
    """
    return MODELING_SCHEMA.validate(frame, lazy=False)


__all__ = [
    "CLAIMS_SCHEMA",
    "INCOME_SCHEMA",
    "MODELING_SCHEMA",
    "SchemaError",
    "report_procedure_codes_in_diagnosis",
    "validate_claims",
    "validate_income",
    "validate_modeling_frame",
]
