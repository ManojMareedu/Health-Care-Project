"""Tests for the pandera schemas."""

from __future__ import annotations

import pandas as pd
import pytest

from src.healthcare_mlops import data_ingestion as ingestion
from src.healthcare_mlops import data_validation as validation


def test_valid_claims_pass(synthetic_claims):
    validated = validation.validate_claims(synthetic_claims)
    assert len(validated) == len(synthetic_claims)


def test_valid_income_passes(synthetic_income):
    assert len(validation.validate_income(synthetic_income)) == 53


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [
        ("CLM_E_POA_IND_SW1", "X"),
        ("PRNCPAL_DGNS_CD_inp", "!!bad"),
        ("PRVDR_STATE_CD_inp", 999),
        ("CLM_TOT_CHRG_AMT_inp", -1.0),
        ("Number_of_Claims_inp", -3),
    ],
)
def test_malformed_values_are_rejected(synthetic_claims, column, bad_value):
    corrupted = synthetic_claims.copy()
    corrupted.loc[corrupted.index[0], column] = bad_value
    with pytest.raises(validation.SchemaError):
        validation.validate_claims(corrupted)


def test_state_code_disagreement_is_rejected(synthetic_claims):
    """The _inp-only income merge is only valid while the two columns agree."""
    corrupted = synthetic_claims.copy()
    corrupted.loc[corrupted.index[0], "PRVDR_STATE_CD_out"] = (
        corrupted.loc[corrupted.index[0], "PRVDR_STATE_CD_inp"] + 1
    )
    with pytest.raises(validation.SchemaError):
        validation.validate_claims(corrupted)


def test_covid_code_is_accepted(synthetic_claims):
    """U07.1 is a real diagnosis code; a stricter regex wrongly rejected it."""
    frame = synthetic_claims.copy()
    frame.loc[frame.index[0], "PRNCPAL_DGNS_CD_inp"] = "U071"
    validation.validate_claims(frame)


def test_procedure_codes_are_reported_not_raised(synthetic_claims):
    """PCS codes in a diagnosis column are surfaced without failing the run."""
    frame = synthetic_claims.copy()
    frame.loc[frame.index[0], "PRNCPAL_DGNS_CD_out"] = "0FB43ZZ"
    validation.validate_claims(frame)

    findings = validation.report_procedure_codes_in_diagnosis(frame)
    assert findings["PRNCPAL_DGNS_CD_out"] == ["0FB43ZZ"]
    assert findings["PRNCPAL_DGNS_CD_inp"] == []


def test_currency_formatted_income_is_parsed():
    """Regression test for the defect that silently deleted Wyoming.

    The legacy pipeline coerced this string to NA and then dropped every row
    that inherited it. Parsing must keep the value.
    """
    raw = pd.Series([60510, "$60,510 ", "$1,234"])
    parsed = ingestion._parse_currency(raw)
    assert parsed.tolist() == [60510.0, 60510.0, 1234.0]
    assert parsed.notna().all()
