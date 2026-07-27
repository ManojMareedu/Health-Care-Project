"""Load the raw Excel sources and derive the modeling frame.

Ingestion boundary
------------------
``Patient_Claim_Data.xlsx`` is treated as the ingestion boundary. It was built by
the legacy notebook from CMS inpatient/outpatient extracts that are not in this
repository (too large to redistribute), so this module reads the merged file
rather than attempting to regenerate it.

State-code rule (resolves a legacy ambiguity)
---------------------------------------------
The source carries both ``PRVDR_STATE_CD_inp`` and ``PRVDR_STATE_CD_out``. The
legacy R script kept only ``_inp`` and silently discarded ``_out``, leaving the
reader unable to tell whether that dropped real signal.

**Decision: ``PRVDR_STATE_CD_inp`` is the beneficiary's state for income
context, and ``PRVDR_STATE_CD_out`` is dropped as redundant.**

This is safe rather than arbitrary: the two columns agree on all 46,059 rows of
the current extract, so ``_out`` carries no information ``_inp`` does not. The
rule is not left on trust -- ``data_validation`` asserts the equality as a schema
invariant, so a future extract where the two disagree fails loudly at validation
instead of quietly modeling on a half-truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def load_claims(path=None) -> pd.DataFrame:
    """Read the merged inpatient/outpatient claims extract.

    Args:
        path: Override for the claims workbook. Defaults to ``config.CLAIMS_FILE``.

    Returns:
        The raw claims frame, unmodified apart from being read into memory.
    """
    return pd.read_excel(path or config.CLAIMS_FILE)


def _parse_currency(series: pd.Series) -> pd.Series:
    """Coerce a mixed numeric/currency-string column to float.

    The income workbook stores 50 states as plain integers and Wyoming as the
    text ``"$60,510 "`` (trailing non-breaking space). Stripping currency
    punctuation before coercion is what keeps that row a number.

    Args:
        series: Raw ``Median_Income`` values, possibly of mixed type.

    Returns:
        A float series with currency formatting removed.
    """
    cleaned = (
        series.astype(str).str.replace(r"[\$,\s ]", "", regex=True).replace({"": None, "nan": None})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def load_income(path=None) -> pd.DataFrame:
    """Read the state median-income lookup, normalising currency formatting.

    Why the parsing step matters
    ----------------------------
    One row (Wyoming) is stored as a formatted currency string rather than a
    number. The legacy R pipeline ran ``as.numeric()`` over this column, which
    turned that single value into ``NA``, and a later ``na.omit()`` dropped every
    row that inherited it -- all 47 Wyoming claims.

    That loss was recorded in the original write-up as routine "missing value
    handling", but nothing was actually missing: a formatting artifact silently
    removed an entire state from an analysis whose stated purpose includes
    health-equity reporting by state income. Parsing the string here keeps all 51
    states and all 46,059 rows in the model.

    Args:
        path: Override for the income workbook. Defaults to ``config.INCOME_FILE``.

    Returns:
        One row per state with ``PRVDR_STATE_CD``, ``STATE``, and a numeric
        ``Median_Income``.
    """
    income = pd.read_excel(path or config.INCOME_FILE)
    income["Median_Income"] = _parse_currency(income["Median_Income"])
    return income


def merge_income(claims: pd.DataFrame, income: pd.DataFrame) -> pd.DataFrame:
    """Attach state median income using the inpatient provider state code.

    See the module docstring for why ``_inp`` is the join key and ``_out`` is
    dropped. A left join is used deliberately: an unmatched state code should
    surface as a null that validation rejects, not vanish with the row.

    Args:
        claims: Raw claims frame.
        income: State income lookup.

    Returns:
        Claims with ``STATE`` and ``Median_Income`` attached.
    """
    return claims.merge(
        income,
        left_on="PRVDR_STATE_CD_inp",
        right_on="PRVDR_STATE_CD",
        how="left",
    ).drop(columns=["PRVDR_STATE_CD"])


def add_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive the regression and classification targets.

    ``TOTAL_CHARGE`` sums the inpatient and outpatient charge amounts. The
    regression target is its natural log, because raw charges span five orders of
    magnitude with an extreme right tail (see ``legacy/figures``); the log
    transform is what makes a linear model meaningful at all.

    Args:
        frame: Claims frame carrying both charge columns.

    Returns:
        The frame with ``TOTAL_CHARGE``, ``LOG_TOTAL_CHARGE``, and ``TC_class`` added.
    """
    out = frame.copy()
    out[config.TARGET_CHARGE] = out["CLM_TOT_CHRG_AMT_inp"] + out["CLM_TOT_CHRG_AMT_out"]
    out[config.TARGET_LOG_CHARGE] = np.log(out[config.TARGET_CHARGE])
    out[config.TARGET_TIER] = pd.cut(
        out[config.TARGET_CHARGE],
        bins=config.TIER_BINS,
        labels=config.TIER_LABELS,
    ).astype(int)
    return out


def build_modeling_frame(claims_path=None, income_path=None) -> pd.DataFrame:
    """Run the full ingestion path: load, merge income, derive targets.

    Args:
        claims_path: Override for the claims workbook.
        income_path: Override for the income workbook.

    Returns:
        A frame ready for validation and feature engineering.
    """
    claims = load_claims(claims_path)
    income = load_income(income_path)
    return add_targets(merge_income(claims, income))
