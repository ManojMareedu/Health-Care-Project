"""Shared fixtures.

The synthetic frame exists so most tests run in milliseconds without touching
the 2MB Excel source or the DVC cache -- which matters in CI, where the raw data
is not checked out at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.healthcare_mlops import config


@pytest.fixture
def synthetic_claims() -> pd.DataFrame:
    """Build a small claims frame with the same shape as the real extract.

    Beneficiaries deliberately repeat across rows, because that repetition is
    the thing the grouped-split tests are checking.

    Returns:
        A frame with claims columns and both targets.
    """
    rng = np.random.default_rng(0)
    rows = 300
    beneficiaries = rng.integers(1, 40, size=rows)
    states = rng.integers(1, 52, size=rows)
    charge_inp = rng.uniform(100, 50_000, size=rows)
    charge_out = rng.uniform(100, 50_000, size=rows)
    total = charge_inp + charge_out

    frame = pd.DataFrame(
        {
            config.GROUP_KEY: beneficiaries,
            # A long tail of codes, mirroring the real extract's 174/211
            # distinct values. Breadth matters: with only a handful of codes
            # every value lands in both folds and the leakage-sensitivity test
            # has nothing to detect.
            "PRNCPAL_DGNS_CD_inp": rng.choice(
                ["I10", "E119", "J449", "U071"] + [f"K{n:03d}" for n in range(60)],
                size=rows,
            ),
            "PRNCPAL_DGNS_CD_out": rng.choice(
                ["Z3400", "C50919", "I509"] + [f"M{n:03d}" for n in range(60)],
                size=rows,
            ),
            "CLM_E_POA_IND_SW1": rng.choice(["Y", "U"], size=rows),
            "PRVDR_STATE_CD_inp": states,
            "PRVDR_STATE_CD_out": states,
            "Number_of_Claims_inp": rng.integers(1, 20, size=rows),
            "Number_of_Claims_out": rng.integers(1, 20, size=rows),
            "CLM_TOT_CHRG_AMT_inp": charge_inp,
            "CLM_TOT_CHRG_AMT_out": charge_out,
            "Median_Income": rng.uniform(40_000, 90_000, size=rows),
        }
    )
    frame[config.TARGET_CHARGE] = total
    frame[config.TARGET_LOG_CHARGE] = np.log(total)
    frame[config.TARGET_TIER] = pd.cut(
        total, bins=config.TIER_BINS, labels=config.TIER_LABELS
    ).astype(int)
    return frame


@pytest.fixture
def synthetic_income() -> pd.DataFrame:
    """Build an income lookup covering every state code.

    Returns:
        A frame with one row per state code.
    """
    codes = list(range(1, 54))
    return pd.DataFrame(
        {
            "PRVDR_STATE_CD": codes,
            "STATE": [f"State {code}" for code in codes],
            "Median_Income": [50_000.0 + code * 100 for code in codes],
        }
    )
