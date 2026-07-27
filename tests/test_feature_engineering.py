"""Leakage regression tests.

These are the most important tests in the suite. Both failures they guard
against are silent: the pipeline still runs, the metrics still print, they are
just wrong in a flattering direction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.healthcare_mlops import config
from src.healthcare_mlops import feature_engineering as features


def test_split_shares_no_beneficiary(synthetic_claims):
    train, test = features.split_by_beneficiary(synthetic_claims)
    assert not set(train[config.GROUP_KEY]) & set(test[config.GROUP_KEY])


def test_split_keeps_every_row(synthetic_claims):
    train, test = features.split_by_beneficiary(synthetic_claims)
    assert len(train) + len(test) == len(synthetic_claims)


def test_naive_row_split_would_leak(synthetic_claims):
    """Document why the grouped split exists, not just that it works."""
    train, test = train_test_split(synthetic_claims, test_size=0.3, random_state=0)
    assert set(train[config.GROUP_KEY]) & set(test[config.GROUP_KEY])


def _leaked_categories(fit_on, train, test) -> set[str]:
    """Count categories the encoder learned that exist only in the test fold.

    Args:
        fit_on: The frame the preprocessor is fitted on.
        train: The training fold.
        test: The held-out fold.

    Returns:
        The set of leaked category values across both diagnosis columns.
    """
    preprocessor = features.build_preprocessor().fit(features.feature_frame(fit_on))
    encoder = preprocessor.named_transformers_["diagnosis_freq"]
    leaked: set[str] = set()
    for column in config.FREQUENCY_ENCODED_COLUMNS:
        test_only = set(test[column]) - set(train[column])
        leaked |= set(encoder.frequency_maps_[column]) & test_only
    return leaked


def test_encoder_never_sees_test_only_categories(synthetic_claims):
    """The core no-leakage assertion: fit on train must not know test values."""
    train, test = features.split_by_beneficiary(synthetic_claims)
    assert _leaked_categories(train, train, test) == set()


def test_leakage_detector_is_sensitive(synthetic_claims):
    """Prove the assertion above can fail, by leaking on purpose.

    Fitting the encoder on the full frame before splitting is exactly what the
    legacy pipeline did. If this test ever stops finding leakage, the check
    above has gone blind and is no longer guarding anything.
    """
    train, test = features.split_by_beneficiary(synthetic_claims)
    test_only = set()
    for column in config.FREQUENCY_ENCODED_COLUMNS:
        test_only |= set(test[column]) - set(train[column])
    if not test_only:
        pytest.skip("fixture produced no test-only categories to leak")

    assert _leaked_categories(synthetic_claims, train, test) == test_only


def test_unseen_category_maps_to_training_minimum(synthetic_claims):
    """An unseen code must transform, not raise -- inference sees novel codes.

    The expected value is computed from the training data independently of the
    encoder, so this cannot pass by comparing the encoder against itself.
    """
    train, test = features.split_by_beneficiary(synthetic_claims)
    preprocessor = features.build_preprocessor().fit(features.feature_frame(train))

    expected = train["PRNCPAL_DGNS_CD_inp"].value_counts(normalize=True).min()

    probe = features.feature_frame(test).copy()
    probe.iloc[0, probe.columns.get_loc("PRNCPAL_DGNS_CD_inp")] = "ZZZ999"
    transformed = preprocessor.transform(probe)

    assert not np.isnan(transformed).any()
    assert transformed[0, 0] == pytest.approx(expected)
    # Zero would collide with a genuinely absent value in distance-based models.
    assert transformed[0, 0] > 0


def test_unseen_poa_value_encodes_as_zeros(synthetic_claims):
    """The legacy encoder had no behaviour at all for an unseen POA value."""
    train, _ = features.split_by_beneficiary(synthetic_claims)
    preprocessor = features.build_preprocessor().fit(features.feature_frame(train))

    probe = features.feature_frame(train).head(1).copy()
    probe.iloc[0, probe.columns.get_loc("CLM_E_POA_IND_SW1")] = "UNSEEN"
    transformed = preprocessor.transform(probe)

    names = list(preprocessor.get_feature_names_out())
    poa_indices = [i for i, name in enumerate(names) if "CLM_E_POA_IND_SW1" in name]
    assert poa_indices
    assert transformed[0, poa_indices].sum() == 0


def test_poa_encoding_produces_both_levels(synthetic_claims):
    """Legacy produced only a _Y column; both levels should now be present."""
    train, _ = features.split_by_beneficiary(synthetic_claims)
    preprocessor = features.build_preprocessor().fit(features.feature_frame(train))
    names = " ".join(preprocessor.get_feature_names_out())
    assert "CLM_E_POA_IND_SW1_Y" in names
    assert "CLM_E_POA_IND_SW1_U" in names


def test_frequency_encoder_is_deterministic(synthetic_claims):
    frame = features.feature_frame(synthetic_claims)
    first = features.FrequencyEncoder().fit(frame[config.FREQUENCY_ENCODED_COLUMNS])
    second = features.FrequencyEncoder().fit(frame[config.FREQUENCY_ENCODED_COLUMNS])
    assert first.frequency_maps_ == second.frequency_maps_


def test_frequencies_come_only_from_training_rows():
    """A category common in test but rare in train must encode as rare."""
    train = pd.DataFrame(
        {
            "PRNCPAL_DGNS_CD_inp": ["A00"] * 99 + ["B00"],
            "PRNCPAL_DGNS_CD_out": ["C00"] * 100,
        }
    )
    encoder = features.FrequencyEncoder().fit(train)
    assert encoder.frequency_maps_["PRNCPAL_DGNS_CD_inp"]["B00"] == 0.01

    test = pd.DataFrame(
        {"PRNCPAL_DGNS_CD_inp": ["B00"] * 100, "PRNCPAL_DGNS_CD_out": ["C00"] * 100}
    )
    assert encoder.transform(test)[0, 0] == 0.01
