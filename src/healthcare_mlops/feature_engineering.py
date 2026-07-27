"""Leakage-free splitting and encoding.

This module carries the two most consequential corrections to the legacy
pipeline. Both were silent accuracy inflators, which is the dangerous kind of
bug: the model looked better than it was.

Correction 1 -- split by beneficiary, not by row
------------------------------------------------
``Patient_Claim_Data`` holds 46,059 rows but only 5,416 distinct ``BENE_ID``
values: roughly 8.5 rows per beneficiary, one per outpatient diagnosis. Those
rows share a beneficiary's inpatient charge, claim counts, and state. A plain
random 70/30 row split therefore places nearly every beneficiary on *both* sides
of the split, so the test set is largely a paraphrase of the training set. This
module splits on unique ``BENE_ID`` with ``GroupShuffleSplit`` instead.

Correction 2 -- fit encoders on the training fold only
-------------------------------------------------------
The legacy script computed diagnosis-code frequencies and POA dummies over the
entire dataset before splitting, so test-set distribution leaked into the
features every model trained on. Here both encoders live inside a scikit-learn
``ColumnTransformer`` that is fit on the training fold and merely applied to
test and inference data.

Unseen categories are a real inference-time concern once encoders are fit on a
subset, so both encoders handle them explicitly rather than crashing: unseen
diagnosis codes take the training-set minimum frequency (the rarest thing we
saw, which is what an unseen code is), and unseen POA values encode as all-zero
via ``handle_unknown="ignore"``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encode high-cardinality categories by their training-set frequency.

    Diagnosis codes have 174 (inpatient) and 211 (outpatient) distinct values.
    One-hot encoding them would add hundreds of near-empty columns; frequency
    encoding keeps one informative numeric column each, on the reasoning that how
    common a diagnosis is carries more signal about cost than its identity.

    The frequency map is learned in ``fit`` and only applied in ``transform``,
    which is what keeps test-set distribution out of training features.

    Attributes:
        frequency_maps_: Per-column mapping of category to training frequency.
        unseen_values_: Per-column fallback used for categories absent from training.
    """

    def __init__(self, normalize: bool = True):
        """Initialise the encoder.

        Args:
            normalize: Encode as a proportion of training rows rather than a raw
                count, so the feature does not rescale with dataset size.
        """
        self.normalize = normalize

    def fit(self, X: pd.DataFrame, y=None) -> FrequencyEncoder:  # noqa: N803
        """Learn category frequencies from the training fold only.

        Args:
            X: Frame of categorical columns to encode.
            y: Unused; present for scikit-learn API compatibility.

        Returns:
            The fitted encoder.
        """
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.frequency_maps_ = {}
        self.unseen_values_ = {}
        for column in X.columns:
            counts = X[column].value_counts(normalize=self.normalize)
            self.frequency_maps_[column] = counts.to_dict()
            # An unseen code is, by definition, rarer than anything observed --
            # so the training minimum is the honest fallback. Never 0, which
            # would collide with "absent" in downstream distance metrics.
            self.unseen_values_[column] = float(counts.min())
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        """Apply the learned frequency maps.

        Args:
            X: Frame of categorical columns matching those seen in ``fit``.

        Returns:
            A float array with one encoded column per input column.
        """
        X = pd.DataFrame(X)
        encoded = pd.DataFrame(index=X.index)
        for column in self.feature_names_in_:
            encoded[column] = (
                X[column].map(self.frequency_maps_[column]).fillna(self.unseen_values_[column])
            )
        return encoded.to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return output column names for downstream introspection.

        Args:
            input_features: Ignored; names are taken from ``fit``.

        Returns:
            Array of ``<column>_freq`` names.
        """
        return np.asarray([f"{c}_freq" for c in self.feature_names_in_], dtype=object)


def split_by_beneficiary(
    frame: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into train/test with no beneficiary appearing in both.

    Args:
        frame: Modeling frame containing ``BENE_ID``.
        test_size: Proportion of *groups* held out.
        random_state: Seed for reproducibility.

    Returns:
        ``(train, test)`` frames sharing no ``BENE_ID``.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(frame, groups=frame[config.GROUP_KEY]))
    return frame.iloc[train_idx].copy(), frame.iloc[test_idx].copy()


def build_preprocessor() -> ColumnTransformer:
    """Assemble the preprocessing stage as a single fittable object.

    Bundling preprocessing with the estimator into one ``Pipeline`` means the
    serving path loads one artifact and cannot drift from training -- there is no
    second copy of the encoding logic to keep in sync.

    Returns:
        An unfitted ``ColumnTransformer`` covering all feature columns.
    """
    return ColumnTransformer(
        transformers=[
            ("diagnosis_freq", FrequencyEncoder(), config.FREQUENCY_ENCODED_COLUMNS),
            (
                "poa_onehot",
                # handle_unknown="ignore" is the fix for the legacy encoder, which
                # produced only a _Y column and had no behaviour at all for a value
                # it had not seen.
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                config.CATEGORICAL_COLUMNS,
            ),
            ("numeric", StandardScaler(), config.NUMERIC_COLUMNS),
        ],
        remainder="drop",
    )


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Select just the model input columns.

    Args:
        frame: Any frame carrying the feature columns.

    Returns:
        A frame limited to ``config.FEATURE_COLUMNS``.
    """
    return frame[config.FEATURE_COLUMNS]
