"""Tests for model wiring, the selection rule, and SHAP explainer dispatch."""

from __future__ import annotations

import numpy as np
import pytest

from src.healthcare_mlops import config, evaluate, explain, models
from src.healthcare_mlops import feature_engineering as features


def test_all_four_classifiers_are_defined():
    assert set(models.classifier_candidates()) == {
        "knn",
        "decision_tree",
        "random_forest",
        "xgboost",
    }


def test_label_offset_round_trips():
    original = np.array([1, 2, 3, 4, 5])
    assert np.array_equal(models.to_tier_labels(models.to_zero_based(original)), original)


def test_sample_weights_favour_rare_classes():
    labels = np.array([0] * 90 + [4] * 10)
    weights = models.sample_weights(labels)
    assert weights[labels == 4].mean() > weights[labels == 0].mean()
    assert weights.mean() == pytest.approx(1.0)


def test_selection_rule_prefers_macro_f1_over_accuracy():
    """A model that ignores rare tiers must not win on accuracy alone."""
    results = {
        "accurate_but_blind": {"accuracy": 0.80, "macro_f1": 0.40},
        "balanced": {"accuracy": 0.75, "macro_f1": 0.70},
    }
    for metrics in results.values():
        metrics["selection_score"] = evaluate.selection_score(metrics)
    assert evaluate.select_best(results) == "balanced"


def test_regression_metrics_report_dollar_error():
    truth = np.log(np.array([1000.0, 10_000.0]))
    metrics = evaluate.regression_metrics(truth, truth)
    assert metrics["rmse_log"] == pytest.approx(0.0)
    assert metrics["median_abs_dollar_error"] == pytest.approx(0.0)


def test_confusion_frame_is_square_over_all_tiers():
    frame = evaluate.confusion_frame([1, 2, 3], [1, 2, 3], config.TIER_LABELS)
    assert frame.shape == (len(config.TIER_LABELS), len(config.TIER_LABELS))


def _fit(name, frame):
    pipeline = models.classifier_candidates()[name]
    pipeline.fit(
        features.feature_frame(frame),
        models.to_zero_based(frame[config.TARGET_TIER]),
    )
    return pipeline


def test_tree_models_use_the_fast_explainer(synthetic_claims):
    assert explain.is_tree_model(_fit("random_forest", synthetic_claims))


def test_knn_falls_back_to_model_agnostic_explainer(synthetic_claims):
    """KNN is a live selection candidate, and TreeExplainer would crash on it."""
    pipeline = _fit("knn", synthetic_claims)
    assert not explain.is_tree_model(pipeline)

    background = features.feature_frame(synthetic_claims).head(20)
    explainer = explain.PredictionExplainer(pipeline, background)
    assert not explainer.uses_tree_explainer

    contributions = explainer.top_contributions(background.head(1), predicted_index=0)
    assert contributions


def test_non_tree_model_without_background_raises(synthetic_claims):
    pipeline = _fit("knn", synthetic_claims)
    with pytest.raises(ValueError, match="background"):
        explain.PredictionExplainer(pipeline, None)


def test_tree_explanations_are_ranked_by_magnitude(synthetic_claims):
    pipeline = _fit("random_forest", synthetic_claims)
    explainer = explain.PredictionExplainer(pipeline)
    contributions = explainer.top_contributions(
        features.feature_frame(synthetic_claims).head(1), predicted_index=0
    )
    magnitudes = [abs(item["contribution"]) for item in contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)
