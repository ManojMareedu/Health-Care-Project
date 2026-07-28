"""ZenML training pipeline with MLflow tracking.

Run it with::

    python -m src.healthcare_mlops.train_pipeline

The pipeline is deliberately thin: each step delegates to the plain functions in
the sibling modules, so the whole thing is testable and debuggable without an
orchestrator attached. ZenML supplies step boundaries, caching, and lineage;
MLflow records params, metrics, and artifacts to a local SQLite file, which needs
no server and costs nothing.

Setting ``DAGSHUB_MLFLOW_URI`` (plus the usual ``MLFLOW_TRACKING_USERNAME`` /
``MLFLOW_TRACKING_PASSWORD``) points tracking at a hosted DagsHub project
instead. It is strictly opt-in -- unset, everything stays local and offline.
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from typing import Annotated

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from zenml import pipeline, step  # noqa: E402

from . import config, evaluate, models  # noqa: E402
from . import data_ingestion as ingestion  # noqa: E402
from . import data_validation as validation  # noqa: E402
from . import feature_engineering as features  # noqa: E402

logger = logging.getLogger(__name__)

# ZenML and MLflow both emit a lot of deprecation noise that drowns the metrics
# we actually want to read off the console.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")


def _tracking_uri() -> str:
    """Resolve the MLflow tracking URI, preferring an opt-in hosted server.

    Returns:
        A DagsHub URI when ``DAGSHUB_MLFLOW_URI`` is set, else local SQLite.
    """
    return os.environ.get("DAGSHUB_MLFLOW_URI", config.MLFLOW_TRACKING_URI)


@step
def ingest_and_validate() -> Annotated[pd.DataFrame, "modeling_frame"]:
    """Load both sources, validate them, and derive the modeling frame.

    Returns:
        The validated modeling frame.
    """
    claims = validation.validate_claims(ingestion.load_claims())
    income = validation.validate_income(ingestion.load_income())

    contamination = validation.report_procedure_codes_in_diagnosis(claims)
    for column, codes in contamination.items():
        if codes:
            logger.warning("Procedure codes found in diagnosis column %s: %s", column, codes)

    frame = ingestion.add_targets(ingestion.merge_income(claims, income))
    frame = validation.validate_modeling_frame(frame)
    logger.info(
        "Validated modeling frame: %s rows, %s beneficiaries",
        len(frame),
        frame[config.GROUP_KEY].nunique(),
    )
    return frame


@step
def split_data(
    frame: pd.DataFrame,
) -> tuple[Annotated[pd.DataFrame, "train"], Annotated[pd.DataFrame, "test"]]:
    """Split by beneficiary so no ``BENE_ID`` spans both folds.

    Args:
        frame: Validated modeling frame.

    Returns:
        ``(train, test)`` frames.
    """
    train, test = features.split_by_beneficiary(frame)
    overlap = set(train[config.GROUP_KEY]) & set(test[config.GROUP_KEY])
    if overlap:
        raise ValueError(f"Grouped split leaked {len(overlap)} beneficiaries across folds")
    logger.info("Split: %s train / %s test rows, 0 shared beneficiaries", len(train), len(test))
    return train, test


def _log_confusion_plot(frame: pd.DataFrame, name: str) -> None:
    """Log a confusion matrix heatmap to the active MLflow run.

    Args:
        frame: Labelled confusion matrix.
        name: Model name, used in the artifact filename.
    """
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.imshow(frame.to_numpy(), cmap="Blues")
    axis.set_xticks(range(len(frame.columns)), frame.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(frame.index)), frame.index)
    for row in range(frame.shape[0]):
        for col in range(frame.shape[1]):
            axis.text(col, row, frame.iat[row, col], ha="center", va="center", fontsize=8)
    axis.set_title(f"Confusion matrix - {name}")
    fig.tight_layout()
    mlflow.log_figure(fig, f"confusion_matrix_{name}.png")
    plt.close(fig)


def _log_importance_plot(pipeline_obj: Pipeline, name: str) -> None:
    """Log a feature importance bar chart when the estimator exposes one.

    Args:
        pipeline_obj: Fitted pipeline.
        name: Model name, used in the artifact filename.
    """
    estimator = pipeline_obj.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        return
    names = pipeline_obj.named_steps["preprocessor"].get_feature_names_out()
    importances = pd.Series(estimator.feature_importances_, index=names).sort_values()
    fig, axis = plt.subplots(figsize=(7, 4))
    importances.plot.barh(ax=axis)
    axis.set_title(f"Feature importance - {name}")
    fig.tight_layout()
    mlflow.log_figure(fig, f"feature_importance_{name}.png")
    plt.close(fig)


@step
def tune_classifiers(train: pd.DataFrame) -> Annotated[dict, "tuned_params"]:
    """Search hyperparameters for the two strongest classifiers.

    Only the tree ensembles are tuned. KNN and a single decision tree are in the
    comparison as baselines, and tuning them would spend compute without changing
    which model ships.

    Args:
        train: Training fold. The search never sees the held-out fold.

    Returns:
        Mapping of model name to its best parameters and cross-validated score.
    """
    mlflow.set_tracking_uri(_tracking_uri())
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    x_train = features.feature_frame(train)
    y_train = models.to_zero_based(train[config.TARGET_TIER])
    groups = train[config.GROUP_KEY]

    tuned: dict[str, dict] = {}
    for name in models.CLASSIFIER_SEARCH_SPACES:
        with mlflow.start_run(run_name=f"tune-{name}"):
            _, best_params, best_score = models.tune_classifier(name, x_train, y_train, groups)
            mlflow.log_param("model", name)
            mlflow.log_param("task", "hyperparameter_search")
            mlflow.log_param("cv", "GroupKFold(3) on BENE_ID")
            mlflow.log_param("scoring", "f1_macro")
            mlflow.log_params({k: v for k, v in best_params.items()})
            mlflow.log_metric("cv_macro_f1", best_score)
            tuned[name] = {"best_params": best_params, "cv_macro_f1": best_score}
            logger.info("tuned %s -> cv_macro_f1=%.4f %s", name, best_score, best_params)
    return tuned


@step
def train_classifiers(
    train: pd.DataFrame, test: pd.DataFrame, tuned: dict
) -> Annotated[dict, "classifier_results"]:
    """Train and evaluate all four tier classifiers, logging each to MLflow.

    Args:
        train: Training fold.
        test: Held-out fold.
        tuned: Best parameters from ``tune_classifiers``, applied where present.

    Returns:
        Mapping of model name to its test metrics.
    """
    mlflow.set_tracking_uri(_tracking_uri())
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    x_train = features.feature_frame(train)
    x_test = features.feature_frame(test)
    y_train = models.to_zero_based(train[config.TARGET_TIER])
    y_test = models.to_zero_based(test[config.TARGET_TIER])
    weights = models.sample_weights(y_train)

    results: dict[str, dict[str, float]] = {}
    for name, candidate in models.classifier_candidates().items():
        if name in tuned:
            candidate.set_params(**tuned[name]["best_params"])
        with mlflow.start_run(run_name=f"classifier-{name}"):
            if name == "xgboost":
                candidate.fit(x_train, y_train, model__sample_weight=weights)
            else:
                candidate.fit(x_train, y_train)

            predictions = candidate.predict(x_test)
            probabilities = (
                candidate.predict_proba(x_test) if hasattr(candidate, "predict_proba") else None
            )
            metrics = evaluate.classification_metrics(y_test, predictions, probabilities)

            mlflow.log_param("model", name)
            mlflow.log_param("task", "tier_classification")
            mlflow.log_param("split", "GroupShuffleSplit on BENE_ID")
            mlflow.log_param("selection_rule", "0.6*macro_f1 + 0.4*accuracy")
            mlflow.log_params(
                {
                    f"hp_{k}": v
                    for k, v in candidate.named_steps["model"].get_params().items()
                    if isinstance(v, (int, float, str, bool)) or v is None
                }
            )
            mlflow.log_metrics(metrics)

            matrix = evaluate.confusion_frame(
                models.to_tier_labels(y_test),
                models.to_tier_labels(predictions),
                config.TIER_LABELS,
            )
            mlflow.log_text(matrix.to_string(), f"confusion_matrix_{name}.txt")
            _log_confusion_plot(matrix, name)
            _log_importance_plot(candidate, name)
            mlflow.sklearn.log_model(candidate, artifact_path="pipeline")

            results[name] = metrics
            logger.info("%s -> %s", name, {k: round(v, 4) for k, v in metrics.items()})
    return results


@step
def train_regressors(
    train: pd.DataFrame, test: pd.DataFrame
) -> Annotated[dict, "regressor_results"]:
    """Train and evaluate the log-charge regressors, logging each to MLflow.

    Args:
        train: Training fold.
        test: Held-out fold.

    Returns:
        Mapping of model name to its test metrics.
    """
    mlflow.set_tracking_uri(_tracking_uri())
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    x_train = features.feature_frame(train)
    x_test = features.feature_frame(test)
    y_train = train[config.TARGET_LOG_CHARGE]
    y_test = test[config.TARGET_LOG_CHARGE]

    results: dict[str, dict[str, float]] = {}
    for name, candidate in models.regressor_candidates().items():
        with mlflow.start_run(run_name=f"regressor-{name}"):
            candidate.fit(x_train, y_train)
            metrics = evaluate.regression_metrics(y_test, candidate.predict(x_test))

            mlflow.log_param("model", name)
            mlflow.log_param("task", "log_charge_regression")
            mlflow.log_param("split", "GroupShuffleSplit on BENE_ID")
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(candidate, artifact_path="pipeline")

            results[name] = metrics
            logger.info("%s -> %s", name, {k: round(v, 4) for k, v in metrics.items()})
    return results


@step
def export_best_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    classifier_results: dict,
    regressor_results: dict,
    tuned: dict,
) -> Annotated[dict, "export_metadata"]:
    """Refit and export the winning models so the demo runs without retraining.

    The classifier is chosen by the documented weighted rule; the regressor by
    lowest log-space RMSE. Both are written to ``exported_model/`` in MLflow
    pyfunc format alongside a metadata file recording which model won and why.

    Args:
        train: Training fold.
        test: Held-out fold.
        classifier_results: Metrics from ``train_classifiers``.
        regressor_results: Metrics from ``train_regressors``.
        tuned: Best parameters from ``tune_classifiers``, reapplied before the
            final refit so the exported model matches the evaluated one.

    Returns:
        The metadata written to disk.
    """
    import shutil

    best_classifier = evaluate.select_best(classifier_results)
    best_regressor = min(regressor_results, key=lambda n: regressor_results[n]["rmse_log"])

    x_train = features.feature_frame(train)
    y_train_tier = models.to_zero_based(train[config.TARGET_TIER])

    classifier = models.classifier_candidates()[best_classifier]
    if best_classifier in tuned:
        classifier.set_params(**tuned[best_classifier]["best_params"])
    if best_classifier == "xgboost":
        classifier.fit(
            x_train, y_train_tier, model__sample_weight=models.sample_weights(y_train_tier)
        )
    else:
        classifier.fit(x_train, y_train_tier)

    regressor = models.regressor_candidates()[best_regressor]
    regressor.fit(x_train, train[config.TARGET_LOG_CHARGE])

    config.EXPORTED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for directory in (config.CLASSIFIER_DIR, config.REGRESSOR_DIR):
        if directory.exists():
            shutil.rmtree(directory)
    mlflow.sklearn.save_model(classifier, str(config.CLASSIFIER_DIR))
    mlflow.sklearn.save_model(regressor, str(config.REGRESSOR_DIR))

    # SHAP needs a background distribution, and the model-agnostic explainer
    # needs one even to start. The raw data is DVC-tracked and absent from the
    # serving container, so a small representative sample ships with the model.
    x_train.sample(
        n=min(config.SHAP_BACKGROUND_ROWS, len(x_train)),
        random_state=config.RANDOM_STATE,
    ).to_csv(config.SHAP_BACKGROUND_FILE, index=False)

    metadata = {
        "classifier": {
            "name": best_classifier,
            "selection_rule": "0.6*macro_f1 + 0.4*accuracy",
            "metrics": classifier_results[best_classifier],
            "all_candidates": classifier_results,
        },
        "regressor": {
            "name": best_regressor,
            "selection_rule": "lowest rmse_log",
            "metrics": regressor_results[best_regressor],
            "all_candidates": regressor_results,
        },
        "tuning": tuned,
        "tier_labels": config.TIER_LABELS,
        "tier_descriptions": {str(k): v for k, v in config.TIER_DESCRIPTIONS.items()},
        "feature_columns": config.FEATURE_COLUMNS,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_beneficiaries": int(train[config.GROUP_KEY].nunique()),
        "test_beneficiaries": int(test[config.GROUP_KEY].nunique()),
    }
    config.METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Exported classifier=%s regressor=%s", best_classifier, best_regressor)
    return metadata


# Caching is off deliberately. ZenML keys a step's cache on its inputs, so
# changing only the *body* of a step -- adding a model candidate, say -- replays
# the stale result and silently exports metrics for models that never ran. A
# training pipeline that reports numbers must re-execute.
@pipeline(name="healthcare_claims_training", enable_cache=False)
def training_pipeline() -> None:
    """Wire the steps into the end-to-end training run."""
    frame = ingest_and_validate()
    train, test = split_data(frame)
    tuned = tune_classifiers(train)
    classifier_results = train_classifiers(train, test, tuned)
    regressor_results = train_regressors(train, test)
    export_best_models(train, test, classifier_results, regressor_results, tuned)


def main() -> None:
    """Entry point for ``python -m src.healthcare_mlops.train_pipeline``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    training_pipeline()


if __name__ == "__main__":
    main()
