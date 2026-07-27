"""Evidently data and prediction drift reporting.

A model trained on one claims extract degrades quietly when the incoming
distribution moves -- new diagnosis codes come into use, coding practice
changes, a payer mix shifts. Accuracy metrics will not tell you until labels
arrive months later, so this compares the live feature distribution against the
training reference and flags the drift directly.

Two reports are produced:

* **Feature drift** -- reference training features against a current batch.
* **Prediction drift** -- the tier distribution the model produced on each,
  which catches the case where inputs look stable but the model's behaviour has
  moved anyway.

Run it with::

    python -m monitoring.drift_report

With no arguments it demonstrates the check by comparing the training fold
against the held-out fold. Point ``--current`` at a CSV or Parquet file of real
incoming claims to run it for real.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mlflow
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from src.healthcare_mlops import config, models
from src.healthcare_mlops import data_ingestion as ingestion
from src.healthcare_mlops import feature_engineering as features

logger = logging.getLogger(__name__)

REPORTS_DIR = config.PROJECT_ROOT / "monitoring" / "reports"

PREDICTION_COLUMN = "predicted_tier"


def _data_definition(frame: pd.DataFrame) -> DataDefinition:
    """Describe column roles so Evidently picks the right statistical test.

    Categorical and numeric columns get different drift tests, and getting the
    split wrong produces confident nonsense -- so the roles are declared rather
    than inferred.

    Args:
        frame: Frame whose columns should be described.

    Returns:
        A populated ``DataDefinition``.
    """
    categorical = [c for c in frame.columns if frame[c].dtype == object]
    numerical = [c for c in frame.columns if c not in categorical]
    return DataDefinition(
        numerical_columns=numerical,
        categorical_columns=categorical,
    )


def build_report(reference: pd.DataFrame, current: pd.DataFrame, output: Path) -> Path:
    """Run a drift report between two frames and write it as HTML.

    Args:
        reference: The baseline distribution (training data).
        current: The distribution to test against the baseline.
        output: Destination HTML path.

    Returns:
        The path written.
    """
    definition = _data_definition(reference)
    report = Report(metrics=[DataDriftPreset()])
    result = report.run(
        current_data=Dataset.from_pandas(current, data_definition=definition),
        reference_data=Dataset.from_pandas(reference, data_definition=definition),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save_html(str(output))
    return output


def summarise(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Compute a compact drift summary suitable for logging or CI gating.

    The full Evidently payload is a large nested structure aimed at the HTML
    renderer. This flattens it to the three things a caller actually decides on:
    how many columns drifted, what share that is, and the per-column scores.

    Args:
        reference: The baseline distribution.
        current: The distribution to test.

    Returns:
        Mapping with ``drifted_columns``, ``drifted_share``, and ``columns``.
    """
    definition = _data_definition(reference)
    report = Report(metrics=[DataDriftPreset()])
    result = report.run(
        current_data=Dataset.from_pandas(current, data_definition=definition),
        reference_data=Dataset.from_pandas(reference, data_definition=definition),
    )

    summary: dict = {"drifted_columns": 0, "drifted_share": 0.0, "columns": {}}
    for metric in result.dict().get("metrics", []):
        name = str(metric.get("metric_name", ""))
        value = metric.get("value")
        if name.startswith("DriftedColumnsCount") and isinstance(value, dict):
            summary["drifted_columns"] = int(value.get("count", 0))
            summary["drifted_share"] = float(value.get("share", 0.0))
        elif name.startswith("ValueDrift(column="):
            column = name.split("column=", 1)[1].split(",", 1)[0]
            summary["columns"][column] = value
    return summary


def load_current(path: Path | None) -> pd.DataFrame | None:
    """Read a batch of incoming claims from CSV or Parquet.

    Args:
        path: File to read, or None.

    Returns:
        The loaded frame, or None when no path was given.

    Raises:
        ValueError: If the file extension is not recognised.
    """
    if path is None:
        return None
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type for drift input: {path.suffix}")


def main() -> None:
    """Generate feature and prediction drift reports."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--current",
        type=Path,
        default=None,
        help="CSV/Parquet of incoming claims. Defaults to the held-out test fold.",
    )
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    arguments = parser.parse_args()

    frame = ingestion.build_modeling_frame()
    train, test = features.split_by_beneficiary(frame)
    reference = features.feature_frame(train)

    current_raw = load_current(arguments.current)
    if current_raw is None:
        logger.info("No --current supplied; comparing training fold against held-out fold")
        current = features.feature_frame(test)
    else:
        current = current_raw[config.FEATURE_COLUMNS]

    feature_path = build_report(reference, current, arguments.output_dir / "feature_drift.html")
    logger.info("Feature drift report: %s", feature_path)

    classifier = mlflow.sklearn.load_model(str(config.CLASSIFIER_DIR))
    reference_predictions = pd.DataFrame(
        {PREDICTION_COLUMN: models.to_tier_labels(classifier.predict(reference))}
    )
    current_predictions = pd.DataFrame(
        {PREDICTION_COLUMN: models.to_tier_labels(classifier.predict(current))}
    )
    prediction_path = build_report(
        reference_predictions,
        current_predictions,
        arguments.output_dir / "prediction_drift.html",
    )
    logger.info("Prediction drift report: %s", prediction_path)

    summary = summarise(reference, current)
    logger.info(
        "Drifted columns: %s of %s (%.1f%%)",
        summary["drifted_columns"],
        len(summary["columns"]),
        summary["drifted_share"] * 100,
    )
    for column, score in summary["columns"].items():
        logger.info("  %-28s %s", column, score)


if __name__ == "__main__":
    main()
