"""
forecast_pipeline.py — Train the NEXYRA energy forecast model
==============================================================

Run independently from the main training pipeline.
Does NOT modify or overwrite any existing artifacts.

Usage:
    python forecast_pipeline.py
"""

import json
import logging
import os

from data.loader import load_and_prepare_data
from ml.artifacts import get_artifact_path
from ml.forecast import ARTIFACT_NAME, train_forecast_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def run_forecast_pipeline() -> None:
    logging.info("Loading dataset...")
    train_df, _ = load_and_prepare_data("development_dataset.csv")
    logging.info("Train rows: %d", len(train_df))

    save_path = get_artifact_path(ARTIFACT_NAME)
    logging.info("Training energy forecast model -> %s", save_path)

    metrics = train_forecast_model(train_df, save_path)

    # Save forecast metrics separately
    metrics_path = get_artifact_path("forecast_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logging.info("Forecast metrics saved to %s", metrics_path)
    logging.info("Forecast pipeline complete.")
    for eq, m in metrics.items():
        logging.info("  %s: %s", eq, m)


if __name__ == "__main__":
    run_forecast_pipeline()
