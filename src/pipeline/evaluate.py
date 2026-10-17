"""Model evaluation module for computing performance metrics and saving reports."""

import json
import logging
import os
import pickle
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.core.settings import settings
from src.pipeline.config_loader import load_params

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "2")


class ModelEvaluator:
    """Class responsible for evaluating model performance and saving metrics."""

    def __init__(self, tracking_uri: str, experiment_name: str) -> None:
        """Initialize evaluator with MLflow tracking settings."""
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

    def load_model(self, model_path: str | Path) -> Pipeline:
        """Load trained pipeline from local pickle file.

        Args:
            model_path: Path to serialized model.

        Returns:
            Fitted scikit-learn Pipeline.
        """
        path = Path(model_path)
        with path.open("rb") as f:
            model: Pipeline = pickle.load(f)
        return model

    def load_test_data(
        self, data_path: str | Path, target_col: str
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Load test dataset and separate features and target.

        Args:
            data_path: Path to test.csv.
            target_col: Target column name.

        Returns:
            Tuple with features and target.
        """
        df = pd.read_csv(data_path)
        y = df[target_col].astype(bool)
        x = df.drop(columns=[target_col])
        return x, y

    def calculate_metrics(
        self, model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series
    ) -> dict[str, float]:
        """Calculate classification metrics.

        Args:
            model: Trained pipeline.
            x_test: Test features.
            y_test: True labels.

        Returns:
            Dictionary with evaluation metrics.
        """
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
        }
        logger.info("Evaluation metrics: %s", metrics)
        return metrics

    def save_metrics(self, metrics: dict[str, float], output_path: str | Path) -> None:
        """Persist metrics to JSON file for DVC tracking.

        Args:
            metrics: Computed metrics dictionary.
            output_path: Path to write json file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved metrics to %s", path)

    def log_metrics_to_mlflow(self, metrics: dict[str, float]) -> None:
        """Log calculated metrics to active or new MLflow run."""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            with mlflow.start_run(nested=True):
                mlflow.log_metrics(metrics)
        except Exception as e:
            logger.warning("Could not log metrics to MLflow: %s", e)


def main() -> None:
    """Entry point for model evaluation stage."""
    params = load_params()
    eval_cfg = params.get("evaluate", {})
    train_cfg = params.get("train", {})

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", settings.MLFLOW_TRACKING_URI)
    experiment_name = str(
        train_cfg.get("experiment_name", "purchase-intention-challenge")
    )

    evaluator = ModelEvaluator(
        tracking_uri=tracking_uri, experiment_name=experiment_name
    )
    model = evaluator.load_model(eval_cfg.get("model_path", "models/model.pkl"))
    x_test, y_test = evaluator.load_test_data(
        eval_cfg.get("test_data_path", "data/processed/test.csv"),
        target_col="Revenue",
    )

    metrics = evaluator.calculate_metrics(model, x_test, y_test)
    evaluator.save_metrics(
        metrics, eval_cfg.get("metrics_path", "reports/metrics.json")
    )
    evaluator.log_metrics_to_mlflow(metrics)


if __name__ == "__main__":
    main()
