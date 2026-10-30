"""Model training module integrated with MLflow experiment tracking."""

import logging
import os
import pickle
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline

from src.core.settings import settings
from src.pipeline.config_loader import load_params
from src.pipeline.model_pipeline import build_model_pipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "2")


class ModelTrainer:
    """Class responsible for training the classification model with MLflow tracking."""

    def __init__(self, tracking_uri: str, experiment_name: str) -> None:
        """Initialize trainer with MLflow tracking settings."""
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

    def load_dataset(
        self, data_path: str | Path, target_col: str
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Load training data and separate features and target.

        Args:
            data_path: Path to train.csv.
            target_col: Name of target column.

        Returns:
            Tuple containing feature DataFrame and target Series.
        """
        df = pd.read_csv(data_path)
        y = df[target_col].astype(bool)
        x = df.drop(columns=[target_col])
        return x, y

    def train_and_log(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        model_params: dict[str, Any],
        output_model_path: str | Path,
    ) -> str:
        """Fit pipeline and log parameters, artifacts, and model to MLflow.

        Args:
            x_train: Training features DataFrame.
            y_train: Training target Series.
            model_params: Dictionary with hyperparameters.
            output_model_path: Local filesystem destination for model artifact.

        Returns:
            Active MLflow run ID.
        """
        pipeline = build_model_pipeline(**model_params)
        pipeline.fit(x_train, y_train)
        self.save_local_artifact(pipeline, output_model_path)

        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            with mlflow.start_run() as run:
                run_id = run.info.run_id
                logger.info("Started MLflow run %s", run_id)
                mlflow.log_params(model_params)
                mlflow.sklearn.log_model(
                    sk_model=pipeline,
                    artifact_path="model",
                    input_example=x_train.head(2),
                    serialization_format="pickle",
                )
                logger.info(
                    "Successfully trained and logged model to MLflow (run_id: %s)",
                    run_id,
                )
                return run_id
        except Exception as e:
            logger.warning("MLflow server not reached (%s). Model saved locally.", e)
            return "local_run"

    def save_local_artifact(self, model: Pipeline, output_path: str | Path) -> None:
        """Save model pipeline to local disk via pickle.

        Args:
            model: Fitted pipeline.
            output_path: File path to save model.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(model, f)
        logger.info("Saved local model artifact to %s", path)


def main() -> None:
    """Entry point for model training stage."""
    params = load_params()
    seed = int(params.get("seed", 42))
    train_cfg = params.get("train", {})

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", settings.MLFLOW_TRACKING_URI)
    experiment_name = str(
        train_cfg.get("experiment_name", "purchase-intention-challenge")
    )

    trainer = ModelTrainer(tracking_uri=tracking_uri, experiment_name=experiment_name)
    train_file = Path(train_cfg.get("processed_dir", "data/processed")) / "train.csv"
    x_train, y_train = trainer.load_dataset(train_file, target_col="Revenue")

    model_params = {
        "seed": seed,
        "n_estimators": int(train_cfg.get("n_estimators", 150)),
        "max_depth": int(train_cfg.get("max_depth", 8)),
        "min_samples_split": int(train_cfg.get("min_samples_split", 5)),
    }
    output_model = Path(train_cfg.get("model_dir", "models")) / "model.pkl"
    trainer.train_and_log(x_train, y_train, model_params, output_model)


if __name__ == "__main__":
    main()
