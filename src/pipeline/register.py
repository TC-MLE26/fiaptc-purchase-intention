"""Model registration module for promoting the champion model in MLflow."""

import logging
import os

import mlflow
from mlflow import MlflowClient

from src.core.settings import settings
from src.pipeline.config_loader import load_params

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "2")


class ModelRegistrar:
    """Class responsible for promoting models in MLflow Model Registry."""

    def __init__(self, tracking_uri: str) -> None:
        """Initialize MLflow Client."""
        self.tracking_uri = tracking_uri
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        mlflow.set_tracking_uri(self.tracking_uri)

    def get_latest_run_id(self, experiment_name: str) -> str:
        """Find the latest successful run ID for an experiment.

        Args:
            experiment_name: Name of the MLflow experiment.

        Returns:
            Latest run ID string.
        """
        experiment = self.client.get_experiment_by_name(experiment_name)
        if not experiment:
            raise ValueError(f"Experiment '{experiment_name}' not found.")

        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            raise ValueError(f"No runs found in experiment '{experiment_name}'.")

        run_id = runs[0].info.run_id
        logger.info("Found latest run %s in experiment %s", run_id, experiment_name)
        return run_id

    def register_and_promote(
        self,
        run_id: str,
        model_name: str,
        model_alias: str = "champion",
    ) -> str:
        """Register model version from a run and assign alias.

        Args:
            run_id: MLflow run ID containing the logged model.
            model_name: Target registered model name.
            model_alias: Alias to assign (e.g. champion).

        Returns:
            Version number of registered model.
        """
        model_uri = f"runs:/{run_id}/model"
        logger.info("Registering model from %s as '%s'", model_uri, model_name)
        model_version = mlflow.register_model(model_uri=model_uri, name=model_name)

        version_str = str(model_version.version)
        self.client.set_registered_model_alias(
            name=model_name,
            alias=model_alias,
            version=version_str,
        )
        logger.info(
            "Assigned alias '@%s' to version %s of '%s'",
            model_alias,
            version_str,
            model_name,
        )
        return version_str


def main() -> None:
    """Entry point for model registration stage."""
    params = load_params()
    train_cfg = params.get("train", {})

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", settings.MLFLOW_TRACKING_URI)
    experiment_name = str(
        train_cfg.get("experiment_name", "purchase-intention-challenge")
    )
    model_name = str(train_cfg.get("model_name", "purchase-intention-classifier"))
    model_alias = str(train_cfg.get("model_alias", "champion"))

    try:
        registrar = ModelRegistrar(tracking_uri=tracking_uri)
        run_id = registrar.get_latest_run_id(experiment_name)
        registrar.register_and_promote(
            run_id, model_name=model_name, model_alias=model_alias
        )
    except Exception as e:
        logger.warning(
            "MLflow server not reached (%s). Model saved at models/model.pkl. "
            "Start MLflow ('mlflow ui') to register in the Model Registry.",
            e,
        )


if __name__ == "__main__":
    main()
