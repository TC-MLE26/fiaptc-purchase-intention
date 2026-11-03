"""Model registration module for promoting the champion model in MLflow."""

import logging
import os
import sys
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from src.core.settings import settings
from src.pipeline.config_loader import load_params
from src.pipeline.console import configure_utf8_console
from src.pipeline.run_id import LOCAL_RUN_ID, read_run_id, run_id_path

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

    def ensure_reachable(self) -> None:
        """Probe the MLflow server with a cheap read.

        Raises:
            Exception: Whatever the client raises when the server cannot be used.
        """
        self.client.search_experiments(max_results=1)

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
        self._assign_alias(model_name, model_alias, version_str)
        return version_str

    def _assign_alias(self, model_name: str, model_alias: str, version: str) -> None:
        """Point an alias of a registered model to a version and log it.

        Args:
            model_name: Registered model name.
            model_alias: Alias to assign (e.g. champion).
            version: Model version that receives the alias.
        """
        self.client.set_registered_model_alias(
            name=model_name, alias=model_alias, version=version
        )
        logger.info(
            "Assigned alias '@%s' to version %s of '%s'",
            model_alias,
            version,
            model_name,
        )


def require_train_run_id(run_id_file: Path) -> str:
    """Return the run ID of the train stage, which holds the model.

    Args:
        run_id_file: File written by the train stage.

    Returns:
        MLflow run ID.

    Raises:
        ValueError: If the train stage did not log a run to MLflow.
    """
    run_id = read_run_id(run_id_file)
    if run_id is None or run_id == LOCAL_RUN_ID:
        raise ValueError(
            f"The train stage did not log a run to MLflow ({run_id_file}). "
            "Re-run the pipeline with MLflow up: dvc repro --force"
        )
    return run_id


def run_registration(
    registrar: ModelRegistrar, run_id_file: Path, model_name: str, model_alias: str
) -> int:
    """Register and promote the train run, telling offline from real failures.

    Args:
        registrar: Registrar bound to the MLflow server.
        run_id_file: File written by the train stage.
        model_name: Target registered model name.
        model_alias: Alias to assign (e.g. champion).

    Returns:
        Process exit code: 0 when registered or when MLflow is unreachable
        (offline runs are allowed), 1 when MLflow is reachable but the
        registration failed.
    """
    try:
        registrar.ensure_reachable()
    except Exception as e:
        logger.warning(
            "MLflow unreachable at %s (%s). Model saved at models/model.pkl; "
            "registration skipped. Start MLflow ('mlflow ui') and re-run "
            "'dvc repro --force' to register it.",
            registrar.tracking_uri,
            e,
        )
        return 0
    try:
        run_id = require_train_run_id(run_id_file)
        registrar.register_and_promote(run_id, model_name, model_alias)
    except Exception as e:
        logger.error("Registration FAILED with MLflow reachable: %s", e)
        return 1
    return 0


def main() -> None:
    """Entry point for model registration stage."""
    configure_utf8_console()
    train_cfg = load_params().get("train", {})
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", settings.MLFLOW_TRACKING_URI)
    exit_code = run_registration(
        ModelRegistrar(tracking_uri=tracking_uri),
        run_id_path(train_cfg),
        model_name=str(train_cfg.get("model_name", "purchase-intention-classifier")),
        model_alias=str(train_cfg.get("model_alias", "champion")),
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
