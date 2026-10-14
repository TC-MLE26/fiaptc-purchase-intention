"""Load the promoted model from the MLflow Model Registry."""

import logging
import os
from dataclasses import dataclass
from typing import Any

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

logger = logging.getLogger(__name__)

HTTP_MAX_RETRIES = "1"
HTTP_TIMEOUT_SECONDS = "10"


@dataclass(frozen=True)
class RegisteredModel:
    """A model loaded from the registry with its identification."""

    estimator: Any
    name: str
    version: str


def configure_mlflow_http() -> None:
    """Fail fast when the tracking server is down, unless already configured."""
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", HTTP_MAX_RETRIES)
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", HTTP_TIMEOUT_SECONDS)


def load_registered_model(
    model_name: str, model_alias: str, tracking_uri: str
) -> RegisteredModel:
    """Resolve the version behind an alias and load exactly that version."""
    client = MlflowClient(tracking_uri=tracking_uri)
    version = str(client.get_model_version_by_alias(model_name, model_alias).version)

    mlflow.set_tracking_uri(tracking_uri)
    estimator = mlflow.sklearn.load_model(f"models:/{model_name}/{version}")

    logger.info("Model loaded: %s v%s", model_name, version)
    return RegisteredModel(estimator, model_name, version)
