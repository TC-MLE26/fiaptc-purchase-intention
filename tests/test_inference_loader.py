"""Tests for loading the model from the MLflow Model Registry."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.inference import model_loader


@pytest.fixture()
def mlflow_mocks(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Replace every MLflow call used by the loader."""
    estimator = object()
    client = MagicMock()
    client.get_model_version_by_alias.return_value = SimpleNamespace(version=7)
    mocks = SimpleNamespace(
        estimator=estimator,
        client=client,
        client_class=MagicMock(return_value=client),
        set_tracking_uri=MagicMock(),
        load_model=MagicMock(return_value=estimator),
    )
    monkeypatch.setattr(model_loader, "MlflowClient", mocks.client_class)
    monkeypatch.setattr(model_loader.mlflow, "set_tracking_uri", mocks.set_tracking_uri)
    monkeypatch.setattr(model_loader.mlflow.sklearn, "load_model", mocks.load_model)
    return mocks


def test_load_registered_model_loads_the_resolved_version(mlflow_mocks):
    """The alias is resolved first and that exact version is loaded."""
    registered = model_loader.load_registered_model(
        "purchase-intention-classifier", "champion", "http://mlflow:5000"
    )

    mlflow_mocks.client_class.assert_called_once_with(tracking_uri="http://mlflow:5000")
    mlflow_mocks.client.get_model_version_by_alias.assert_called_once_with(
        "purchase-intention-classifier", "champion"
    )
    mlflow_mocks.set_tracking_uri.assert_called_once_with("http://mlflow:5000")
    mlflow_mocks.load_model.assert_called_once_with(
        "models:/purchase-intention-classifier/7"
    )
    assert registered.estimator is mlflow_mocks.estimator
    assert registered.name == "purchase-intention-classifier"
    assert registered.version == "7"


def test_load_registered_model_propagates_registry_errors(mlflow_mocks):
    """Registry failures are not swallowed and nothing is loaded."""
    mlflow_mocks.client.get_model_version_by_alias.side_effect = RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        model_loader.load_registered_model("m", "champion", "http://mlflow:5000")

    mlflow_mocks.load_model.assert_not_called()


def test_configure_mlflow_http_sets_defaults(monkeypatch):
    """Fail-fast HTTP settings are applied when not configured."""
    monkeypatch.delenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", raising=False)
    monkeypatch.delenv("MLFLOW_HTTP_REQUEST_TIMEOUT", raising=False)

    model_loader.configure_mlflow_http()

    assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "1"
    assert os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] == "10"


def test_configure_mlflow_http_keeps_existing_values(monkeypatch):
    """Values defined in the environment take precedence over the defaults."""
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "5")

    model_loader.configure_mlflow_http()

    assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "5"
