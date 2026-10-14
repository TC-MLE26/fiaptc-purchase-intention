"""Shared pytest fixtures for the API test suite."""

from collections.abc import Callable, Iterator

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import main
from src.inference import provider as provider_module
from src.inference.model_loader import RegisteredModel

MODEL_NAME = "purchase-intention-classifier"
MODEL_VERSION = "3"


class FakeEstimator:
    """Estimator stub that returns a fixed purchase probability."""

    def __init__(self, probability: float = 0.8) -> None:
        self.probability = probability
        self.received: list[pd.DataFrame] = []

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return [P(no purchase), P(purchase)] for each received row."""
        self.received.append(features)
        return np.array([[1 - self.probability, self.probability]] * len(features))


@pytest.fixture(autouse=True)
def mlflow_http_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the MLflow HTTP defaults applied at startup out of the real env."""
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")


@pytest.fixture()
def fake_estimator() -> FakeEstimator:
    """Estimator stub with a purchase probability of 0.8."""
    return FakeEstimator()


@pytest.fixture()
def client_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., TestClient]:
    """Build clients whose model loader is replaced by the given callable."""
    clients: list[TestClient] = []

    def build(loader: Callable[..., RegisteredModel]) -> TestClient:
        monkeypatch.setattr(provider_module, "load_registered_model", loader)
        test_client = TestClient(main.app)
        test_client.__enter__()
        clients.append(test_client)
        return test_client

    yield build

    for test_client in clients:
        test_client.__exit__(None, None, None)


@pytest.fixture()
def client(client_factory, fake_estimator: FakeEstimator) -> Iterator[TestClient]:
    """Test client with a fake model available."""
    registered = RegisteredModel(fake_estimator, MODEL_NAME, MODEL_VERSION)
    yield client_factory(lambda *args: registered)


@pytest.fixture()
def client_without_model(client_factory) -> Iterator[TestClient]:
    """Test client whose model loader always fails."""

    def failing_loader(*args: object) -> RegisteredModel:
        raise ConnectionError("mlflow offline")

    yield client_factory(failing_loader)


@pytest.fixture()
def valid_payload() -> dict[str, object]:
    """A known-valid PredictRequest payload (mirrors the schema example)."""
    return {
        "administrative": 9,
        "administrative_duration": 111.5,
        "informational": 1,
        "informational_duration": 48.5,
        "product_related": 49,
        "product_related_duration": 1868.819697,
        "bounce_rates": 0.0,
        "exit_rates": 0.020708874,
        "page_values": 1.706014966,
        "special_day": 0.0,
        "month": "Mar",
        "operating_systems": 2,
        "browser": 2,
        "region": 7,
        "traffic_type": 2,
        "visitor_type": "Returning_Visitor",
        "weekend": False,
    }
