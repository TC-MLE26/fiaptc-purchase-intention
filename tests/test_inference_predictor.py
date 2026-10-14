"""Tests for the predictor and the model provider."""

import threading

import pytest

from src.inference import provider as provider_module
from src.inference.errors import ModelUnavailableError
from src.inference.model_loader import RegisteredModel
from src.inference.predictor import PurchasePredictor
from src.inference.provider import FAILURE_COOLDOWN_SECONDS, ModelProvider
from src.schemas.predict_schema import PredictRequest


class FakeClock:
    """Controllable replacement for time.monotonic."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class ScriptedLoader:
    """Loader that fails while ``failing`` is set and counts its calls."""

    def __init__(self, estimator: object) -> None:
        self.estimator = estimator
        self.failing = True
        self.calls = 0

    def __call__(self, *args: str) -> RegisteredModel:
        self.calls += 1
        if self.failing:
            raise ConnectionError("mlflow offline")
        return RegisteredModel(self.estimator, "m", "1")


@pytest.fixture()
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture()
def loader(monkeypatch, fake_estimator) -> ScriptedLoader:
    scripted = ScriptedLoader(fake_estimator)
    monkeypatch.setattr(provider_module, "load_registered_model", scripted)
    return scripted


@pytest.fixture()
def provider(clock, loader) -> ModelProvider:
    return ModelProvider("m", "champion", "http://mlflow:5000", 0.5, clock)


def test_predictor_applies_threshold(fake_estimator, valid_payload):
    """The decision follows the configured threshold."""
    request = PredictRequest.model_validate(valid_payload)

    strict = PurchasePredictor(fake_estimator, 0.9, "m", "1").predict(request)
    lenient = PurchasePredictor(fake_estimator, 0.7, "m", "1").predict(request)

    assert strict.purchase_prediction is False
    assert lenient.purchase_prediction is True
    assert strict.threshold == 0.9


def test_provider_caches_loaded_model(provider, loader):
    """The model is loaded once and reused."""
    loader.failing = False

    assert provider.is_loaded is False
    first = provider.get_predictor()
    second = provider.get_predictor()

    assert first is second
    assert provider.is_loaded is True
    assert loader.calls == 1


def test_provider_wraps_loader_failure_and_logs(provider, caplog):
    """Loader failures become ModelUnavailableError and are logged as errors."""
    with pytest.raises(ModelUnavailableError, match="m@champion"):
        provider.get_predictor()

    assert provider.is_loaded is False
    error_records = [r for r in caplog.records if r.levelname == "ERROR"]
    assert error_records
    assert error_records[0].exc_info is not None


def test_provider_fails_fast_during_cooldown(provider, loader, clock):
    """After a failure, calls inside the cooldown do not reach the loader."""
    with pytest.raises(ModelUnavailableError):
        provider.get_predictor()
    clock.advance(FAILURE_COOLDOWN_SECONDS - 1)

    with pytest.raises(ModelUnavailableError):
        provider.get_predictor()

    assert loader.calls == 1


def test_provider_retries_after_cooldown(provider, loader, clock):
    """Once the cooldown elapsed, the next call loads the model again."""
    with pytest.raises(ModelUnavailableError):
        provider.get_predictor()
    loader.failing = False
    clock.advance(FAILURE_COOLDOWN_SECONDS)

    provider.get_predictor()

    assert loader.calls == 2
    assert provider.is_loaded is True


def test_provider_extends_cooldown_after_repeated_failure(provider, loader, clock):
    """A failed retry starts a new cooldown window."""
    with pytest.raises(ModelUnavailableError):
        provider.get_predictor()
    clock.advance(FAILURE_COOLDOWN_SECONDS)
    with pytest.raises(ModelUnavailableError):
        provider.get_predictor()
    clock.advance(FAILURE_COOLDOWN_SECONDS - 1)

    with pytest.raises(ModelUnavailableError):
        provider.get_predictor()

    assert loader.calls == 2


def test_provider_loads_once_under_concurrency(monkeypatch, fake_estimator, clock):
    """Concurrent requests during a load do not start extra loads."""
    entered = threading.Event()
    release = threading.Event()
    calls: list[int] = []

    def slow_loader(*args: str) -> RegisteredModel:
        calls.append(1)
        entered.set()
        release.wait(timeout=5)
        return RegisteredModel(fake_estimator, "m", "1")

    monkeypatch.setattr(provider_module, "load_registered_model", slow_loader)
    provider = ModelProvider("m", "champion", "http://mlflow:5000", 0.5, clock)
    first = threading.Thread(target=provider.get_predictor)
    first.start()
    assert entered.wait(timeout=5)

    errors: list[Exception] = []
    others = [
        threading.Thread(target=_collect_error, args=(provider, errors))
        for _ in range(4)
    ]
    for thread in others:
        thread.start()
    for thread in others:
        thread.join(timeout=5)
    release.set()
    first.join(timeout=5)

    assert len(errors) == 4
    assert len(calls) == 1
    assert provider.is_loaded is True


def _collect_error(provider: ModelProvider, errors: list[Exception]) -> None:
    try:
        provider.get_predictor()
    except ModelUnavailableError as exc:
        errors.append(exc)
