"""Lazy, non-blocking access to the registered model."""

import logging
import threading
import time
from collections.abc import Callable

from src.inference.errors import ModelUnavailableError
from src.inference.model_loader import RegisteredModel, load_registered_model
from src.inference.predictor import PurchasePredictor

logger = logging.getLogger(__name__)

FAILURE_COOLDOWN_SECONDS = 10.0


class ModelProvider:
    """Hold the predictor and retry the load with a minimum interval.

    Requests never wait for a load in progress: while one thread loads the
    model, or while the failure cooldown is active, the others fail fast.
    """

    def __init__(
        self,
        model_name: str,
        model_alias: str,
        tracking_uri: str,
        threshold: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._model_name = model_name
        self._model_alias = model_alias
        self._tracking_uri = tracking_uri
        self._threshold = threshold
        self._clock = clock
        self._predictor: PurchasePredictor | None = None
        self._retry_at = 0.0
        self._load_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        """Whether a model is currently available for inference."""
        return self._predictor is not None

    def get_predictor(self) -> PurchasePredictor:
        """Return the predictor, loading the model if needed."""
        predictor = self._predictor
        if predictor is not None:
            return predictor
        if not self._load_lock.acquire(blocking=False):
            raise self._unavailable("carregamento em andamento")
        try:
            return self._load_if_due()
        finally:
            self._load_lock.release()

    def _load_if_due(self) -> PurchasePredictor:
        """Load the model unless it is already loaded or in cooldown."""
        if self._predictor is not None:
            return self._predictor
        if self._clock() < self._retry_at:
            raise self._unavailable("nova tentativa em instantes")
        try:
            self._predictor = self._build_predictor(self._fetch_model())
        except ModelUnavailableError:
            self._retry_at = self._clock() + FAILURE_COOLDOWN_SECONDS
            raise
        return self._predictor

    def _fetch_model(self) -> RegisteredModel:
        """Load the model from the registry, logging any failure."""
        try:
            return load_registered_model(
                self._model_name, self._model_alias, self._tracking_uri
            )
        except Exception as exc:
            logger.exception(
                "Failed to load model %s@%s from %s",
                self._model_name,
                self._model_alias,
                self._tracking_uri,
            )
            raise self._unavailable("falha ao carregar") from exc

    def _build_predictor(self, registered: RegisteredModel) -> PurchasePredictor:
        """Wrap a registered model with the configured threshold."""
        return PurchasePredictor(
            registered.estimator,
            self._threshold,
            registered.name,
            registered.version,
        )

    def _unavailable(self, reason: str) -> ModelUnavailableError:
        """Build the error raised while the model cannot be served."""
        return ModelUnavailableError(
            f"Modelo '{self._model_name}@{self._model_alias}' indisponível ({reason})"
        )
