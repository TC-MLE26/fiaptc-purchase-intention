"""Wrapper that applies the decision threshold to a loaded model."""

from typing import Any

from src.schemas.predict_schema import PredictRequest, PredictResponse


class PurchasePredictor:
    """Encapsulate a probabilistic classifier and its decision threshold.

    The estimator must expose ``predict_proba`` and accept the raw dataset
    columns; column 1 of its output is the probability of a purchase.
    """

    def __init__(
        self,
        estimator: Any,
        threshold: float,
        model_name: str,
        model_version: str,
    ) -> None:
        self._estimator = estimator
        self._threshold = threshold
        self._model_name = model_name
        self._model_version = model_version

    def predict(self, request: PredictRequest) -> PredictResponse:
        """Return the purchase probability and decision for one session."""
        probability = float(self._estimator.predict_proba(request.to_frame())[0, 1])

        return PredictResponse(
            purchase_probability=probability,
            purchase_prediction=probability >= self._threshold,
            threshold=self._threshold,
            model_name=self._model_name,
            model_version=self._model_version,
        )
