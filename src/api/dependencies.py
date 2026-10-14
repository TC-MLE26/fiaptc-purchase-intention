"""Shared FastAPI dependencies."""

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from src.inference.errors import ModelUnavailableError
from src.inference.predictor import PurchasePredictor
from src.inference.provider import ModelProvider


def get_model_provider(request: Request) -> ModelProvider:
    """Return the model provider created at application startup."""
    provider: ModelProvider = request.app.state.model_provider
    return provider


def get_predictor(
    provider: Annotated[ModelProvider, Depends(get_model_provider)],
) -> PurchasePredictor:
    """Return the predictor or fail with 503 when the model is unavailable."""
    try:
        return provider.get_predictor()
    except ModelUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{exc}. Tente novamente em instantes.",
        ) from exc
