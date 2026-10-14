"""Application startup and shutdown hooks."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.core.settings import settings
from src.inference.errors import ModelUnavailableError
from src.inference.model_loader import configure_mlflow_http
from src.inference.provider import ModelProvider

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create the model provider and try to load the model once at startup."""
    configure_mlflow_http()
    provider = ModelProvider(
        model_name=settings.MODEL_NAME,
        model_alias=settings.MODEL_ALIAS,
        tracking_uri=settings.MLFLOW_TRACKING_URI,
        threshold=settings.PREDICTION_THRESHOLD,
    )
    app.state.model_provider = provider

    try:
        provider.get_predictor()
    except ModelUnavailableError:
        logger.warning("API started without a model; /predict will retry the load")

    yield
