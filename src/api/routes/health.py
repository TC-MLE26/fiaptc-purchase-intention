"""Health check route."""

from typing import Annotated

from fastapi import APIRouter, Depends

from src.api.dependencies import get_model_provider
from src.core.settings import settings
from src.inference.provider import ModelProvider
from src.schemas.health_schema import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health(
    provider: Annotated[ModelProvider, Depends(get_model_provider)],
) -> HealthResponse:
    """Return the API health status and whether the model is loaded."""
    return HealthResponse(
        status="API em execução",
        service=settings.PROJECT_NAME,
        model_loaded=provider.is_loaded,
    )
