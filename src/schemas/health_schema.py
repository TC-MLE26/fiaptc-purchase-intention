"""Response contract for the health endpoint."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """API health status."""

    status: str = Field(description="Estado da API.")
    service: str = Field(description="Nome do serviço.")
    model_loaded: bool = Field(description="Indica se o modelo está carregado.")
