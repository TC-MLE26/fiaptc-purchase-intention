"""Application settings loaded from environment variables."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration shared by the API components."""

    PROJECT_NAME: str = "fiaptc-purchase-intention"
    PROJECT_DESCRIPTION: str = "API de inferência para previsão de intenção de compra"
    VERSION: str = "0.1.0"
    DEBUG: bool = False
    DOCS_ENABLED: bool = True

    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:8000"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list[str] = ["*"]
    CORS_ALLOW_HEADERS: list[str] = ["*"]

    HOST: str = "0.0.0.0"
    PORT: int = Field(default=8075, ge=1, le=65535)

    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MODEL_NAME: str = "purchase-intention-classifier"
    MODEL_ALIAS: str = "champion"
    PREDICTION_THRESHOLD: float = Field(default=0.5, ge=0, le=1)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
