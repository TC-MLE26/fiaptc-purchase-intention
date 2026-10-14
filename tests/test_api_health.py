"""Tests for GET /health."""

import main
from src.core.settings import settings


def test_health_reports_model_loaded(client):
    """/health reports the service and that the model is loaded."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "API em execução",
        "service": settings.PROJECT_NAME,
        "model_loaded": True,
    }


def test_health_reports_model_not_loaded(client_without_model):
    """/health stays 200 and reports model_loaded false without a model."""
    response = client_without_model.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is False


def test_app_metadata_comes_from_settings():
    """The application title follows the configured project name."""
    assert main.app.title == settings.PROJECT_NAME
