"""Tests for POST /predict."""

import pytest

from src.inference import provider as provider_module
from src.inference.model_loader import RegisteredModel
from src.schemas.predict_schema import FEATURE_COLUMNS, PredictRequest


def test_predict_returns_probability_and_decision(client, valid_payload):
    """A valid payload yields the model probability and the threshold decision."""
    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    assert response.json() == {
        "purchase_probability": 0.8,
        "purchase_prediction": True,
        "threshold": 0.5,
        "model_name": "purchase-intention-classifier",
        "model_version": "3",
    }


def test_predict_decision_is_false_below_threshold(
    client, fake_estimator, valid_payload
):
    """A probability under the threshold gives a negative prediction."""
    fake_estimator.probability = 0.3

    body = client.post("/predict", json=valid_payload).json()

    assert body["purchase_probability"] == 0.3
    assert body["purchase_prediction"] is False


def test_predict_decision_is_true_at_threshold(client, fake_estimator, valid_payload):
    """A probability equal to the threshold counts as a purchase."""
    fake_estimator.probability = 0.5

    body = client.post("/predict", json=valid_payload).json()

    assert body["purchase_prediction"] is True


def test_predict_sends_original_columns_to_model(client, fake_estimator, valid_payload):
    """The model receives one row with the original dataset columns in order."""
    client.post("/predict", json=valid_payload)

    features = fake_estimator.received[0]
    assert list(features.columns) == list(FEATURE_COLUMNS.values())
    assert len(features) == 1
    assert features.loc[0, "PageValues"] == valid_payload["page_values"]
    assert features.loc[0, "Month"] == "Mar"
    assert features.loc[0, "Weekend"].item() is False


def test_predict_missing_required_field_returns_422(client, valid_payload):
    """Dropping a required field triggers request validation."""
    payload = {k: v for k, v in valid_payload.items() if k != "page_values"}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "page_values"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("month", "Abril"),
        ("visitor_type", "Guest"),
        ("administrative", -1),
        ("bounce_rates", 1.5),
        ("special_day", 2),
        ("browser", 14),
        ("operating_systems", 0),
        ("region", 10),
        ("traffic_type", 21),
        ("product_related_duration", -1),
    ],
)
def test_predict_invalid_value_returns_422(client, valid_payload, field, value):
    """Values outside the categorical or numeric domain are rejected."""
    response = client.post(
        "/predict", json={**valid_payload, "month": "Mar", field: value}
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", field]


def test_predict_unknown_field_returns_422(client, valid_payload):
    """Fields outside the request contract are rejected."""
    response = client.post("/predict", json={**valid_payload, "Revenue": True})

    assert response.status_code == 422


def test_predict_without_model_returns_503(client_without_model, valid_payload):
    """/predict answers 503 with a clear message when the model is unavailable."""
    response = client_without_model.post("/predict", json=valid_payload)

    assert response.status_code == 503
    assert "indisponível" in response.json()["detail"]


def test_predict_retries_model_load_after_startup_failure(
    monkeypatch, client_factory, fake_estimator, valid_payload
):
    """A model that appears after startup is picked up by the next /predict."""
    monkeypatch.setattr(provider_module, "FAILURE_COOLDOWN_SECONDS", 0.0)
    attempts: list[int] = []

    def flaky_loader(*args: object) -> RegisteredModel:
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionError("mlflow offline")
        return RegisteredModel(fake_estimator, "m", "1")

    client = client_factory(flaky_loader)

    assert client.get("/health").json()["model_loaded"] is False
    assert client.post("/predict", json=valid_payload).status_code == 200
    assert client.get("/health").json()["model_loaded"] is True


def test_predict_request_accepts_dataset_features(valid_payload):
    """The request contract accepts the 17 dataset features."""
    request = PredictRequest.model_validate(valid_payload)

    assert request.model_dump() == valid_payload
    assert set(valid_payload) == set(FEATURE_COLUMNS)


def test_predict_contract_is_exposed_in_openapi(client):
    """OpenAPI keeps the request and response contracts fixed."""
    operation = client.get("/openapi.json").json()["paths"]["/predict"]["post"]

    request_schema = operation["requestBody"]["content"]["application/json"]["schema"]
    response_schema = operation["responses"]["200"]["content"]["application/json"][
        "schema"
    ]

    assert request_schema["$ref"].endswith("/PredictRequest")
    assert response_schema["$ref"].endswith("/PredictResponse")
    assert "503" in operation["responses"]
