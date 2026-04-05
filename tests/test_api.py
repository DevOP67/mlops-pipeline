"""
tests/test_api.py
FastAPI endpoint tests using the built-in TestClient.
These run in CI without a live server.
"""

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    """Create a TestClient that shares one app instance for the whole module."""
    from backend.app import app

    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Health / Root
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_returns_json(self, client):
        response = client.get("/")
        assert response.headers["content-type"] == "application/json"

    def test_root_message_present(self, client):
        response = client.get("/")
        data = response.json()
        assert "message" in data


# ---------------------------------------------------------------------------
# /predict endpoint
# ---------------------------------------------------------------------------

VALID_PAYLOAD = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}

SETOSA_PAYLOAD = {
    "sepal_length": 4.9,
    "sepal_width": 3.0,
    "petal_length": 1.4,
    "petal_width": 0.2,
}

VIRGINICA_PAYLOAD = {
    "sepal_length": 6.7,
    "sepal_width": 3.0,
    "petal_length": 5.2,
    "petal_width": 2.3,
}


class TestPredictEndpoint:
    def test_predict_returns_200(self, client):
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200, response.text

    def test_predict_returns_prediction_key(self, client):
        response = client.post("/predict", json=VALID_PAYLOAD)
        data = response.json()
        assert "prediction" in data, f"Expected 'prediction' key, got: {data}"

    def test_predict_returns_string_or_int(self, client):
        response = client.post("/predict", json=VALID_PAYLOAD)
        pred = response.json()["prediction"]
        assert isinstance(pred, (str, int, float)), f"Unexpected prediction type: {type(pred)}"

    def test_predict_setosa(self, client):
        """Setosa sample should not return None."""
        response = client.post("/predict", json=SETOSA_PAYLOAD)
        assert response.status_code == 200
        assert response.json().get("prediction") is not None

    def test_predict_virginica(self, client):
        """Virginica sample should not return None."""
        response = client.post("/predict", json=VIRGINICA_PAYLOAD)
        assert response.status_code == 200
        assert response.json().get("prediction") is not None

    def test_predict_missing_field_returns_422(self, client):
        """Omitting a required field should return HTTP 422 Unprocessable Entity."""
        incomplete = {"sepal_length": 5.1, "sepal_width": 3.5}
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422

    def test_predict_invalid_types_returns_422(self, client):
        """Passing strings for numeric fields should return HTTP 422."""
        bad_payload = {
            "sepal_length": "not-a-number",
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        }
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422

    def test_predict_empty_body_returns_422(self, client):
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_extra_fields_ignored(self, client):
        """Extra fields should not cause a crash."""
        payload = {**VALID_PAYLOAD, "unknown_field": "should_be_ignored"}
        response = client.post("/predict", json=payload)
        assert response.status_code in (200, 422)
