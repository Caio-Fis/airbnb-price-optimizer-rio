"""Testes para a API FastAPI."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


@pytest.fixture
def client():
    with patch("src.serving.predict.Predictor._load"):
        from src.serving.main import app
        return TestClient(app)


@pytest.fixture
def mock_predictor():
    predictor = MagicMock()
    predictor.is_ready.return_value = True
    predictor.model_version = "test-v1"
    from src.serving.schemas import PredictionResponse
    predictor.predict.return_value = PredictionResponse(
        predicted_price=350.0,
        price_range_low=297.5,
        price_range_high=402.5,
    )
    return predictor


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_predict_valid_request(client, mock_predictor):
    import src.serving.main as main_module
    main_module.predictor = mock_predictor

    payload = {
        "neighbourhood": "Pinheiros",
        "room_type": "Entire home/apt",
        "accommodates": 4,
        "bathrooms": 1.0,
        "bedrooms": 2,
        "beds": 2,
        "amenities": ["wifi", "kitchen"],
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price" in data
    assert "price_range_low" in data
    assert "price_range_high" in data
    assert data["price_range_low"] <= data["predicted_price"] <= data["price_range_high"]


def test_predict_missing_required_field(client):
    payload = {"neighbourhood": "Pinheiros"}  # faltam campos obrigatórios
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_accommodates(client, mock_predictor):
    import src.serving.main as main_module
    main_module.predictor = mock_predictor

    payload = {
        "neighbourhood": "Pinheiros",
        "room_type": "Entire home/apt",
        "accommodates": 50,  # > 20, inválido
        "bathrooms": 1.0,
        "bedrooms": 2,
        "beds": 2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_root_endpoint(client):
    """Com o build do frontend presente, '/' serve o index.html; sem ele, JSON informativo."""
    response = client.get("/")
    assert response.status_code == 200
    content_type = response.headers.get("content-type", "")
    if "text/html" in content_type:
        assert '<div id="root">' in response.text
    else:
        assert "docs" in response.json()


def test_listings_map_endpoint(client):
    response = client.get("/listings/map")
    assert response.status_code == 200
    assert response.headers.get("cache-control") == "public, max-age=3600"
    data = response.json()
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) > 40000
    # ids devem ser strings — inteiros > 2^53 corrompem em JavaScript
    assert isinstance(data["features"][0]["properties"]["id"], str)
