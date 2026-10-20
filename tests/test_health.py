"""
Tests for root metadata and health check endpoints.
"""

from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
from app.config import Settings
from app.main import app
from app.model_service import ModelService


def test_root_endpoint(loaded_client: TestClient):
    """Verify that GET / returns service discovery information."""
    response = loaded_client.get("/")
    assert response.status_code == 200
    data = response.json()

    assert "name" in data
    assert "version" in data
    assert "task" in data
    assert "endpoints" in data
    assert "SST-2" in data["task"]

    endpoints = data["endpoints"]
    assert "/" in endpoints
    assert "/health" in endpoints
    assert "/predict" in endpoints
    assert "/predict/batch" in endpoints
    assert "/docs" in endpoints


def test_health_endpoint_when_ready(loaded_client: TestClient):
    """Verify that GET /health returns 200 and operational details when ready."""
    response = loaded_client.get("/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "ready"
    assert data["model_loaded"] is True
    assert "cpu" in data["device"] or "mps" in data["device"] or "cuda" in data["device"]
    assert data["base_model"] == "distilbert/distilbert-base-uncased"
    assert "lora-sst2-adapter" in data["adapter_path"]
    assert data["lora_rank"] == 16


def test_health_endpoint_when_not_ready():
    """Verify that GET /health returns 503 when the model has not loaded."""
    mock_service = ModelService()
    mock_service.is_ready = False
    mock_service.load_error = "Adapter weights not found"

    with patch("app.main.model_service", mock_service):
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unavailable"
        assert data["model_loaded"] is False
        assert "Adapter weights not found" in data["details"]


def test_device_resolution_fallback():
    """Verify device resolution behaves predictably across configurations."""
    cpu_settings = Settings(device="cpu")
    assert str(cpu_settings.resolve_device()) == "cpu"

    auto_settings = Settings(device="auto")
    device = auto_settings.resolve_device()
    assert str(device) in ["cpu", "mps", "cuda"]
