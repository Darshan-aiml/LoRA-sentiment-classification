"""
Pytest fixtures and configuration for FastAPI LoRA test suite.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.model_service import model_service


@pytest.fixture(scope="session")
def loaded_client():
    """
    Session-scoped TestClient that triggers lifespan startup to load the real model once.
    """
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="session")
def service_instance():
    """
    Direct access to the loaded ModelService singleton.
    """
    if not model_service.is_ready:
        model_service.load()
    return model_service
