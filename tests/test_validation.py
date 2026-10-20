"""
Lightweight API validation tests and adapter loading failure handling tests.
These tests verify input boundaries, HTTP 422 errors, and error handling for missing/corrupted adapters.
"""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from app.config import Settings
from app.model_service import ModelLoadingError, ModelService


def test_empty_text_returns_422(loaded_client: TestClient):
    """Empty string should be rejected with HTTP 422."""
    response = loaded_client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_whitespace_text_returns_422(loaded_client: TestClient):
    """Whitespace-only string should be rejected with HTTP 422."""
    response = loaded_client.post("/predict", json={"text": "   \n\t  "})
    assert response.status_code == 422


def test_excessive_length_text_returns_422(loaded_client: TestClient):
    """Text exceeding 1000 characters should be rejected with HTTP 422."""
    excessive_text = "word " * 250  # > 1000 chars
    response = loaded_client.post("/predict", json={"text": excessive_text})
    assert response.status_code == 422


def test_missing_text_field_returns_422(loaded_client: TestClient):
    """Payload without 'text' field should be rejected with HTTP 422."""
    response = loaded_client.post("/predict", json={"unrelated": "data"})
    assert response.status_code == 422


def test_extra_fields_forbidden_returns_422(loaded_client: TestClient):
    """Payload with extra fields should be rejected with HTTP 422."""
    response = loaded_client.post(
        "/predict",
        json={"text": "Valid text.", "malicious_injected_field": True},
    )
    assert response.status_code == 422


def test_batch_empty_list_returns_422(loaded_client: TestClient):
    """Batch payload with empty list should be rejected with HTTP 422."""
    response = loaded_client.post("/predict/batch", json={"texts": []})
    assert response.status_code == 422


def test_batch_exceeding_max_items_returns_422(loaded_client: TestClient):
    """Batch payload exceeding 32 items should be rejected with HTTP 422."""
    items = ["Sample text item."] * 33
    response = loaded_client.post("/predict/batch", json={"texts": items})
    assert response.status_code == 422


def test_batch_with_whitespace_item_returns_422(loaded_client: TestClient):
    """Batch containing a blank item should be rejected with HTTP 422."""
    response = loaded_client.post(
        "/predict/batch",
        json={"texts": ["Valid review.", "    "]},
    )
    assert response.status_code == 422


def test_adapter_missing_directory_raises_clear_error(tmp_path: Path):
    """Verifying non-existent adapter path raises ModelLoadingError with clear instructions."""
    non_existent = tmp_path / "does_not_exist"
    service = ModelService(Settings(adapter_path=non_existent))

    with pytest.raises(ModelLoadingError) as exc_info:
        service.verify_adapter_artifacts(non_existent)
    assert "Adapter directory not found" in str(exc_info.value)
    assert "Please ensure the adapter files are extracted" in str(exc_info.value)


def test_adapter_missing_config_raises_clear_error(tmp_path: Path):
    """Verifying missing adapter_config.json raises ModelLoadingError."""
    adapter_dir = tmp_path / "empty_adapter"
    adapter_dir.mkdir()
    service = ModelService(Settings(adapter_path=adapter_dir))

    with pytest.raises(ModelLoadingError) as exc_info:
        service.verify_adapter_artifacts(adapter_dir)
    assert "Missing 'adapter_config.json'" in str(exc_info.value)


def test_adapter_missing_weights_raises_clear_error(tmp_path: Path):
    """Verifying missing adapter weights file raises ModelLoadingError."""
    adapter_dir = tmp_path / "adapter_no_weights"
    adapter_dir.mkdir()
    config_file = adapter_dir / "adapter_config.json"
    with open(config_file, "w") as f:
        json.dump({"peft_type": "LORA", "r": 16}, f)

    service = ModelService(Settings(adapter_path=adapter_dir))
    with pytest.raises(ModelLoadingError) as exc_info:
        service.verify_adapter_artifacts(adapter_dir)
    assert "Missing adapter weights" in str(exc_info.value)


def test_adapter_rank_mismatch_raises_clear_error(tmp_path: Path):
    """Verifying rank mismatch in config raises ModelLoadingError."""
    adapter_dir = tmp_path / "adapter_wrong_rank"
    adapter_dir.mkdir()
    config_file = adapter_dir / "adapter_config.json"
    weights_file = adapter_dir / "adapter_model.safetensors"
    weights_file.touch()

    with open(config_file, "w") as f:
        json.dump({"peft_type": "LORA", "r": 8}, f)

    service = ModelService(Settings(adapter_path=adapter_dir, expected_lora_rank=16))
    with pytest.raises(ModelLoadingError) as exc_info:
        service.verify_adapter_artifacts(adapter_dir)
    assert "LoRA rank mismatch: expected 16, got 8" in str(exc_info.value)
