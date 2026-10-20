"""
Tests for model prediction endpoints and inference behavior.
Executes against the real loaded DistilBERT + LoRA adapter model.
"""

import copy
import pytest
import torch
from fastapi.testclient import TestClient
from app.model_service import ModelService


def test_predict_positive_sentiment(loaded_client: TestClient):
    """Verify that an overtly positive text is correctly classified as positive."""
    payload = {"text": "This product is fantastic and works perfectly."}
    response = loaded_client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["text"] == payload["text"]
    assert data["sentiment"] == "positive"
    assert data["confidence"] > 0.5
    assert data["model"] == "distilbert/distilbert-base-uncased"
    assert data["adapter"] == "LoRA"
    assert data["rank"] == 16

    probs = data["probabilities"]
    assert "positive" in probs and "negative" in probs
    assert probs["positive"] > probs["negative"]
    assert 0.0 <= probs["positive"] <= 1.0
    assert 0.0 <= probs["negative"] <= 1.0
    assert abs((probs["positive"] + probs["negative"]) - 1.0) < 0.01


def test_predict_negative_sentiment(loaded_client: TestClient):
    """Verify that an overtly negative text is correctly classified as negative."""
    payload = {"text": "This is completely terrible, worthless, and utterly disappointing."}
    response = loaded_client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["text"] == payload["text"]
    assert data["sentiment"] == "negative"
    assert data["confidence"] > 0.5
    assert data["model"] == "distilbert/distilbert-base-uncased"
    assert data["adapter"] == "LoRA"
    assert data["rank"] == 16

    probs = data["probabilities"]
    assert probs["negative"] > probs["positive"]
    assert 0.0 <= probs["positive"] <= 1.0
    assert 0.0 <= probs["negative"] <= 1.0
    assert abs((probs["positive"] + probs["negative"]) - 1.0) < 0.01


def test_batch_prediction_order_and_parity(loaded_client: TestClient):
    """Verify batch predictions preserve exact ordering and match single predictions."""
    texts = [
        "An absolute masterpiece with brilliant acting.",
        "A boring and poorly written waste of time.",
        "Remarkably well crafted, truly enjoyed every moment.",
    ]
    batch_response = loaded_client.post("/predict/batch", json={"texts": texts})
    assert batch_response.status_code == 200
    batch_data = batch_response.json()

    assert batch_data["count"] == 3
    assert len(batch_data["predictions"]) == 3

    for i, text in enumerate(texts):
        pred = batch_data["predictions"][i]
        assert pred["text"] == text

        # Cross-validate against single /predict endpoint
        single_response = loaded_client.post("/predict", json={"text": text})
        assert single_response.status_code == 200
        single_data = single_response.json()

        assert pred["sentiment"] == single_data["sentiment"]
        assert pred["confidence"] == single_data["confidence"]
        assert pred["probabilities"]["positive"] == single_data["probabilities"]["positive"]
        assert pred["probabilities"]["negative"] == single_data["probabilities"]["negative"]


def test_repeated_inference_does_not_modify_weights(service_instance: ModelService):
    """Verify that repeated inference runs in evaluation mode and does not alter model weights."""
    model = service_instance.model
    assert model is not None

    # Snapshot parameter clones on CPU
    initial_weights = {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
    }

    # Run multiple inferences
    test_inputs = [
        "First inference run test sentence.",
        "Second inference run test sentence with different words.",
        "Third evaluation to ensure stability.",
    ]
    for text in test_inputs:
        service_instance.predict(text)

    # Check weights remain identical
    for name, param in model.named_parameters():
        assert torch.equal(initial_weights[name], param.detach().cpu()), (
            f"Model parameter '{name}' was modified during inference!"
        )
