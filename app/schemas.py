"""
Pydantic schemas for request and response validation.
"""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class RootResponse(BaseModel):
    """Response schema for the root service discovery endpoint."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    task: str = Field(..., description="Task description")
    endpoints: Dict[str, str] = Field(..., description="Map of available endpoints and methods")


class HealthResponse(BaseModel):
    """Response schema for the health check endpoint."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ready", "degraded", "initializing", "unavailable"] = Field(
        ..., description="Application operational status"
    )
    model_loaded: bool = Field(..., description="True if base model and adapter are loaded and ready")
    device: str = Field(..., description="Current compute device (mps, cpu, cuda)")
    base_model: str = Field(..., description="Underlying base model identifier")
    adapter_path: str = Field(..., description="Path to the configured LoRA adapter")
    lora_rank: Optional[int] = Field(None, description="Verified LoRA rank parameter")
    details: Optional[str] = Field(None, description="Additional health or diagnostic details")


class SentimentProbabilities(BaseModel):
    """Normalized class probability distribution."""

    model_config = ConfigDict(extra="forbid")

    negative: float = Field(..., ge=0.0, le=1.0, description="Softmax probability for negative class")
    positive: float = Field(..., ge=0.0, le=1.0, description="Softmax probability for positive class")


class PredictRequest(BaseModel):
    """Request payload for single-text sentiment prediction."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The text input to classify for sentiment analysis",
        examples=["This product is fantastic and works perfectly."],
    )

    @field_validator("text")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        """Ensure input text is not empty or composed solely of whitespace."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("Input text must not be empty or blank whitespace.")
        return stripped


class PredictResponse(BaseModel):
    """Response payload for single-text sentiment prediction."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., description="Original input text")
    sentiment: Literal["positive", "negative"] = Field(
        ..., description="Predicted sentiment class (positive or negative)"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Softmax output for the predicted class. Note: Softmax values represent uncalibrated relative class weights.",
    )
    probabilities: SentimentProbabilities = Field(
        ..., description="Softmax probability distribution over sentiment classes"
    )
    model: str = Field(..., description="Base model identifier")
    adapter: str = Field(default="LoRA", description="Adapter fine-tuning technique")
    rank: int = Field(..., description="LoRA rank (r) configuration")


class BatchPredictRequest(BaseModel):
    """Request payload for batch sentiment prediction."""

    model_config = ConfigDict(extra="forbid")

    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=32,
        description="Bounded list of texts to classify (maximum 32 items)",
        examples=[
            [
                "The film was gripping and beautifully directed.",
                "Terrible experience, completely disappointed.",
            ]
        ],
    )

    @field_validator("texts")
    @classmethod
    def validate_texts_list(cls, value: List[str]) -> List[str]:
        """Ensure the batch contains valid non-empty entries and strip them."""
        cleaned: List[str] = []
        for i, text in enumerate(value):
            if not isinstance(text, str):
                raise ValueError(f"Item at index {i} must be a string.")
            stripped = text.strip()
            if not stripped:
                raise ValueError(f"Item at index {i} must not be empty or blank whitespace.")
            if len(stripped) > 1000:
                raise ValueError(
                    f"Item at index {i} exceeds maximum allowed length of 1000 characters."
                )
            cleaned.append(stripped)
        return cleaned


class BatchPredictResponse(BaseModel):
    """Response payload for batch sentiment prediction."""

    model_config = ConfigDict(extra="forbid")

    predictions: List[PredictResponse] = Field(..., description="List of predictions in original order")
    count: int = Field(..., description="Number of items evaluated")
