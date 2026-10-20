"""
Application configuration module using Pydantic Settings.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal
import torch
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings for the FastAPI inference service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Metadata
    app_name: str = Field(
        default="DistilBERT LoRA Sentiment Classification API",
        description="Name of the API application",
    )
    app_version: str = Field(
        default="1.0.0",
        description="Semantic version of the application",
    )
    debug: bool = Field(
        default=False,
        description="Whether to run in debug mode",
    )
    host: str = Field(
        default="127.0.0.1",
        description="Host interface for the server",
    )
    port: int = Field(
        default=8000,
        description="Port for the server",
    )

    # Model & Adapter Paths
    base_model_name_or_path: str = Field(
        default="distilbert/distilbert-base-uncased",
        description="Pretrained base model identifier or path",
    )
    adapter_path: Path = Field(
        default=Path("Models/lora-sst2-adapter"),
        description="Path to the trained LoRA adapter directory",
    )

    # Device & Computation Settings
    device: Literal["auto", "mps", "cpu", "cuda"] = Field(
        default="auto",
        description="Target compute device: auto, mps, cpu, or cuda",
    )

    # Inference & Constraints
    max_sequence_length: int = Field(
        default=128,
        ge=16,
        le=512,
        description="Maximum token sequence length (aligned with training setup)",
    )
    expected_lora_rank: int = Field(
        default=16,
        description="Expected LoRA rank parameter for verification",
    )
    decimal_precision: int = Field(
        default=4,
        ge=1,
        le=8,
        description="Number of decimal places for output probabilities and confidence",
    )
    max_batch_size: int = Field(
        default=32,
        ge=1,
        le=128,
        description="Maximum allowed batch size for batch predictions",
    )
    max_input_char_length: int = Field(
        default=1000,
        ge=10,
        le=10000,
        description="Maximum allowable raw input character length per text",
    )

    # CORS Settings
    frontend_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000",
        description="Comma-separated list of allowed frontend origins for CORS",
    )

    def get_allowed_origins(self) -> list[str]:
        """Parse comma-separated allowed frontend origins for CORS."""
        return [origin.strip() for origin in self.frontend_origins.split(",") if origin.strip()]

    def resolve_device(self) -> torch.device:
        """
        Resolve the PyTorch device based on configuration and hardware availability.
        Prefers Apple Silicon MPS or CUDA when set to 'auto', gracefully falling back to CPU.
        """
        if self.device == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")

        if self.device == "mps":
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                return torch.device("mps")
            return torch.device("cpu")

        if self.device == "cpu":
            return torch.device("cpu")

        # Auto detection
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


@lru_cache()
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    return Settings()
