"""
Model service module for loading the base model, LoRA adapter, tokenizer,
and executing production-grade sentiment classification inference.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from app.config import Settings, get_settings
from app.schemas import (
    BatchPredictResponse,
    PredictResponse,
    SentimentProbabilities,
)

logger = logging.getLogger(__name__)


class ModelLoadingError(RuntimeError):
    """Raised when the base model, adapter, or tokenizer fails to load properly."""


class ModelNotReadyError(RuntimeError):
    """Raised when inference is attempted on an uninitialized model service."""


class ModelService:
    """
    Manages the lifecycle, verification, and inference of the DistilBERT LoRA sentiment model.
    Loads model and tokenizer once at startup and evaluates inference in evaluation mode.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings: Settings = settings or get_settings()
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.device: torch.device = torch.device("cpu")
        self.is_ready: bool = False
        self.lora_rank: Optional[int] = None
        self.load_error: Optional[str] = None

    def verify_adapter_artifacts(self, adapter_path: Path) -> dict:
        """
        Verify the existence of required adapter files and inspect configuration.
        """
        if not adapter_path.exists() or not adapter_path.is_dir():
            raise ModelLoadingError(
                f"Adapter directory not found at: '{adapter_path.resolve()}'. "
                f"Please ensure the adapter files are extracted into this folder."
            )

        config_path = adapter_path / "adapter_config.json"
        if not config_path.exists():
            raise ModelLoadingError(
                f"Missing 'adapter_config.json' inside '{adapter_path.resolve()}'. "
                f"Expected LoRA configuration file was not found."
            )

        # Check weights file
        safetensors_path = adapter_path / "adapter_model.safetensors"
        bin_path = adapter_path / "adapter_model.bin"
        if not safetensors_path.exists() and not bin_path.exists():
            raise ModelLoadingError(
                f"Missing adapter weights ('adapter_model.safetensors' or 'adapter_model.bin') "
                f"inside '{adapter_path.resolve()}'."
            )

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
        except Exception as e:
            raise ModelLoadingError(f"Failed to parse 'adapter_config.json': {e}") from e

        # Validate LoRA configuration
        peft_type = config_data.get("peft_type")
        if peft_type != "LORA":
            raise ModelLoadingError(f"Unexpected PEFT type '{peft_type}', expected 'LORA'.")

        rank = config_data.get("r")
        if rank != self.settings.expected_lora_rank:
            raise ModelLoadingError(
                f"LoRA rank mismatch: expected {self.settings.expected_lora_rank}, got {rank}."
            )

        self.lora_rank = rank
        logger.info(
            f"Verified adapter configuration: rank={self.lora_rank}, alpha={config_data.get('lora_alpha')}, "
            f"target_modules={config_data.get('target_modules')}, "
            f"modules_to_save={config_data.get('modules_to_save')}"
        )
        return config_data

    def load(self) -> None:
        """
        Load base model, apply PEFT LoRA adapter, load tokenizer, and move to target device.
        Must be called once during application startup.
        """
        try:
            logger.info("Initializing ModelService loading sequence...")
            adapter_dir = self.settings.adapter_path.resolve()

            # 1. Verify adapter files and configuration
            self.verify_adapter_artifacts(adapter_dir)

            # 2. Resolve target device
            self.device = self.settings.resolve_device()
            logger.info(f"Using compute device: {self.device}")

            # 3. Load DistilBERT Sequence Classification base model
            logger.info(
                f"Loading base model '{self.settings.base_model_name_or_path}' with num_labels=2..."
            )
            base_model = AutoModelForSequenceClassification.from_pretrained(
                self.settings.base_model_name_or_path,
                num_labels=2,
            )

            # 4. Attach trained LoRA adapter (including saved classification head weights via modules_to_save)
            logger.info(f"Loading LoRA adapter from '{adapter_dir}'...")
            peft_model = PeftModel.from_pretrained(
                base_model,
                str(adapter_dir),
                is_trainable=False,
            )

            # 5. Move to resolved device and switch to evaluation mode
            peft_model.to(self.device)
            peft_model.eval()
            self.model = peft_model

            # 6. Load tokenizer from adapter directory with fallback to base model
            logger.info(f"Loading tokenizer from '{adapter_dir}'...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
            except Exception as e:
                logger.warning(
                    f"Failed loading tokenizer from '{adapter_dir}' ({e}); "
                    f"falling back to base model tokenizer '{self.settings.base_model_name_or_path}'."
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.settings.base_model_name_or_path
                )

            self.is_ready = True
            self.load_error = None
            logger.info(
                f"ModelService successfully initialized on {self.device}. "
                f"Evaluation mode active. Base model: {self.settings.base_model_name_or_path}, LoRA rank: {self.lora_rank}."
            )

        except Exception as exc:
            self.is_ready = False
            self.load_error = str(exc)
            logger.error(f"Failed to load sentiment classification model: {exc}", exc_info=True)
            raise

    def predict(self, text: str) -> PredictResponse:
        """
        Run inference on a single text string using SST-2 label mapping:
        Class 0 = negative, Class 1 = positive.
        """
        if not self.is_ready or self.model is None or self.tokenizer is None:
            raise ModelNotReadyError(
                f"Model is not ready for inference. Status: {self.load_error or 'Initializing'}"
            )

        # Tokenize with max sequence length and truncation
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.settings.max_sequence_length,
            padding=False,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)
            logits = outputs.logits  # shape: [1, 2]
            probabilities = torch.softmax(logits, dim=-1)[0]

            neg_prob_raw = probabilities[0].item()
            pos_prob_raw = probabilities[1].item()

            pred_class = int(torch.argmax(logits, dim=-1)[0].item())
            sentiment = "positive" if pred_class == 1 else "negative"

            # Precision-rounded values
            neg_prob = round(float(neg_prob_raw), self.settings.decimal_precision)
            pos_prob = round(float(pos_prob_raw), self.settings.decimal_precision)
            confidence = pos_prob if pred_class == 1 else neg_prob

        return PredictResponse(
            text=text,
            sentiment=sentiment,
            confidence=confidence,
            probabilities=SentimentProbabilities(
                negative=neg_prob,
                positive=pos_prob,
            ),
            model=self.settings.base_model_name_or_path,
            adapter="LoRA",
            rank=self.lora_rank or self.settings.expected_lora_rank,
        )

    def predict_batch(self, texts: List[str]) -> BatchPredictResponse:
        """
        Run batched inference across a bounded list of texts preserving original order.
        """
        if not self.is_ready or self.model is None or self.tokenizer is None:
            raise ModelNotReadyError(
                f"Model is not ready for inference. Status: {self.load_error or 'Initializing'}"
            )

        if not texts:
            return BatchPredictResponse(predictions=[], count=0)

        # Batch tokenization with padding
        encoded = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.settings.max_sequence_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in encoded.items()}

        predictions: List[PredictResponse] = []
        with torch.inference_mode():
            outputs = self.model(**inputs)
            logits = outputs.logits  # shape: [batch_size, 2]
            probabilities = torch.softmax(logits, dim=-1)
            pred_classes = torch.argmax(logits, dim=-1)

            for i, text in enumerate(texts):
                neg_prob_raw = probabilities[i, 0].item()
                pos_prob_raw = probabilities[i, 1].item()
                pred_class = int(pred_classes[i].item())

                sentiment = "positive" if pred_class == 1 else "negative"
                neg_prob = round(float(neg_prob_raw), self.settings.decimal_precision)
                pos_prob = round(float(pos_prob_raw), self.settings.decimal_precision)
                confidence = pos_prob if pred_class == 1 else neg_prob

                predictions.append(
                    PredictResponse(
                        text=text,
                        sentiment=sentiment,
                        confidence=confidence,
                        probabilities=SentimentProbabilities(
                            negative=neg_prob,
                            positive=pos_prob,
                        ),
                        model=self.settings.base_model_name_or_path,
                        adapter="LoRA",
                        rank=self.lora_rank or self.settings.expected_lora_rank,
                    )
                )

        return BatchPredictResponse(predictions=predictions, count=len(predictions))


# Global service instance
model_service = ModelService()
