"""
FastAPI application for DistilBERT LoRA Sentiment Classification.
Exposes root discovery, health check, single prediction, and batch prediction endpoints.
"""

from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.model_service import ModelLoadingError, ModelNotReadyError, model_service
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    RootResponse,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("lora_api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan context manager.
    Loads the base model and LoRA adapter once at application startup.
    """
    settings = get_settings()
    logger.info("Starting up API service and loading model...")
    try:
        model_service.load()
        logger.info("Model loaded successfully and service is ready for inference.")
    except Exception as exc:
        logger.critical(
            f"Failed to load model during startup: {exc}. Service will respond as unhealthy.",
            exc_info=True,
        )
    yield
    logger.info("Shutting down API service.")


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Production-style FastAPI inference service for binary sentiment classification "
        "using DistilBERT fine-tuned with Hugging Face PEFT / LoRA on SST-2."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware with explicit origins and Vercel/localhost regex
allowed_origins = settings.get_allowed_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:[0-9]+)?$|^https://.*\.vercel\.app$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/",
    response_model=RootResponse,
    summary="Service Information and Discovery",
    tags=["General"],
)
async def root() -> RootResponse:
    """Return service name, version, task details, and available endpoints."""
    return RootResponse(
        name=settings.app_name,
        version=settings.app_version,
        task="Binary sentiment classification using SST-2 (0=negative, 1=positive)",
        endpoints={
            "/": "GET - Service metadata and discovery",
            "/health": "GET - Readiness and health diagnostics",
            "/predict": "POST - Single-text sentiment classification",
            "/predict/batch": "POST - Batch sentiment classification (up to 32 texts)",
            "/docs": "GET - Interactive OpenAPI Swagger UI",
            "/redoc": "GET - ReDoc API documentation",
        },
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health and Readiness Check",
    tags=["Diagnostics"],
)
async def health() -> JSONResponse:
    """
    Return operational readiness, device, base model, and adapter status.
    Returns HTTP 200 when model is loaded and ready; HTTP 503 if still initializing or failed.
    """
    if model_service.is_ready:
        body = HealthResponse(
            status="ready",
            model_loaded=True,
            device=str(model_service.device),
            base_model=settings.base_model_name_or_path,
            adapter_path=str(settings.adapter_path),
            lora_rank=model_service.lora_rank,
            details="Service is fully initialized, evaluation mode active.",
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=body.model_dump())

    body = HealthResponse(
        status="unavailable" if model_service.load_error else "initializing",
        model_loaded=False,
        device=str(model_service.device),
        base_model=settings.base_model_name_or_path,
        adapter_path=str(settings.adapter_path),
        lora_rank=None,
        details=model_service.load_error or "Model is currently initializing.",
    )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=body.model_dump()
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict Sentiment for Single Text",
    tags=["Inference"],
    responses={
        200: {"description": "Sentiment classification result with softmax probabilities"},
        422: {"description": "Validation error (e.g., empty or excessively long text)"},
        503: {"description": "Model is not ready or failed to load"},
    },
)
async def predict(payload: PredictRequest) -> PredictResponse:
    """
    Classify the sentiment of a single input text.
    - Validates text is non-empty and within bounds.
    - Computes class probabilities using softmax.
    - SST-2 mapping: class 0 = negative, class 1 = positive.
    """
    if not model_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model service is not ready: {model_service.load_error or 'Initializing'}",
        )
    try:
        return model_service.predict(payload.text)
    except ModelNotReadyError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.error(f"Inference error during /predict: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal inference failure: {e}",
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    summary="Predict Sentiment for Bounded Batch of Texts",
    tags=["Inference"],
    responses={
        200: {"description": "List of sentiment predictions preserving original order"},
        422: {"description": "Validation error (e.g. empty list or batch exceeding max items)"},
        503: {"description": "Model is not ready or failed to load"},
    },
)
async def predict_batch(payload: BatchPredictRequest) -> BatchPredictResponse:
    """
    Classify the sentiment of a bounded list of texts (up to 32 items).
    Returns predictions in the exact same order as input.
    """
    if not model_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model service is not ready: {model_service.load_error or 'Initializing'}",
        )
    try:
        return model_service.predict_batch(payload.texts)
    except ModelNotReadyError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.error(f"Inference error during /predict/batch: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal batch inference failure: {e}",
        )
