import io
import logging

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel
from PIL import Image

from app.core.model import model_manager

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB


# ── Response schemas ──────────────────────────────────────────────────────────

class Prediction(BaseModel):
    label: str
    confidence: float  # percentage, e.g. 87.34


class PredictResponse(BaseModel):
    filename: str
    predictions: list[Prediction]
    inference_ms: float


class LoadModelResponse(BaseModel):
    status: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/predict", response_model=PredictResponse, summary="Classify an image")
async def predict_image(file: UploadFile = File(...)) -> PredictResponse:
    """
    Upload a JPEG/PNG/WebP image and receive the top-5 ImageNet predictions
    with confidence scores and inference latency.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. Upload a JPEG, PNG, or WebP image.",
        )

    contents = await file.read()

    if len(contents) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(contents) // 1024} KB). Maximum allowed size is 10 MB.",
        )

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=422, detail="Could not decode image — ensure the file is a valid image.")

    try:
        result = model_manager.predict(image)
    except Exception:
        logger.exception("Inference failed for file '%s'", file.filename)
        raise HTTPException(status_code=500, detail="Inference failed — check server logs.")

    logger.info(
        "Predicted '%s' (%.0f%%) for file '%s' in %.1f ms",
        result["predictions"][0]["label"],
        result["predictions"][0]["confidence"],
        file.filename,
        result["inference_ms"],
    )

    return PredictResponse(
        filename=file.filename or "upload",
        predictions=[Prediction(**p) for p in result["predictions"]],
        inference_ms=result["inference_ms"],
    )


@router.post(
    "/load-model",
    response_model=LoadModelResponse,
    summary="Reload model from S3",
    description="Trigger a hot-reload of the model from the configured S3 bucket. "
                "Useful for deploying a new fine-tuned checkpoint without restarting the container.",
)
async def load_model() -> LoadModelResponse:
    try:
        model_manager.load_from_s3()
        return LoadModelResponse(status="Model reloaded from S3 successfully")
    except Exception:
        logger.exception("Failed to load model from S3")
        raise HTTPException(status_code=500, detail="Model reload failed — check server logs.")
