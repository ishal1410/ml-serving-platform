import io
import logging
import time
import threading

import boto3
import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image

from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelManager:
    """Thread-safe wrapper around an EfficientNet-B0 image classifier."""

    def __init__(self) -> None:
        self.model: torch.nn.Module | None = None
        self._lock = threading.Lock()

        # Use the official weights object — gives us the correct pre-processing
        # transform AND all 1000 ImageNet class names with a single import.
        self._weights = EfficientNet_B0_Weights.DEFAULT
        self.transform = self._weights.transforms()
        self.labels: list[str] = self._weights.meta["categories"]  # 1000 classes

    # ── Loading ───────────────────────────────────────────────────────────────

    def load_default(self) -> None:
        """Load the official ImageNet-pretrained EfficientNet-B0 weights."""
        logger.info("Loading default EfficientNet-B0 (ImageNet pretrained)…")
        with self._lock:
            self.model = efficientnet_b0(weights=self._weights)
            self.model.eval()
        logger.info("Default model ready (%d output classes)", len(self.labels))

    def load_from_s3(self) -> None:
        """Download a fine-tuned model from S3; fall back to default on error."""
        logger.info(
            "Downloading model from s3://%s/%s…", settings.S3_BUCKET, settings.MODEL_KEY
        )
        try:
            s3 = boto3.client(
                "s3",
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_REGION,
            )
            buf = io.BytesIO()
            s3.download_fileobj(settings.S3_BUCKET, settings.MODEL_KEY, buf)
            buf.seek(0)
            with self._lock:
                self.model = torch.load(buf, map_location="cpu")
                self.model.eval()
            logger.info("S3 model loaded successfully")
        except Exception:
            logger.exception("S3 load failed — falling back to default model")
            self.load_default()

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, image: Image.Image, top_k: int = 5) -> dict:
        """
        Run inference and return the top-k predictions with confidence scores.

        Returns:
            {
                "predictions": [{"label": str, "confidence": float}, ...],
                "inference_ms": float
            }
        """
        if not self.is_ready:
            self.load_default()

        tensor = self.transform(image).unsqueeze(0)

        t0 = time.perf_counter()
        with self._lock:
            with torch.no_grad():
                logits = self.model(tensor)
        inference_ms = round((time.perf_counter() - t0) * 1000, 2)

        probs = torch.nn.functional.softmax(logits[0], dim=0)
        top = torch.topk(probs, min(top_k, len(self.labels)))

        predictions = [
            {
                "label": self.labels[idx.item()],
                "confidence": round(prob.item() * 100, 2),
            }
            for prob, idx in zip(top.values, top.indices)
        ]

        return {"predictions": predictions, "inference_ms": inference_ms}


model_manager = ModelManager()
