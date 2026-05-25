import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from prometheus_fastapi_instrumentator import Instrumentator

from app.api import predict, health
from app.core.config import settings
from app.core.model import model_manager

PUBLIC_DIR = Path(__file__).parent.parent / "public"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once on startup; clean up on shutdown."""
    model_manager.load_default()
    logger.info("%s v%s ready", settings.APP_NAME, settings.VERSION)
    yield
    # Add teardown logic here if needed (e.g. GPU memory release)
    logger.info("Shutting down")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description=(
        "Production ML model serving platform — "
        "EfficientNet-B0 image classification with FastAPI, "
        "AWS S3 model storage, and Prometheus monitoring."
    ),
    lifespan=lifespan,
)

# Expose /metrics for Prometheus scraping
Instrumentator().instrument(app).expose(app)

app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, tags=["Inference"])

# Serve the UI — mount after API routes so /docs, /health etc. take priority
if PUBLIC_DIR.exists():
    app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="static")
