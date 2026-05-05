from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from app.api import predict, health
from app.core.config import settings
from app.core.model import model_manager

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Production ML Model Serving Platform with FastAPI, AWS S3, Docker, and Prometheus monitoring"
)

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

# Routers
app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, tags=["Inference"])

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    model_manager.load_default()
    print(f"{settings.APP_NAME} v{settings.VERSION} started")