"""
Health & Diagnostic Endpoints
"""

from fastapi import APIRouter
from backend.config import settings
from backend.app.schemas.health import HealthResponse, PingResponse
from backend.app.services.model_loader import model_loader
from backend.app.core.registry import registry_manager

router = APIRouter(tags=["Health & Status"])


@router.get("/health", response_model=HealthResponse, summary="System Health Status")
async def get_health():
    """
    Get system health, active compute device, loaded models, and production readiness.
    """
    prod_models = registry_manager.list_production_models()
    loaded_count = model_loader.get_loaded_models_count()

    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        device=str(model_loader.device),
        models_loaded=loaded_count,
        production_models_ready=len(prod_models),
        experimental_models_enabled=settings.ENABLE_EXPERIMENTAL_MODELS,
        details={
            "loaded_models": model_loader.get_loaded_keys(),
            "registry_version": registry_manager.version,
            "project_name": settings.PROJECT_NAME
        }
    )


@router.get("/ping", response_model=PingResponse, summary="Lightweight Ping")
async def get_ping():
    """
    Lightweight ping endpoint for load balancer / heartbeat verification.
    """
    return PingResponse()
