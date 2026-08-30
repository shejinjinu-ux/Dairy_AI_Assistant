"""
Dairy AI Assistant - FastAPI Application Entrypoint
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from backend.config import settings
from backend.app.core.exceptions import register_exception_handlers
from backend.app.services.model_loader import model_loader
from backend.app.api.v1.router import api_router
from backend.app.api.v1.endpoints.chat import router as chat_router
from backend.app.api.v1.endpoints.nutrition import router as nutrition_router
from backend.app.api.v1.endpoints.auth import router as auth_router
from backend.app.schemas.health import HealthResponse
from backend.app.core.registry import registry_manager


# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("dairy_ai.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application Lifespan: Lightweight startup with on-demand lazy loading
    designed for low-memory container environments (e.g. Render 512MB RAM).
    """
    logger.info("=" * 70)
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    logger.info(f"Compute Device: {model_loader.device}")
    logger.info("Model Loading Mode: On-Demand Lazy Loading (Zero Eager Preload)")
    logger.info(f"Experimental Models Enabled: {settings.ENABLE_EXPERIMENTAL_MODELS}")
    logger.info("=" * 70)

    yield

    logger.info(f"Shutting down {settings.PROJECT_NAME}...")
    model_loader.clear_cache()


def create_application() -> FastAPI:
    """FastAPI Application Factory"""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=settings.DESCRIPTION,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )

    # Configure CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register Exception Handlers
    register_exception_handlers(app)

    # Include Main API v1 Router
    app.include_router(api_router, prefix=settings.API_V1_STR)

    # Mount /api/chat, /api/nutrition, and /api/auth directly for backward compatibility and direct route access
    app.include_router(chat_router, prefix="/api", include_in_schema=False)
    app.include_router(nutrition_router, prefix="/api", include_in_schema=False)
    app.include_router(auth_router, prefix="/api", include_in_schema=False)


    # Root Direct Health Endpoints
    @app.get("/health", response_model=HealthResponse, tags=["Health & Status"], include_in_schema=True)
    async def root_health():
        prod_models = registry_manager.list_production_models()
        return HealthResponse(
            status="healthy",
            version=settings.VERSION,
            device=str(model_loader.device),
            models_loaded=model_loader.get_loaded_models_count(),
            production_models_ready=len(prod_models),
            experimental_models_enabled=settings.ENABLE_EXPERIMENTAL_MODELS,
            details={
                "loaded_models": model_loader.get_loaded_keys(),
                "registry_version": registry_manager.version,
                "project_name": settings.PROJECT_NAME
            }
        )

    @app.get("/", include_in_schema=False)
    async def root_redirect():
        return RedirectResponse(url="/docs")

    return app


app = create_application()


if __name__ == "__main__":
    import os
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=False
    )
