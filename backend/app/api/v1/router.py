"""
Aggregated v1 API Router
"""

from fastapi import APIRouter
from backend.app.api.v1.endpoints import (
    health,
    models,
    disease,
    breed,
    milk_production,
    silage,
    feed_nutrition,
    feed_reference,
    visual_screening,
    quality_analysis,
    milk_quality,
    lab_sensor,
    chat,
    nutrition,
    auth,
)

api_router = APIRouter()

# Register Endpoint Routers
api_router.include_router(health.router)
api_router.include_router(models.router)
api_router.include_router(disease.router)
api_router.include_router(breed.router)
api_router.include_router(milk_production.router)
api_router.include_router(silage.router)
api_router.include_router(feed_nutrition.router)
api_router.include_router(feed_reference.router)
api_router.include_router(visual_screening.router)
api_router.include_router(quality_analysis.router)
api_router.include_router(milk_quality.router)
api_router.include_router(lab_sensor.router)
api_router.include_router(chat.router)
api_router.include_router(nutrition.router)
api_router.include_router(auth.router)
