"""
Milk Production Yield Estimation Endpoint
"""

from fastapi import APIRouter, status
from backend.app.schemas.milk_production import (
    MilkProductionInput,
    MilkProductionPredictionResponse
)
from backend.app.services.milk_production_service import milk_production_service

router = APIRouter(prefix="/predict/milk-production", tags=["Milk Production & Lactation"])


@router.post(
    "",
    response_model=MilkProductionPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Estimate Daily Milk Yield"
)
async def predict_milk_production(payload: MilkProductionInput):
    """
    Accepts cow lactation stage, body parameters, feeding intake, rumination, environmental metrics,
    and vaccination status, and estimates expected daily milk yield in Litres using the
    production XGBoost regressor (R² = 0.946).
    """
    return milk_production_service.predict(payload)
