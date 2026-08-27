"""
Silage Quality & Fermentation Quality Index Endpoints
"""

from fastapi import APIRouter, status
from backend.app.schemas.silage import (
    SilageInput,
    SilageQualityClassResponse,
    SilageFQIRegressionResponse,
    SilageComprehensiveResponse
)
from backend.app.services.silage_service import silage_service

router = APIRouter(prefix="/predict/silage", tags=["Silage Quality & Fermentation"])


@router.post(
    "/quality",
    response_model=SilageQualityClassResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify Silage FAO Quality Class"
)
async def predict_silage_quality(payload: SilageInput):
    """
    Accepts 32 proximal and fermentation silage variables and predicts FAO quality class
    ('ea' [early acidity / optimal preservation] vs 'la' [late acidity / secondary risk])
    using the production XGBoost classifier (accuracy: 97.19%, macro F1: 0.967).
    """
    return silage_service.predict_quality_class(payload)


@router.post(
    "/fqi",
    response_model=SilageFQIRegressionResponse,
    status_code=status.HTTP_200_OK,
    summary="Estimate Silage Fermentation Quality Index (FQI)"
)
async def predict_silage_fqi(payload: SilageInput):
    """
    Estimates the numerical Fermentation Quality Index (FQI) of silage
    using the production XGBoost regressor (R² = 0.9719).
    """
    return silage_service.predict_fqi(payload)


@router.post(
    "/comprehensive",
    response_model=SilageComprehensiveResponse,
    status_code=status.HTTP_200_OK,
    summary="Comprehensive Silage Quality & FQI Assessment"
)
async def predict_silage_comprehensive(payload: SilageInput):
    """
    Runs both quality classification and FQI score regression in a single consolidated response.
    """
    return silage_service.predict_comprehensive(payload)
