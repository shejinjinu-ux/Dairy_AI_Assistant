"""
NIR Milk Quality Spectroscopy Endpoints
"""

from fastapi import APIRouter, status
from backend.app.schemas.milk_quality import (
    MilkNIRSpectralInput,
    MilkFatPredictionResponse,
    MilkProteinPredictionResponse
)
from backend.app.services.milk_quality_service import milk_quality_service

router = APIRouter(prefix="/predict/milk-quality", tags=["Milk Quality & NIR Spectroscopy"])


@router.post(
    "/fat",
    response_model=MilkFatPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Estimate Milk Fat Percentage from NIR Spectrum"
)
async def predict_milk_fat(payload: MilkNIRSpectralInput):
    """
    Accepts 1032 NIR spectrometer absorbance values across wavelengths and predicts
    milk fat percentage using the production PCA(95) + XGBoost regression pipeline (R² = 0.8723).
    """
    return milk_quality_service.predict_fat(payload)


@router.post(
    "/protein",
    response_model=MilkProteinPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Estimate Milk Protein from NIR Spectrum (Experimental)"
)
async def predict_milk_protein(payload: MilkNIRSpectralInput):
    """
    Experimental model for NIR milk protein estimation (R² = 0.5562).
    Requires ENABLE_EXPERIMENTAL_MODELS=true.
    """
    return milk_quality_service.predict_protein_experimental(payload)
