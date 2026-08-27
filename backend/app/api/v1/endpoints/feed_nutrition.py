"""
Feed Nutrition Multi-Target Prediction Endpoints
"""

from fastapi import APIRouter, status
from backend.app.schemas.feed_nutrition import (
    FeedNutritionInput,
    FeedNutritionMultiTargetResponse,
    NutritionalTargetPrediction
)
from backend.app.services.feed_nutrition_service import feed_nutrition_service

router = APIRouter(prefix="/predict/feed-nutrition", tags=["Feed Nutrition & Proximate Analysis"])


@router.post(
    "",
    response_model=FeedNutritionMultiTargetResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict All 7 Feed Nutritional Parameters"
)
async def predict_all_nutrition(payload: FeedNutritionInput):
    """
    Accepts feed category, INRA2018 classification, and proximal components,
    and returns predictions for all 7 registered nutritional targets:
    - **Crude Protein** (R² = 0.9317)
    - **Dry Matter** (R² = 0.9735)
    - **Crude Fibre** (R² = 0.8206)
    - **NDF - Neutral Detergent Fibre** (R² = 0.8556)
    - **ADF - Acid Detergent Fibre** (R² = 0.8513)
    - **ADL - Acid Detergent Lignin** (R² = 0.7649)
    - **Starch** (R² = 0.9588)
    """
    return feed_nutrition_service.predict_all(payload)


@router.post(
    "/{target}",
    response_model=NutritionalTargetPrediction,
    status_code=status.HTTP_200_OK,
    summary="Predict Individual Feed Nutritional Target"
)
async def predict_single_target(target: str, payload: FeedNutritionInput):
    """
    Predict a specific nutritional target:
    `crude_protein`, `dry_matter`, `crude_fibre`, `ndf`, `adf`, `adl`, `starch`.
    """
    return feed_nutrition_service.predict_target(target, payload)
