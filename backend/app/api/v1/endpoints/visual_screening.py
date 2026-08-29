"""
Visual Mould & Spoilage Screening Endpoints (Method 1)
"""

import logging
from fastapi import APIRouter, File, UploadFile, status, HTTPException

from backend.app.schemas.visual_screening import (
    FeedVisualScreeningResponse,
    SilageVisualScreeningResponse
)
from backend.app.services.visual_mould_service import visual_mould_service
from backend.app.core.exceptions import ImageProcessingError

logger = logging.getLogger("dairy_ai.api.visual_screening")

router = APIRouter(prefix="/predict", tags=["Visual Mould & Spoilage Screening (Method 1)"])


@router.post(
    "/feed-visual",
    response_model=FeedVisualScreeningResponse,
    status_code=status.HTTP_200_OK,
    summary="Screen Feed Sample for Visual Mould & Spoilage Risk (Method 1)"
)
async def predict_feed_visual(file: UploadFile = File(..., description="Image of feed/grain/roughage sample (JPEG/PNG/WebP)")):
    """
    Accepts an uploaded image of an animal feed sample and runs rapid visual screening
    for surface discolouration, fungal hyphae/mould clusters, and structural spoilage.

    - **Target classes**: `GOOD`, `MOULD_RISK`, `SPOILED`
    - **Disclaimer**: Visual screening only. Laboratory confirmation is required for fungal toxins and mycotoxins.
    """
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image file provided."
        )

    try:
        content = await file.read()
        return visual_mould_service.predict_feed_visual(content)
    except ImageProcessingError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message
        )
    except Exception as e:
        logger.error(f"Unexpected error in feed visual screening: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Visual screening failed: {str(e)}"
        )


@router.post(
    "/silage-visual",
    response_model=SilageVisualScreeningResponse,
    status_code=status.HTTP_200_OK,
    summary="Screen Silage Bunker Face / Sample for Visual Mould & Aerobic Spoilage (Method 1)"
)
async def predict_silage_visual(file: UploadFile = File(..., description="Image of silage bunker face or sample (JPEG/PNG/WebP)")):
    """
    Accepts an uploaded image of silage and screens for surface mould patches,
    aerobic heating discolouration, and slimy/clostridial decomposition.

    - **Target classes**: `GOOD`, `MOULD_RISK`, `SPOILED`, `POOR_FERMENTATION`
    - **Disclaimer**: Visual screening only. Laboratory confirmation is required for fungal toxins and mycotoxins.
    """
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image file provided."
        )

    try:
        content = await file.read()
        return visual_mould_service.predict_silage_visual(content)
    except ImageProcessingError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message
        )
    except Exception as e:
        logger.error(f"Unexpected error in silage visual screening: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Silage visual screening failed: {str(e)}"
        )
