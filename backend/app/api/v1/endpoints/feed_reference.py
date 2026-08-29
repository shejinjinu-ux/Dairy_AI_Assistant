"""
Feed Reference Database Endpoints
Authoritative feed nutrition calculations (Method 2) and offline caching catalog.
"""

from typing import Any, Dict
from fastapi import APIRouter, HTTPException, status

from backend.app.schemas.feed_reference import (
    FeedReferenceRequest,
    FeedReferenceResponse,
    FeedCatalogResponse
)
from backend.app.services.feed_reference_service import feed_reference_service

router = APIRouter(prefix="/feed/reference", tags=["Feed Reference Nutrition (Method 2)"])


@router.post(
    "",
    response_model=FeedReferenceResponse,
    status_code=status.HTTP_200_OK,
    summary="Compute Reference Nutrition by Feed Name + Quantity"
)
async def compute_feed_reference_nutrition(payload: FeedReferenceRequest):
    """
    Accepts a feed or forage name (e.g., 'Maize', 'Maize Silage', 'Napier Grass', 'Wheat Bran')
    and quantity in kg, and returns exact per-kg and total nutritional contributions
    grounded in ICAR-NIANP, Feedipedia, and BIS composition tables.

    - **total nutrient = per kg nutrient × quantity_kg**
    """
    try:
        return feed_reference_service.calculate_nutrition(payload)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while computing feed reference nutrition: {str(e)}"
        )


@router.get(
    "/all",
    response_model=FeedCatalogResponse,
    status_code=status.HTTP_200_OK,
    summary="Export Complete Reference Database for Offline Caching"
)
async def get_all_reference_feeds():
    """
    Returns the complete catalog of verified Indian feed ingredients with proximate
    percentages, fibre fractions, energy, and minerals.
    Frontend applications can store this payload in IndexedDB/localStorage for offline calculations.
    """
    return feed_reference_service.get_all_feeds()


@router.get(
    "/rules",
    status_code=status.HTTP_200_OK,
    summary="Get Feed Quality Scoring Rules & Benchmarks"
)
async def get_scoring_rules() -> Dict[str, Any]:
    """
    Returns the official agronomic benchmarks, dry matter bands, and crude protein
    evaluation criteria used for dynamic feed quality scoring.
    """
    return {
        "standard": "ICAR-NIANP Feed Quality & Proximate Evaluation Standards",
        "crude_protein_benchmarks_pct_dm": {
            "high_protein_concentrates": {"min": 30.0, "optimal": 38.0},
            "medium_protein_concentrates": {"min": 18.0, "optimal": 22.0},
            "cereal_grains": {"min": 8.0, "optimal": 10.0},
            "legume_green_forages": {"min": 15.0, "optimal": 18.0},
            "non_legume_green_forages": {"min": 7.0, "optimal": 9.0},
            "cereal_straws_dry_roughages": {"min": 3.0, "optimal": 4.5}
        },
        "dry_matter_optimal_ranges_pct": {
            "concentrate_grains_and_meals": {"min": 86.0, "max": 92.0},
            "silage": {"min": 30.0, "max": 38.0},
            "green_roughages": {"min": 15.0, "max": 25.0},
            "dry_roughages_straw": {"min": 88.0, "max": 94.0}
        },
        "fibre_and_lignin_limits_pct_dm": {
            "optimal_ndf_forages": {"min": 45.0, "max": 65.0},
            "maximum_adl_lignin": 6.0
        },
        "scoring_tiers": {
            "EXCELLENT": "85 - 100 Quality Score",
            "GOOD": "70 - 84 Quality Score",
            "FAIR": "50 - 69 Quality Score",
            "POOR": "< 50 Quality Score"
        },
        "disclaimer": "Reference calculation basis. Laboratory proximate testing required for batch validation."
    }


@router.get(
    "/{feed_name}",
    response_model=FeedReferenceResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Feed Reference Profile (1 kg Default)"
)
async def get_single_feed_reference(feed_name: str):
    """
    Fetch reference profile for a specific feed ingredient with default quantity = 1.0 kg.
    """
    try:
        req = FeedReferenceRequest(feed_name=feed_name, quantity_kg=1.0)
        return feed_reference_service.calculate_nutrition(req)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
