"""
Field Nutrition & Least-Cost Ration Optimization Endpoints
Deterministic API endpoints grounded in ICAR Standards and ICAR-NIANP Feed Composition
"""

from typing import Any, Dict, List
from fastapi import APIRouter, status, HTTPException

from backend.app.schemas.nutrition import (
    NutritionRecommendationRequest,
    NutritionRecommendationResponse
)
from backend.app.services.nutrition_engine import nutrition_engine

router = APIRouter(prefix="/nutrition", tags=["Field Nutrition & Ration Recommendation"])


@router.post(
    "/recommend",
    response_model=NutritionRecommendationResponse,
    status_code=status.HTTP_200_OK,
    summary="Compute Least-Cost Balanced Ration (ICAR Standards)"
)
async def recommend_ration(payload: NutritionRecommendationRequest):
    """
    Accepts bovine characteristics (body weight, daily milk yield, milk fat %, breed, species)
    and available feeds/prices, and computes an exact least-cost balanced daily ration
    using Linear Programming (scipy.optimize) conforming to ICAR-2013/2024 standards.
    """
    response = nutrition_engine.optimize_ration(payload)
    if response.status == "invalid_parameters":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=response.message
        )
    return response


@router.get(
    "/feeds",
    response_model=List[Dict[str, Any]],
    status_code=status.HTTP_200_OK,
    summary="List All Verified ICAR-NIANP Feed Ingredients"
)
async def get_all_feeds():
    """
    Returns the comprehensive catalog of 38 verified Indian feed ingredients
    including proximate chemical analysis (Dry Matter, Crude Protein, TDN, Fibre, Minerals)
    and indicative market prices.
    """
    return nutrition_engine.get_all_feeds()


@router.get(
    "/standards",
    status_code=status.HTTP_200_OK,
    summary="Get ICAR Nutritional Partitioning Standards"
)
async def get_icar_standards():
    """
    Returns the official mathematical formulas and partition equations
    governing maintenance, lactation (4% FCM), and pregnancy allowances under ICAR guidelines.
    """
    return {
        "standard": "ICAR-2013 / 2024 Guidelines for Cattle and Buffalo Nutrition",
        "maintenance_equations": {
            "metabolic_body_weight": "W^0.75 (kg)",
            "indigenous_zebu_cattle": {
                "dmi_kg_per_day": "0.022 * W",
                "tdn_kg_per_day": "0.034 * W^0.75",
                "crude_protein_g_per_day": "4.2 * W^0.75",
                "calcium_g_per_day": "0.050 * W",
                "phosphorus_g_per_day": "0.035 * W"
            },
            "crossbred_cattle": {
                "dmi_kg_per_day": "0.024 * W",
                "tdn_kg_per_day": "0.036 * W^0.75",
                "crude_protein_g_per_day": "4.5 * W^0.75",
                "calcium_g_per_day": "0.055 * W",
                "phosphorus_g_per_day": "0.038 * W"
            },
            "buffalo": {
                "dmi_kg_per_day": "0.023 * W",
                "tdn_kg_per_day": "0.035 * W^0.75",
                "crude_protein_g_per_day": "4.3 * W^0.75",
                "calcium_g_per_day": "0.052 * W",
                "phosphorus_g_per_day": "0.036 * W"
            }
        },
        "lactation_equations_4pct_fcm": {
            "fat_corrected_milk_formula": "(0.4 + 0.15 * Fat%) * Milk_Yield_kg",
            "cattle_per_kg_fcm": {
                "tdn_kg": 0.320,
                "crude_protein_g": 85.0,
                "calcium_g": 3.0,
                "phosphorus_g": 2.0,
                "dmi_kg_per_kg_milk": 0.33
            },
            "buffalo_per_kg_fcm": {
                "tdn_kg": 0.340,
                "crude_protein_g": 92.0,
                "calcium_g": 3.5,
                "phosphorus_g": 2.4,
                "dmi_kg_per_kg_milk": 0.36
            }
        },
        "pregnancy_allowance_last_trimester": {
            "dmi_kg_per_day": 1.0,
            "tdn_kg_per_day": 1.20,
            "crude_protein_g_per_day": 250.0,
            "calcium_g_per_day": 12.0,
            "phosphorus_g_per_day": 8.0
        }
    }
