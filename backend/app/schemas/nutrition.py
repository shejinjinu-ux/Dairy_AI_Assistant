"""
Pydantic Schemas for Field Nutrition & Least-Cost Ration Optimization
Grounded in ICAR-2013/2024 Standards & ICAR-NIANP Indian Feed Composition Database
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class NutritionRecommendationRequest(BaseModel):
    """Input payload for bovine ration balancing & nutrient optimization."""
    species: str = Field(default="Cattle", description="Bovine species ('Cattle' or 'Buffalo').")
    breed: Optional[str] = Field(default=None, description="Breed name (e.g. 'Gir', 'Sahiwal', 'HF_Cross', 'Murrah').")
    body_weight_kg: Optional[float] = Field(default=None, description="Live animal body weight in kg (Minimum required).")
    age_years: Optional[float] = Field(default=None, description="Animal age in years.")
    parity: Optional[int] = Field(default=None, description="Calving parity number.")
    lactation_stage: Optional[str] = Field(default=None, description="Lactation stage ('Early', 'Mid', 'Late', 'Dry').")
    days_in_milk: Optional[float] = Field(default=None, description="Days in current milk cycle.")
    daily_milk_yield_kg: Optional[float] = Field(default=None, description="Daily milk yield in kg or litres (Minimum required).")
    milk_fat_percent: Optional[float] = Field(default=None, description="Milk fat percentage (e.g. 4.0% for cows, 7.0% for buffaloes).")
    pregnancy_status: Optional[bool] = Field(default=None, description="Whether the animal is pregnant.")
    pregnancy_month: Optional[int] = Field(default=None, description="Pregnancy month (last trimester >= 7 months requires additional allowance).")

    available_feeds: Optional[List[str]] = Field(default=None, description="List of available feed names on the farm.")
    feed_prices: Optional[Dict[str, float]] = Field(default=None, description="Custom feed prices in INR per kg fresh feed.")


class FeedItemRecommendation(BaseModel):
    """Recommended feed item within a balanced daily ration."""
    feed_id: str = Field(description="Unique ICAR-NIANP feed identifier.")
    feed_name: str = Field(description="Feed ingredient name.")
    feed_category: str = Field(description="Feed category (Green Roughage, Dry Roughage, Concentrate, Byproduct, Mineral Supplement).")
    quantity_kg_per_day: float = Field(description="Optimal daily quantity to feed (kg/day fresh basis).")
    cost_per_kg_inr: float = Field(description="Cost per kg fresh feed in INR.")
    daily_cost_inr: float = Field(description="Total daily cost for this feed item in INR.")
    dm_supplied_kg: float = Field(description="Dry Matter supplied (kg/day).")
    cp_supplied_g: float = Field(description="Crude Protein supplied (g/day).")
    tdn_supplied_kg: float = Field(description="Total Digestible Nutrients supplied (kg/day).")
    calcium_supplied_g: float = Field(description="Calcium supplied (g/day).")
    phosphorus_supplied_g: float = Field(description="Phosphorus supplied (g/day).")


class NutrientRequirementsSummary(BaseModel):
    """Scientific nutrient requirements computed via ICAR partitioning formulas."""
    metabolic_body_weight_kg: float = Field(description="Metabolic Body Weight (W^0.75 in kg).")
    fat_corrected_milk_4pct_kg: float = Field(description="4% Fat-Corrected Milk yield in kg/day.")
    req_dmi_kg_per_day: float = Field(description="Total Dry Matter Intake requirement (kg/day).")
    req_tdn_kg_per_day: float = Field(description="Total Digestible Nutrients requirement (kg/day).")
    req_me_mj_per_day: float = Field(description="Metabolizable Energy requirement (MJ/day).")
    req_cp_g_per_day: float = Field(description="Crude Protein requirement (g/day).")
    req_calcium_g_per_day: float = Field(description="Calcium requirement (g/day).")
    req_phosphorus_g_per_day: float = Field(description="Phosphorus requirement (g/day).")


class NutrientBalanceItem(BaseModel):
    """Comparison between requirement and optimized supply for a nutrient."""
    required: float = Field(description="Daily requirement value.")
    supplied: float = Field(description="Daily supplied value from optimized ration.")
    unit: str = Field(description="Measurement unit (kg/day, g/day, MJ/day).")
    difference: float = Field(description="Supplied minus Required.")
    percentage_fulfilled: float = Field(description="Percentage of requirement fulfilled (%).")
    status: str = Field(description="Nutritional status: 'Balanced', 'Surplus', or 'Deficit'.")


class NutritionRecommendationResponse(BaseModel):
    """Comprehensive structured response from the Least-Cost Ration Optimization Engine."""
    success: bool = Field(description="Whether the optimization was successful.")
    is_deterministic_optimized: bool = Field(default=True, description="True (Deterministic ICAR-NIANP Linear Programming Engine).")
    status: str = Field(description="Status code: 'optimized', 'missing_parameters', or 'infeasible'.")
    message: str = Field(description="Farmer-friendly summary message.")
    animal_profile: Dict[str, Any] = Field(default_factory=dict, description="Processed animal characteristics.")
    missing_critical_parameters: List[str] = Field(default_factory=list, description="Missing inputs required for formulation.")
    nutrient_requirements: Optional[NutrientRequirementsSummary] = Field(default=None, description="ICAR scientific requirements.")
    recommended_ration: List[FeedItemRecommendation] = Field(default_factory=list, description="List of optimal feed quantities.")
    total_daily_cost_inr: float = Field(default=0.0, description="Total daily ration cost in INR.")
    nutrient_supply: Dict[str, float] = Field(default_factory=dict, description="Total nutrients supplied across all ration items.")
    nutrient_balance: Dict[str, NutrientBalanceItem] = Field(default_factory=dict, description="Detailed nutrient-by-nutrient balance.")
    warnings: List[str] = Field(default_factory=list, description="Agronomic or nutritional warnings and advice.")
