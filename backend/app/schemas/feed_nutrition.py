"""
Feed Nutrition Multi-Target Schemas
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class FeedNutritionInput(BaseModel):
    """Input features for feed proximal composition and nutritional profile."""
    model_config = ConfigDict(populate_by_name=True)

    feed_category: str = Field(
        default="Forages",
        alias="Feed-category",
        description="High-level category (e.g. Forages, Concentrates, Co-products)"
    )
    detailed_feed_category: str = Field(
        default="Maize silage",
        alias="Detailed-feed-category-INRA2018",
        description="Detailed INRA2018 feed classification"
    )

    dry_matter_g_per_kg: Optional[float] = Field(
        default=350.0,
        alias="Dry-matter-(g/kg)",
        ge=0.0,
        le=1000.0,
        description="Dry matter in g/kg"
    )
    organic_matter_g_per_kg_dm: Optional[float] = Field(
        default=920.0,
        alias="Organic-matter-(g/kg-DM)-",
        ge=0.0,
        le=1000.0,
        description="Organic matter in g/kg DM"
    )
    ash_g_per_kg_dm: Optional[float] = Field(
        default=80.0,
        alias="Ash-(g/kg-DM)",
        ge=0.0,
        le=1000.0,
        description="Total minerals/ash in g/kg DM"
    )
    crude_fibre_g_per_kg_dm: Optional[float] = Field(
        default=220.0,
        alias="Crude-fibre-(g/kg-DM)",
        ge=0.0,
        le=1000.0,
        description="Crude fibre in g/kg DM"
    )
    ndf_g_per_kg_dm: Optional[float] = Field(
        default=450.0,
        alias="NDF-(g/kg-DM)",
        ge=0.0,
        le=1000.0,
        description="Neutral Detergent Fibre in g/kg DM"
    )
    adf_g_per_kg_dm: Optional[float] = Field(
        default=260.0,
        alias="ADF-(g/kg-DM)",
        ge=0.0,
        le=1000.0,
        description="Acid Detergent Fibre in g/kg DM"
    )
    starch_g_per_kg_dm: Optional[float] = Field(
        default=280.0,
        alias="Starch-(g/kg-DM)",
        ge=0.0,
        le=1000.0,
        description="Starch in g/kg DM"
    )


class NutritionalTargetPrediction(BaseModel):
    """Prediction result for a single nutritional fraction."""
    target_name: str = Field(..., description="Nutritional variable name")
    predicted_value: float = Field(..., description="Estimated content in target unit")
    percentage_value: Optional[float] = Field(default=None, description="Calculated percentage value (g/kg divided by 10)")
    unit: str = Field(..., description="Measurement unit (e.g. g/kg or g/kg DM)")
    model_r2: float = Field(..., description="Validation R2 score")


class FeedNutritionMultiTargetResponse(BaseModel):
    """Consolidated predictions for all 7 feed nutritional targets."""
    feed_category: str = Field(..., description="Input feed category")
    detailed_feed_category: str = Field(..., description="Input detailed INRA category")
    predictions: Dict[str, NutritionalTargetPrediction] = Field(
        ...,
        description="Map of predicted nutritional parameters (crude_protein, dry_matter, crude_fibre, ndf, adf, adl, starch)"
    )
    quality_score: Optional[float] = Field(default=None, description="Dynamic composite nutritional quality score (0 - 100)")
    quality_status: Optional[str] = Field(default=None, description="Dynamic quality status tier: EXCELLENT, GOOD, FAIR, POOR")
    why: Optional[List[str]] = Field(default=None, description="Key agronomic explanation facts")
    recommended_action: Optional[List[str]] = Field(default=None, description="Actionable feed management recommendations")
    disclaimer: str = Field(
        default="Predictions apply strictly to nutritional components represented in the verified dataset.",
        description="Nutritional scope disclaimer"
    )
