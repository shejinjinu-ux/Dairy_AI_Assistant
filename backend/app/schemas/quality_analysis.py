"""
Combined Feed and Silage Quality Analysis Schemas
Structured JSON suitable for modern frontend dashboards and mobile applications.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.app.schemas.feed_reference import FeedReferenceResponse
from backend.app.schemas.feed_nutrition import FeedNutritionMultiTargetResponse
from backend.app.schemas.silage import SilageComprehensiveResponse
from backend.app.schemas.visual_screening import (
    FeedVisualScreeningResponse,
    SilageVisualScreeningResponse
)
from backend.app.schemas.risk_analysis import ComprehensiveRiskAnalysis


class CombinedFeedAnalysisResponse(BaseModel):
    """Consolidated Feed Quality Analysis Output."""
    success: bool = Field(default=True, description="Overall analysis success flag")
    feed_name: str = Field(..., description="Target feed ingredient name")
    category: str = Field(..., description="Feed category")
    quantity_kg: Optional[float] = Field(default=None, description="Quantity analyzed for (if provided)")
    quality_score: float = Field(..., description="Dynamic composite quality score (0 - 100)")
    status: str = Field(..., description="Quality status tier: 'EXCELLENT', 'GOOD', 'FAIR', or 'POOR'")
    nutrition_reference: Optional[FeedReferenceResponse] = Field(
        default=None,
        description="Reference nutritional breakdown (Method 2)"
    )
    nutrition_ml_predictions: Optional[FeedNutritionMultiTargetResponse] = Field(
        default=None,
        description="ML multi-target proximate regression predictions (if proximal input provided)"
    )
    visual_screening: Optional[FeedVisualScreeningResponse] = Field(
        default=None,
        description="Visual mould & spoilage screening results (Method 1, if image provided)"
    )
    risk_analysis: ComprehensiveRiskAnalysis = Field(
        ...,
        description="Biological, chemical, and physical hazard risk assessment"
    )
    why: List[str] = Field(..., description="Structured evidence supporting the quality score")
    recommended_action: List[str] = Field(..., description="Actionable recommendations for feed storage and usage")
    disclaimer: str = Field(
        default="Screening and reference analysis. Laboratory chemical and microbiological assay required for definitive safety certification.",
        description="Analysis scope disclaimer"
    )


class CombinedSilageAnalysisResponse(BaseModel):
    """Consolidated Silage Quality Analysis Output."""
    success: bool = Field(default=True, description="Overall analysis success flag")
    quality_score: float = Field(..., description="Dynamic composite quality score (0 - 100)")
    status: str = Field(..., description="Quality status tier: 'GOOD', 'CAUTION', or 'UNSAFE'")
    fermentation_ml: SilageComprehensiveResponse = Field(
        ...,
        description="Production XGBoost quality classification ('ea' vs 'la') and FQI regression results"
    )
    visual_screening: Optional[SilageVisualScreeningResponse] = Field(
        default=None,
        description="Visual mould & aerobic spoilage screening results (Method 1, if image provided)"
    )
    risk_analysis: ComprehensiveRiskAnalysis = Field(
        ...,
        description="Silage biological and clostridial hazard risk assessment"
    )
    fermentation_metrics: Dict[str, Any] = Field(
        ...,
        description="Key fermentation indicators (pH, Dry Matter %, Lactic Acid %, Butyric Acid %, Ammonia-N %)"
    )
    why: List[str] = Field(..., description="Structured evidence supporting the quality score")
    recommended_action: List[str] = Field(..., description="Actionable recommendations for silo face and herd feeding management")
    disclaimer: str = Field(
        default="Silage screening analysis based on proximal fermentation metrics. Laboratory confirmation required for comprehensive microbiological analysis.",
        description="Analysis scope disclaimer"
    )
