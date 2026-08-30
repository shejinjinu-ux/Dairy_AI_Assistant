"""
Visual Mould & Spoilage Screening Schemas (Method 1)
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class VisualIndicators(BaseModel):
    """Computer vision surface and texture anomaly indicators."""
    surface_discolouration_index: float = Field(default=0.0, description="Normalized discolouration variance (0.0 - 1.0)")
    dark_or_mould_cluster_spots: bool = Field(default=False, description="Whether localized fungal hyphae/mould cluster patches were detected")
    texture_roughness_score: float = Field(default=0.0, description="Normalized high-frequency surface texture roughness")
    white_grey_hyphae_indicators: bool = Field(default=False, description="Presence of white/grey/green/blue-grey surface fungal patterns")


class FeedVisualScreeningResponse(BaseModel):
    """Output schema for visual feed mould & spoilage risk screening."""
    success: bool = Field(default=True, description="Screening status")
    error_type: Optional[str] = Field(default=None, description="Error type if invalid (e.g. 'INVALID_IMAGE')")
    classification: Optional[str] = Field(default=None, description="Domain classification (e.g. 'NOT_FEED_OR_SILAGE' or 'FEED_SAMPLE')")
    message: Optional[str] = Field(default=None, description="Farmer-friendly validation feedback message")
    predicted_class: Optional[str] = Field(default=None, description="Predicted visual condition: 'GOOD', 'MOULD_RISK', or 'SPOILED'")
    confidence: Optional[float] = Field(default=None, description="Prediction confidence (0.0 - 1.0)")
    confidence_percentage: Optional[float] = Field(default=None, description="Confidence expressed as percentage (0.0 - 100.0%)")
    risk_level: Optional[str] = Field(default=None, description="Risk tier: 'LOW', 'MEDIUM', 'HIGH', or 'CRITICAL'")
    screening_type: str = Field(default="visual_mould_screening", description="Designation of screening mechanism")
    probabilities: Optional[Dict[str, float]] = Field(default=None, description="Probability distribution across target classes")
    visual_indicators: Optional[VisualIndicators] = Field(default=None, description="Extracted surface visual indicators")
    why: List[str] = Field(default_factory=list, description="Key visual reasoning points")
    recommended_action: List[str] = Field(default_factory=list, description="Actionable farmer management recommendations")
    disclaimer: str = Field(
        default="Visual screening only. Laboratory confirmation is required for fungal toxins and mycotoxins.",
        description="Scientific screening limitation disclaimer"
    )


class SilageVisualScreeningResponse(BaseModel):
    """Output schema for visual silage aerobic spoilage and mould screening."""
    success: bool = Field(default=True, description="Screening status")
    error_type: Optional[str] = Field(default=None, description="Error type if invalid (e.g. 'INVALID_IMAGE')")
    classification: Optional[str] = Field(default=None, description="Domain classification (e.g. 'NOT_FEED_OR_SILAGE' or 'SILAGE_SAMPLE')")
    message: Optional[str] = Field(default=None, description="Farmer-friendly validation feedback message")
    predicted_class: Optional[str] = Field(default=None, description="Predicted visual condition: 'GOOD', 'MOULD_RISK', 'SPOILED', or 'POOR_FERMENTATION'")
    confidence: Optional[float] = Field(default=None, description="Prediction confidence (0.0 - 1.0)")
    confidence_percentage: Optional[float] = Field(default=None, description="Confidence expressed as percentage (0.0 - 100.0%)")
    risk_level: Optional[str] = Field(default=None, description="Risk tier: 'LOW', 'MEDIUM', 'HIGH', or 'CRITICAL'")
    screening_type: str = Field(default="visual_silage_spoilage_screening", description="Designation of screening mechanism")
    probabilities: Optional[Dict[str, float]] = Field(default=None, description="Probability distribution across target classes")
    visual_indicators: Optional[VisualIndicators] = Field(default=None, description="Extracted surface visual indicators")
    why: List[str] = Field(default_factory=list, description="Key visual reasoning points")
    recommended_action: List[str] = Field(default_factory=list, description="Actionable farmer management recommendations")
    disclaimer: str = Field(
        default="Visual screening only. Laboratory confirmation is required for fungal toxins and mycotoxins.",
        description="Scientific screening limitation disclaimer"
    )
