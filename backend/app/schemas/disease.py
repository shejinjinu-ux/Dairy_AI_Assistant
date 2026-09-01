"""
Bovine Disease Diagnosis Schemas
"""

from typing import Dict, Optional, Any
from pydantic import BaseModel, Field

from backend.app.schemas.user_farm_cattle import VaccinePriceDetail


class DiseasePredictionResponse(BaseModel):
    """Output schema for cattle disease diagnosis with vaccine and timing recommendations."""
    predicted_class: str = Field(..., description="Predicted health condition: 'FMD', 'IBK', 'LSD', or 'Normal'")
    confidence: float = Field(..., description="Prediction probability confidence between 0.0 and 1.0")
    confidence_percentage: float = Field(..., description="Confidence expressed as a percentage (0-100%)")
    is_disease_detected: bool = Field(..., description="True if condition is FMD, IBK, or LSD; False if Normal")
    disease_name_full: str = Field(..., description="Expanded clinical condition name")
    explanation: str = Field(..., description="Clinical presentation, pathology, and symptoms explanation")
    recommended_vaccine: str = Field(..., description="Recommended preventive or therapeutic vaccine where applicable")
    vaccination_timing: str = Field(..., description="Recommended administration timing and schedule")
    estimated_cost: str = Field(..., description="Source-backed vaccine cost representation in INR")
    brand_name: Optional[str] = Field(default=None, description="Brand name if applicable.")
    manufacturer: Optional[str] = Field(default=None, description="Manufacturer name if applicable.")
    price_type: Optional[str] = Field(default=None, description="Price type classification.")
    farmer_cost_display: Optional[str] = Field(default=None, description="Cost to farmer representation.")
    calculated_per_dose_inr: Optional[float] = Field(default=None, description="Calculated single dose price.")
    procurement_cost_display: Optional[str] = Field(default=None, description="Government Procurement Price.")
    retail_price_display: Optional[str] = Field(default=None, description="Private Retail Price.")
    source_name: Optional[str] = Field(default=None, description="Authoritative source citation.")
    source_url: Optional[str] = Field(default=None, description="Authoritative source URL.")
    source_date: Optional[str] = Field(default=None, description="Source date or contract year.")
    is_stale: bool = Field(default=False, description="Staleness status.")
    eligibility_notes: Optional[str] = Field(default=None, description="Government programme eligibility details.")
    price_detail: Optional[VaccinePriceDetail] = Field(default=None, description="Full structured price detail.")
    probabilities: Dict[str, float] = Field(..., description="Probabilities across all 4 classes")
    model_version: str = Field(default="efficientnet_b3", description="Backbone vision architecture")
    device_used: str = Field(..., description="Inference compute device (cpu/cuda)")
    disclaimer: str = Field(
        default="Screening tool for early detection. Consult a certified veterinarian for clinical treatment.",
        description="Clinical diagnostic disclaimer"
    )
    veterinary_disclaimer: str = Field(
        default="Estimated information only. Consult a qualified veterinarian for diagnosis and vaccination decisions.",
        description="Mandatory veterinary pricing and vaccine disclaimer"
    )
