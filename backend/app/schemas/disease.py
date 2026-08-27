"""
Bovine Disease Diagnosis Schemas
"""

from typing import Dict, Optional
from pydantic import BaseModel, Field


class DiseasePredictionResponse(BaseModel):
    """Output schema for cattle disease diagnosis."""
    predicted_class: str = Field(..., description="Predicted health condition: 'FMD', 'IBK', 'LSD', or 'Normal'")
    confidence: float = Field(..., description="Prediction probability confidence between 0.0 and 1.0")
    confidence_percentage: float = Field(..., description="Confidence expressed as a percentage (0-100%)")
    is_disease_detected: bool = Field(..., description="True if condition is FMD, IBK, or LSD; False if Normal")
    disease_name_full: str = Field(..., description="Expanded clinical condition name")
    probabilities: Dict[str, float] = Field(..., description="Probabilities across all 4 classes")
    model_version: str = Field(default="efficientnet_b3", description="Backbone vision architecture")
    device_used: str = Field(..., description="Inference compute device (cpu/cuda)")
    disclaimer: str = Field(
        default="Screening tool for early detection. Consult a certified veterinarian for clinical treatment.",
        description="Clinical diagnostic disclaimer"
    )
