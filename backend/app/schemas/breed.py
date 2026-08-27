"""
Indian Bovine Breed Classification Schemas
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class TopBreedPrediction(BaseModel):
    """Individual top-K breed prediction entry."""
    breed: str = Field(..., description="Bovine breed name", examples=["Gir"])
    confidence: float = Field(..., description="Confidence probability for this breed (0.0 to 1.0)", examples=[0.8542])
    confidence_percentage: float = Field(..., description="Confidence percentage (0-100%)", examples=[85.42])


class BreedPredictionResponse(BaseModel):
    """Output schema for cattle & buffalo breed classification."""
    breed_status: str = Field(
        ...,
        description="Classification certainty status: 'identified' when top confidence >= threshold, or 'uncertain' when below threshold",
        examples=["identified"]
    )
    predicted_breed: Optional[str] = Field(
        None,
        description="Top predicted bovine breed name (null if confidence is below threshold)",
        examples=["Gir"]
    )
    confidence: float = Field(..., description="Top prediction confidence probability (0.0 to 1.0)", examples=[0.8542])
    confidence_percentage: float = Field(..., description="Confidence percentage (0-100%)", examples=[85.42])
    recommendation: Optional[str] = Field(
        None,
        description="Actionable recommendation when prediction confidence is uncertain",
        examples=["Upload a clearer side-profile image with the full animal visible."]
    )
    top_5_predictions: List[TopBreedPrediction] = Field(..., description="Top 5 most likely breeds with probabilities")
    total_classes_supported: int = Field(default=41, description="Number of Indian cattle/buffalo breeds supported")
    model_architecture: str = Field(default="convnext_tiny", description="Vision backbone model architecture")
    device_used: str = Field(..., description="Inference compute device (cpu/cuda)")
