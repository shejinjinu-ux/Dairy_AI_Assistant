"""
NIR Milk Quality Spectroscopy Schemas
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field, field_validator


class MilkNIRSpectralInput(BaseModel):
    """Input payload containing NIR spectrometer absorbance measurements."""

    spectra: List[float] = Field(
        ...,
        description="Array of NIR absorbance values across 1024 spectral channels/wavelengths"
    )
    sample_id: Optional[str] = Field(default="SAMPLE_001", description="Laboratory or farm sample ID")
    temperature_c: Optional[float] = Field(default=20.0, description="Milk sample temperature during scan (°C)")

    @field_validator("spectra")
    @classmethod
    def validate_spectral_length(cls, v: List[float]) -> List[float]:
        if len(v) != 1024:
            raise ValueError(
                f"NIR milk quality model requires exactly 1024 spectral absorbance features, but received {len(v)} features."
            )
        return v


class MilkFatPredictionResponse(BaseModel):
    """Output schema for NIR milk fat percentage regression."""
    sample_id: str = Field(..., description="Sample identifier")
    predicted_fat_percentage: float = Field(..., description="Estimated milk fat content (g/100g or %)")
    unit: str = "% (g/100g)"
    spectral_channels_used: int = Field(default=1024, description="Number of spectral features processed")
    pca_components: int = Field(default=95, description="PCA dimension reduction components")
    model_r2_score: float = Field(default=0.8792, description="Validation R2 score")
    interpretation: str = Field(..., description="Fat tier (e.g., Standard, High Fat, Low Fat)")


class MilkProteinPredictionResponse(BaseModel):
    """Output schema for experimental NIR milk protein regression."""
    sample_id: str = Field(..., description="Sample identifier")
    predicted_protein_percentage: float = Field(..., description="Estimated crude protein content (%)")
    unit: str = "% (g/100g)"
    model_status: str = "experimental"
    model_r2_score: float = Field(default=0.4305, description="Experimental validation R2 score")
    disclaimer: str = Field(
        default="Experimental model output. Validated for preliminary research screening only.",
        description="Experimental disclaimer"
    )
