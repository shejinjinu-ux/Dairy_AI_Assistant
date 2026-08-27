"""
Silage Quality & Fermentation Quality Index (FQI) Schemas
"""

from typing import Dict, Optional
from pydantic import BaseModel, Field, ConfigDict


class SilageInput(BaseModel):
    """32 chemical and fermentation parameters for silage assessment."""
    model_config = ConfigDict(populate_by_name=True)

    # Fresh forage parameters (.f)
    dm_f: float = Field(default=32.5, alias="dm.f", description="Fresh forage dry matter (%)")
    ash_f: float = Field(default=6.8, alias="ash.f", description="Fresh forage ash (% DM)")
    cp_f: float = Field(default=14.2, alias="cp.f", description="Fresh forage crude protein (% DM)")
    ee_f: float = Field(default=2.5, alias="ee.f", description="Fresh forage ether extract (% DM)")
    ndf_f: float = Field(default=48.0, alias="ndf.f", description="Fresh forage NDF (% DM)")
    adf_f: float = Field(default=28.5, alias="adf.f", description="Fresh forage ADF (% DM)")
    lignin_f: float = Field(default=4.2, alias="lignin.f", description="Fresh forage lignin (% DM)")
    wsc_f: float = Field(default=12.0, alias="wsc.f", description="Fresh forage water-soluble carbohydrates (% DM)")
    starch_f: float = Field(default=22.0, alias="starch.f", description="Fresh forage starch (% DM)")

    # Silaged forage parameters (.s)
    dm_s: float = Field(default=31.8, alias="dm.s", description="Silage dry matter (%)")
    ash_s: float = Field(default=7.1, alias="ash.s", description="Silage ash (% DM)")
    cp_s: float = Field(default=14.0, alias="cp.s", description="Silage crude protein (% DM)")
    ee_s: float = Field(default=2.8, alias="ee.s", description="Silage ether extract (% DM)")
    ndf_s: float = Field(default=46.5, alias="ndf.s", description="Silage NDF (% DM)")
    adf_s: float = Field(default=27.9, alias="adf.s", description="Silage ADF (% DM)")
    lignin_s: float = Field(default=4.4, alias="lignin.s", description="Silage lignin (% DM)")
    starch_s: float = Field(default=21.0, alias="starch.s", description="Silage starch (% DM)")
    wsc_s: float = Field(default=3.5, alias="wsc.s", description="Silage residual WSC (% DM)")

    # Fermentation & Acid profile
    pH: float = Field(default=3.85, ge=2.5, le=9.0, description="Silage pH")
    ammonia_s: float = Field(default=6.5, alias="ammonia.s", description="Ammonia-N (% total N)")
    glucose_s: float = Field(default=1.2, alias="glucose.s", description="Silage glucose (% DM)")
    fructose_s: float = Field(default=0.8, alias="fructose.s", description="Silage fructose (% DM)")
    mannithol_s: float = Field(default=0.5, alias="mannithol.s", description="Silage mannitol (% DM)")
    ethanol_s: float = Field(default=1.1, alias="ethanol.s", description="Silage ethanol (% DM)")
    lactic_ac_s: float = Field(default=6.2, alias="lactic.ac.s", description="Silage lactic acid (% DM)")
    acetic_ac_s: float = Field(default=1.8, alias="acetic.ac.s", description="Silage acetic acid (% DM)")
    propionic_ac_s: float = Field(default=0.2, alias="propionic.ac.s", description="Silage propionic acid (% DM)")
    butyric_ac_s: float = Field(default=0.05, alias="butyric.ac.s", description="Silage butyric acid (% DM)")

    # Physical & Losses
    dm_loss: float = Field(default=4.5, alias="dm.loss", description="Dry matter loss during ensiling (%)")
    dm_ret: float = Field(default=95.5, alias="dm.ret", description="Dry matter recovery/retention (%)")
    porosity: float = Field(default=0.42, description="Silage packing porosity")
    density_1: float = Field(default=220.0, alias="density.1", description="Silage packing bulk density (kg DM/m3)")


class SilageQualityClassResponse(BaseModel):
    """Output schema for FAO silage classification."""
    predicted_class: str = Field(..., description="Predicted class code: 'ea' (Early Acidity / High Quality) or 'la' (Late Acidity / Low Quality)")
    class_label: str = Field(..., description="Human-readable silage quality designation")
    confidence: float = Field(..., description="Probability confidence score")
    probabilities: Dict[str, float] = Field(..., description="Probability for each FAO class")
    model_accuracy: float = Field(default=0.9719, description="Validation accuracy")


class SilageFQIRegressionResponse(BaseModel):
    """Output schema for Fermentation Quality Index regression."""
    predicted_fqi: float = Field(..., description="Predicted Fermentation Quality Index score")
    interpretation: str = Field(..., description="Quality tier (e.g., Excellent, Good, Fair, Poor)")
    model_r2_score: float = Field(default=0.9719, description="Validation R2 score")


class SilageComprehensiveResponse(BaseModel):
    """Combined silage assessment output."""
    quality_classification: SilageQualityClassResponse
    fermentation_quality_index: SilageFQIRegressionResponse
