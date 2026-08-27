"""
Sensor & Laboratory-Ready Schemas
For Contamination, Mycotoxin Screening, Urea & Silica Ingestion.

STRICT PRINCIPLE:
Do NOT fabricate artificial chemical concentrations or fake lab results.
Accepts structured sensor telemetry or certified laboratory measurements,
validates integrity against physical boundaries, and returns auditable diagnostic logs.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field


class SensorTelemetryMetadata(BaseModel):
    """Metadata detailing the origin physical sensor or laboratory instrument."""
    device_id: str = Field(..., description="Unique physical sensor serial or spectrometer ID")
    sensor_type: str = Field(..., description="Sensor modality (e.g., 'NIR_Spectrometer', 'EC_Inline_Sensor', 'HPLC_Analyzer')")
    calibration_date: Optional[str] = Field(None, description="ISO timestamp of last sensor calibration")
    firmware_version: Optional[str] = Field(None, description="Sensor hardware firmware version")


class ContaminationScreenInput(BaseModel):
    """Input payload for inline milk physical integrity & contamination screening."""
    electrical_conductivity_ms_cm: Optional[float] = Field(
        None, ge=1.0, le=15.0, description="Inline electrical conductivity (mS/cm)"
    )
    freezing_point_c: Optional[float] = Field(
        None, ge=-0.65, le=-0.40, description="Cryoscopic freezing point depression (°C)"
    )
    milk_ph: Optional[float] = Field(
        None, ge=5.0, le=8.5, description="Inline pH measurement"
    )
    turbidity_ntu: Optional[float] = Field(
        None, ge=0.0, le=5000.0, description="Optical turbidity (NTU)"
    )
    somatic_cell_count_raw: Optional[float] = Field(
        None, ge=0.0, le=10000.0, description="Raw SCC counter (x10^3 cells/mL)"
    )
    sensor_metadata: Optional[SensorTelemetryMetadata] = Field(
        None, description="Physical sensor device provenance"
    )


class ContaminationScreenResponse(BaseModel):
    """Output for physical contamination & integrity screening."""
    status: str = Field(default="sensor_telemetry_analyzed", description="Diagnostic processing state")
    is_sensor_data_valid: bool = Field(..., description="True if telemetry passed physical boundary checks")
    water_adulteration_suspected: bool = Field(..., description="True if freezing point indicates extraneous water")
    subclinical_mastitis_risk: str = Field(..., description="Risk tier based on conductivity and SCC (Low/Medium/High/Normal)")
    acidity_anomaly: bool = Field(..., description="True if pH is outside normal 6.5-6.8 fresh milk range")
    parameters_evaluated: List[str] = Field(..., description="List of physical sensor parameters provided")
    lab_verification_required: bool = Field(
        default=True,
        description="True if anomalies warrant certified laboratory confirmation"
    )
    disclaimer: str = Field(
        default="Screening based on physical sensor telemetry. No synthetic lab chemistry is fabricated.",
        description="Integrity disclaimer"
    )


class MycotoxinDonInput(BaseModel):
    """Input parameters for Deoxynivalenol (DON) screening in feed (corn/grain)."""
    protein_percent: float = Field(..., ge=0.0, le=50.0, description="Feed crude protein (%)")
    fat_percent: float = Field(..., ge=0.0, le=30.0, description="Feed crude fat (%)")
    moisture_percent: float = Field(..., ge=0.0, le=40.0, description="Moisture content (%)")
    fiber_percent: float = Field(..., ge=0.0, le=50.0, description="Crude fiber (%)")
    starch_percent: float = Field(..., ge=0.0, le=90.0, description="Starch content (%)")
    ash_ai_percent: float = Field(..., ge=0.0, le=20.0, description="Acid-insoluble ash (%)")
    l_sci: float = Field(default=75.0, description="CIE L* color lightness")
    a_sci: float = Field(default=5.0, description="CIE a* color red/green coordinate")
    b_sci: float = Field(default=30.0, description="CIE b* color yellow/blue coordinate")
    harvest_year: int = Field(default=2024, ge=2000, le=2030, description="Crop harvest year")
    sample_type: str = Field(default="Corn Grain", description="Feed ingredient matrix")
    sample_location: str = Field(default="Silo A", description="Storage location")


class MycotoxinDonResponse(BaseModel):
    """Output for DON mycotoxin screening."""
    status: str = Field(..., description="Model screening status")
    predicted_don_ppm: Optional[float] = Field(None, description="Estimated DON concentration in parts-per-million (ppm)")
    fda_threshold_guideline_ppm: float = Field(default=5.0, description="FDA advisory level for dairy cattle feed (ppm)")
    is_above_advisory_limit: Optional[bool] = Field(None, description="True if estimated DON exceeds 5.0 ppm threshold")
    model_r2_score: float = Field(default=0.4372, description="Experimental benchmark R2 score")
    is_experimental: bool = True
    disclaimer: str = Field(
        default="Screening prototype. Confirmatory LC-MS/MS or ELISA laboratory testing is required for official regulatory decisions.",
        description="Regulatory compliance disclaimer"
    )


class UreaSilicaScreenInput(BaseModel):
    """Input payload for laboratory spectroscopy/chemical screening for Urea & Silica."""
    sample_matrix: str = Field(..., description="'Milk' or 'Feed'")
    spectral_absorption_peaks: Optional[Dict[str, float]] = Field(
        None, description="Key absorbance wavelengths (e.g., {'1450nm': 0.32, '1940nm': 0.85})"
    )
    wet_chemistry_value: Optional[float] = Field(
        None, description="Direct laboratory test result if available"
    )
    sensor_metadata: Optional[SensorTelemetryMetadata] = Field(None)


class UreaSilicaScreenResponse(BaseModel):
    """Contract response for Urea / Silica screening."""
    status: str = "sensor_lab_contract_active"
    sample_matrix: str
    target_compound: str
    lab_data_provided: bool
    status_summary: str
    disclaimer: str = (
        "Physical lab or calibrated NIR spectrometer telemetry is required. "
        "Synthetic chemical concentrations are strictly not generated."
    )
