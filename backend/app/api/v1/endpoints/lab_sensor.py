"""
Sensor & Laboratory-Ready Diagnostic Endpoints
"""

from fastapi import APIRouter, status
from backend.app.schemas.lab_sensor import (
    ContaminationScreenInput,
    ContaminationScreenResponse,
    MycotoxinDonInput,
    MycotoxinDonResponse,
    UreaSilicaScreenInput,
    UreaSilicaScreenResponse
)
from backend.app.services.sensor_lab_service import sensor_lab_service

router = APIRouter(prefix="/sensor-lab", tags=["Sensor & Laboratory Telemetry"])


@router.post(
    "/contamination-screen",
    response_model=ContaminationScreenResponse,
    status_code=status.HTTP_200_OK,
    summary="Screen Milk Physical Integrity & Adulteration Telemetry"
)
async def screen_contamination(payload: ContaminationScreenInput):
    """
    Accepts inline physical sensor telemetry (freezing point depression, electrical conductivity,
    pH, optical turbidity, raw SCC) and analyzes for physical anomalies, extraneous water adulteration,
    and subclinical mastitis risk.

    **Note:** Operates on physical sensor boundaries and does not hallucinate fake wet-lab chemistry.
    """
    return sensor_lab_service.screen_contamination(payload)


@router.post(
    "/mycotoxin-don",
    response_model=MycotoxinDonResponse,
    status_code=status.HTTP_200_OK,
    summary="Screen Feed for Deoxynivalenol (DON) Mycotoxin (Experimental)"
)
async def screen_mycotoxin_don(payload: MycotoxinDonInput):
    """
    Screen corn/cereal feed samples for Deoxynivalenol (DON ppm) using the experimental
    XGBoost model (R² = 0.4372).

    **Requirement:** Requires `ENABLE_EXPERIMENTAL_MODELS=true` in backend configuration.
    Official regulatory actions require certified HPLC / LC-MS laboratory confirmation.
    """
    return sensor_lab_service.screen_mycotoxin_don(payload)


@router.post(
    "/urea-silica-screen",
    response_model=UreaSilicaScreenResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest & Validate Laboratory Screening for Urea & Silica"
)
async def screen_urea_silica(payload: UreaSilicaScreenInput):
    """
    Ingests calibrated mid-IR spectral peak scan or laboratory wet-chemistry titration data
    for sample matrices (Milk or Feed).
    """
    return sensor_lab_service.screen_urea_silica(payload)
