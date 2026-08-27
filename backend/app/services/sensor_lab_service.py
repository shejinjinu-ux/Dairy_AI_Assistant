"""
Sensor & Laboratory-Ready Processing Service
"""

import numpy as np
import pandas as pd
from typing import List
from backend.config import settings
from backend.app.services.model_loader import model_loader
from backend.app.schemas.lab_sensor import (
    ContaminationScreenInput,
    ContaminationScreenResponse,
    MycotoxinDonInput,
    MycotoxinDonResponse,
    UreaSilicaScreenInput,
    UreaSilicaScreenResponse
)
from backend.app.core.exceptions import ModelDisabledError, ModelInferenceError


class SensorLabProcessingService:
    """
    Sensor & Laboratory Diagnostic Service.
    Enforces real telemetry and wet-chemistry laboratory validation without synthesizing fake metrics.
    """

    def screen_contamination(self, payload: ContaminationScreenInput) -> ContaminationScreenResponse:
        """
        Analyze inline physical sensor telemetry to detect water adulteration,
        abnormal acidity, or mastitis risk indicators.
        """
        params_evaluated: List[str] = []
        water_adulteration = False
        mastitis_risk = "Normal"
        acidity_anomaly = False

        # Freezing point check (Normal bovine milk freezing point: -0.525°C to -0.560°C)
        if payload.freezing_point_c is not None:
            params_evaluated.append("freezing_point_c")
            if payload.freezing_point_c > -0.505:
                water_adulteration = True

        # Inline Electrical Conductivity & SCC check
        ec = payload.electrical_conductivity_ms_cm
        scc = payload.somatic_cell_count_raw

        if ec is not None or scc is not None:
            if ec is not None:
                params_evaluated.append("electrical_conductivity_ms_cm")
            if scc is not None:
                params_evaluated.append("somatic_cell_count_raw")

            if (ec and ec > 6.8) or (scc and scc > 400.0):
                mastitis_risk = "High"
            elif (ec and ec > 6.2) or (scc and scc > 250.0):
                mastitis_risk = "Medium / Subclinical Risk"
            else:
                mastitis_risk = "Low / Normal"

        # pH range check (Fresh milk pH: 6.55 to 6.75)
        if payload.milk_ph is not None:
            params_evaluated.append("milk_ph")
            if payload.milk_ph < 6.50 or payload.milk_ph > 6.85:
                acidity_anomaly = True

        if payload.turbidity_ntu is not None:
            params_evaluated.append("turbidity_ntu")

        lab_req = water_adulteration or (mastitis_risk in {"High", "Medium / Subclinical Risk"}) or acidity_anomaly

        return ContaminationScreenResponse(
            status="sensor_telemetry_analyzed",
            is_sensor_data_valid=len(params_evaluated) > 0,
            water_adulteration_suspected=water_adulteration,
            subclinical_mastitis_risk=mastitis_risk,
            acidity_anomaly=acidity_anomaly,
            parameters_evaluated=params_evaluated,
            lab_verification_required=lab_req
        )

    def screen_mycotoxin_don(self, payload: MycotoxinDonInput) -> MycotoxinDonResponse:
        """
        Screen for Deoxynivalenol (DON) in feed ingredients using experimental XGBoost model.
        Requires ENABLE_EXPERIMENTAL_MODELS=true.
        """
        if not settings.ENABLE_EXPERIMENTAL_MODELS:
            raise ModelDisabledError(
                "mycotoxin_don",
                reason="Mycotoxin DON model is experimental (R2=0.4372). It is kept disabled by default in production. "
                       "Confirmatory wet-lab chromatography (LC-MS/MS) is required."
            )

        pipeline = model_loader.load_joblib_pipeline("mycotoxin_don")

        try:
            # Features: Protein, Fat, Moisture, Fiber, Starch, AshAI, L*(D65) SCI, a*(D65) SCI, b*(D65) SCI, harvest year, sample type, Sample Location
            feature_dict = {
                "Protein": payload.protein_percent,
                "Fat": payload.fat_percent,
                "Moisture": payload.moisture_percent,
                "Fiber": payload.fiber_percent,
                "Starch": payload.starch_percent,
                "AshAI": payload.ash_ai_percent,
                "L*(D65) SCI": payload.l_sci,
                "a*(D65) SCI": payload.a_sci,
                "b*(D65) SCI": payload.b_sci,
                "harvest year": payload.harvest_year,
                "sample type": payload.sample_type,
                "Sample Location": payload.sample_location
            }

            df = pd.DataFrame([feature_dict])
            log_pred = float(pipeline.predict(df)[0])
            don_ppm = max(0.0, float(np.expm1(log_pred)))

            return MycotoxinDonResponse(
                status="experimental_screening_completed",
                predicted_don_ppm=round(don_ppm, 3),
                fda_threshold_guideline_ppm=5.0,
                is_above_advisory_limit=bool(don_ppm >= 5.0),
                model_r2_score=0.4372,
                is_experimental=True
            )

        except Exception as e:
            if isinstance(e, ModelDisabledError):
                raise
            raise ModelInferenceError("mycotoxin_don", str(e))

    def screen_urea_silica(self, payload: UreaSilicaScreenInput) -> UreaSilicaScreenResponse:
        """
        Contract intake endpoint for Urea and Silica lab screening.
        """
        has_lab = payload.wet_chemistry_value is not None
        has_peaks = payload.spectral_absorption_peaks is not None and len(payload.spectral_absorption_peaks) > 0

        target = "Urea / Silica"
        summary = "Telemetry accepted. Awaiting calibrated mid-IR spectral peak scan or laboratory chemical titration."
        if has_lab:
            summary = f"Certified laboratory value ({payload.wet_chemistry_value}) ingested for sample matrix '{payload.sample_matrix}'."
        elif has_peaks:
            summary = f"Spectral absorption peaks ingested for sample matrix '{payload.sample_matrix}'."

        return UreaSilicaScreenResponse(
            sample_matrix=payload.sample_matrix,
            target_compound=target,
            lab_data_provided=has_lab,
            status_summary=summary
        )


sensor_lab_service = SensorLabProcessingService()
