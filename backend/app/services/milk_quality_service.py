"""
NIR Milk Quality Spectroscopy Inference Service
"""

import numpy as np
import pandas as pd
from backend.app.services.model_loader import model_loader
from backend.app.schemas.milk_quality import (
    MilkNIRSpectralInput,
    MilkFatPredictionResponse,
    MilkProteinPredictionResponse
)
from backend.app.core.exceptions import ModelInferenceError


def interpret_fat(fat_percent: float) -> str:
    """Classify milk fat content into standard commercial tiers."""
    if fat_percent >= 4.5:
        return "High Fat (Whole/Buffalo Milk Grade)"
    elif fat_percent >= 3.5:
        return "Standard Full Cream Milk"
    elif fat_percent >= 2.0:
        return "Toned / Semi-Skimmed Milk"
    else:
        return "Low Fat / Skimmed Milk"


class MilkQualityInferenceService:
    """Service handling NIR spectral spectroscopy inference for milk quality."""

    def _prepare_matrix(self, pipeline, spectra: list):
        """Prepare input matching pipeline feature names if present."""
        X_arr = np.array(spectra, dtype=np.float64).reshape(1, -1)
        scaler = pipeline.named_steps.get("scaler")
        if scaler is not None and hasattr(scaler, "feature_names_in_"):
            return pd.DataFrame(X_arr, columns=scaler.feature_names_in_)
        return X_arr

    def predict_fat(self, input_data: MilkNIRSpectralInput) -> MilkFatPredictionResponse:
        """
        Estimate milk fat percentage from 1024 NIR absorbance channels.
        """
        pipeline = model_loader.load_joblib_pipeline("milk_quality_fat")
        try:
            X = self._prepare_matrix(pipeline, input_data.spectra)
            predicted_fat = float(pipeline.predict(X)[0])
            predicted_fat = max(0.0, predicted_fat)

            return MilkFatPredictionResponse(
                sample_id=input_data.sample_id or "UNKNOWN",
                predicted_fat_percentage=round(predicted_fat, 2),
                spectral_channels_used=len(input_data.spectra),
                pca_components=95,
                model_r2_score=0.8792,
                interpretation=interpret_fat(predicted_fat)
            )
        except Exception as e:
            raise ModelInferenceError("milk_quality_fat", str(e))

    def predict_protein_experimental(self, input_data: MilkNIRSpectralInput) -> MilkProteinPredictionResponse:
        """
        Estimate milk protein percentage (Experimental model, enabled only if ENABLE_EXPERIMENTAL_MODELS=true).
        """
        pipeline = model_loader.load_joblib_pipeline("milk_quality_protein")
        try:
            X = self._prepare_matrix(pipeline, input_data.spectra)
            predicted_prot = float(pipeline.predict(X)[0])
            predicted_prot = max(0.0, predicted_prot)

            return MilkProteinPredictionResponse(
                sample_id=input_data.sample_id or "UNKNOWN",
                predicted_protein_percentage=round(predicted_prot, 2),
                model_r2_score=0.4305
            )
        except Exception as e:
            raise ModelInferenceError("milk_quality_protein", str(e))


milk_quality_service = MilkQualityInferenceService()

