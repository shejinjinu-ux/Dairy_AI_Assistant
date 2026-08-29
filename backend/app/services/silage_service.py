"""
Silage Quality Classification & Fermentation Quality Index Service
"""

import pandas as pd
from backend.app.services.model_loader import model_loader
from backend.app.schemas.silage import (
    SilageInput,
    SilageQualityClassResponse,
    SilageFQIRegressionResponse,
    SilageComprehensiveResponse,
    SilageQualityScreeningResult
)
from backend.app.services.silage_scoring import evaluate_silage_screening
from backend.app.core.exceptions import ModelInferenceError


SILAGE_CLASS_LABELS = {
    "ea": "Early Acidity (High Fermentation Quality / Well Preserved)",
    "la": "Late Acidity (Slower Fermentation / Higher Secondary Fermentation Risk)"
}


def interpret_fqi(score: float) -> str:
    """Provide standard agronomic tier for Fermentation Quality Index score."""
    if score >= 90:
        return "Excellent (Superior preservation, optimal lactic acid dominance)"
    elif score >= 75:
        return "Good (Well-fermented, stable aerobic stability)"
    elif score >= 50:
        return "Fair (Moderate fermentation quality, check compaction and moisture)"
    else:
        return "Poor (High clostridial or secondary fermentation risk)"


class SilageInferenceService:
    """Service handling silage FAO quality classification and FQI score regression."""

    def _prepare_dataframe(self, input_data: SilageInput) -> pd.DataFrame:
        """Dump input model with original dotted column aliases."""
        data_dict = input_data.model_dump(by_alias=True)
        return pd.DataFrame([data_dict])

    def predict_quality_class(self, input_data: SilageInput) -> SilageQualityClassResponse:
        """
        Classify silage into FAO quality class ('ea' vs 'la').
        """
        pipeline = model_loader.load_joblib_pipeline("silage_quality_classifier")
        try:
            df = self._prepare_dataframe(input_data)
            pred_idx = pipeline.predict(df)[0]
            pred_probs = pipeline.predict_proba(df)[0]

            # 0 -> 'ea', 1 -> 'la'
            classes = ["ea", "la"]
            pred_class = classes[int(pred_idx)]
            confidence = float(pred_probs[int(pred_idx)])

            prob_map = {cls_name: round(float(prob), 4) for cls_name, prob in zip(classes, pred_probs)}

            return SilageQualityClassResponse(
                predicted_class=pred_class,
                class_label=SILAGE_CLASS_LABELS.get(pred_class, pred_class),
                confidence=round(confidence, 4),
                probabilities=prob_map,
                model_accuracy=0.9719
            )

        except Exception as e:
            raise ModelInferenceError("silage_quality_classifier", str(e))

    def predict_fqi(self, input_data: SilageInput) -> SilageFQIRegressionResponse:
        """
        Estimate Fermentation Quality Index (FQI).
        """
        pipeline = model_loader.load_joblib_pipeline("silage_fqi")
        try:
            df = self._prepare_dataframe(input_data)
            fqi_score = float(pipeline.predict(df)[0])

            return SilageFQIRegressionResponse(
                predicted_fqi=round(fqi_score, 2),
                interpretation=interpret_fqi(fqi_score),
                model_r2_score=0.9719
            )

        except Exception as e:
            raise ModelInferenceError("silage_fqi", str(e))

    def predict_comprehensive(self, input_data: SilageInput) -> SilageComprehensiveResponse:
        """
        Compute quality class, FQI score, and dynamic agronomic screening result.
        """
        quality = self.predict_quality_class(input_data)
        fqi = self.predict_fqi(input_data)

        # Dynamic Agronomic Screening Interpretation Layer
        screening_dict = evaluate_silage_screening(
            predicted_fqi=fqi.predicted_fqi,
            predicted_class=quality.predicted_class,
            class_confidence=quality.confidence,
            ph=input_data.pH,
            dm_s=input_data.dm_s,
            cp_s=input_data.cp_s,
            lactic_acid_pct=input_data.lactic_ac_s,
            acetic_acid_pct=input_data.acetic_ac_s,
            butyric_acid_pct=input_data.butyric_ac_s,
            ammonia_n_pct=input_data.ammonia_s
        )
        screening_obj = SilageQualityScreeningResult(**screening_dict)

        return SilageComprehensiveResponse(
            quality_classification=quality,
            fermentation_quality_index=fqi,
            screening_result=screening_obj
        )


silage_service = SilageInferenceService()
