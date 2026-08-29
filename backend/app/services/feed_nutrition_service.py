"""
Feed Nutrition Multi-Target Inference Service
"""

import pandas as pd
from typing import Dict
from backend.app.services.model_loader import model_loader
from backend.app.schemas.feed_nutrition import (
    FeedNutritionInput,
    NutritionalTargetPrediction,
    FeedNutritionMultiTargetResponse
)
from backend.app.services.feed_scoring import (
    g_per_kg_to_percentage,
    calculate_feed_quality_score
)
from backend.app.core.exceptions import ModelInferenceError, ModelNotFoundError


TARGET_METADATA = {
    "crude_protein": {
        "model_key": "feed_crude_protein",
        "display_name": "Crude Protein",
        "unit": "g/kg DM",
        "r2": 0.9317
    },
    "dry_matter": {
        "model_key": "feed_dry_matter",
        "display_name": "Dry Matter",
        "unit": "g/kg",
        "r2": 0.9735
    },
    "crude_fibre": {
        "model_key": "feed_crude_fibre",
        "display_name": "Crude Fibre",
        "unit": "g/kg DM",
        "r2": 0.8206
    },
    "ndf": {
        "model_key": "feed_ndf",
        "display_name": "Neutral Detergent Fibre (NDF)",
        "unit": "g/kg DM",
        "r2": 0.8556
    },
    "adf": {
        "model_key": "feed_adf",
        "display_name": "Acid Detergent Fibre (ADF)",
        "unit": "g/kg DM",
        "r2": 0.8513
    },
    "adl": {
        "model_key": "feed_adl",
        "display_name": "Acid Detergent Lignin (ADL)",
        "unit": "g/kg DM",
        "r2": 0.7649
    },
    "starch": {
        "model_key": "feed_starch",
        "display_name": "Starch",
        "unit": "g/kg DM",
        "r2": 0.9588
    },
}


class FeedNutritionInferenceService:
    """Service handling multi-target feed nutrition regression with dynamic scoring."""

    def _prepare_dataframe(self, input_data: FeedNutritionInput) -> pd.DataFrame:
        """Convert input model into DataFrame matching pipeline column names."""
        raw_dict = input_data.model_dump(by_alias=True)
        return pd.DataFrame([raw_dict])

    def predict_target(self, target_name: str, input_data: FeedNutritionInput) -> NutritionalTargetPrediction:
        """Predict a single nutritional target with unit conversion to percentage."""
        clean_target = target_name.lower().replace("-", "_").replace(" ", "_")
        meta = TARGET_METADATA.get(clean_target)
        if not meta:
            valid_targets = list(TARGET_METADATA.keys())
            raise ModelNotFoundError(f"Unknown feed target '{target_name}'. Available: {valid_targets}")

        model_key = meta["model_key"]
        pipeline = model_loader.load_joblib_pipeline(model_key)
        try:
            df = self._prepare_dataframe(input_data)
            predicted_val = float(pipeline.predict(df)[0])
            predicted_val = max(0.0, predicted_val)
            pct_val = g_per_kg_to_percentage(predicted_val)

            return NutritionalTargetPrediction(
                target_name=meta["display_name"],
                predicted_value=round(predicted_val, 2),
                percentage_value=pct_val,
                unit=meta["unit"],
                model_r2=meta["r2"]
            )
        except Exception as e:
            raise ModelInferenceError(model_key, str(e))

    def predict_all(self, input_data: FeedNutritionInput) -> FeedNutritionMultiTargetResponse:
        """Predict all 7 nutritional fractions simultaneously and compute dynamic quality score."""
        predictions: Dict[str, NutritionalTargetPrediction] = {}
        for target_key in TARGET_METADATA.keys():
            predictions[target_key] = self.predict_target(target_key, input_data)

        # Dynamic score computation based on actual model outputs
        dm_val = predictions["dry_matter"].predicted_value
        cp_val = predictions["crude_protein"].predicted_value
        cf_val = predictions["crude_fibre"].predicted_value
        ndf_val = predictions["ndf"].predicted_value
        adf_val = predictions["adf"].predicted_value
        adl_val = predictions["adl"].predicted_value
        starch_val = predictions["starch"].predicted_value

        score, status_tier, why_list, action_list = calculate_feed_quality_score(
            feed_category=input_data.feed_category,
            dry_matter_g_per_kg=dm_val,
            crude_protein_g_per_kg_dm=cp_val,
            crude_fibre_g_per_kg_dm=cf_val,
            ndf_g_per_kg_dm=ndf_val,
            adf_g_per_kg_dm=adf_val,
            adl_g_per_kg_dm=adl_val,
            starch_g_per_kg_dm=starch_val
        )

        return FeedNutritionMultiTargetResponse(
            feed_category=input_data.feed_category,
            detailed_feed_category=input_data.detailed_feed_category,
            predictions=predictions,
            quality_score=score,
            quality_status=status_tier,
            why=why_list,
            recommended_action=action_list
        )


feed_nutrition_service = FeedNutritionInferenceService()
