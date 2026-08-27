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
    """Service handling multi-target feed nutrition regression."""

    def _prepare_dataframe(self, input_data: FeedNutritionInput) -> pd.DataFrame:
        """Convert input model into DataFrame matching pipeline column names."""
        raw_dict = input_data.model_dump(by_alias=True)
        return pd.DataFrame([raw_dict])

    def predict_target(self, target_name: str, input_data: FeedNutritionInput) -> NutritionalTargetPrediction:
        """Predict a single nutritional target."""
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

            return NutritionalTargetPrediction(
                target_name=meta["display_name"],
                predicted_value=round(predicted_val, 2),
                unit=meta["unit"],
                model_r2=meta["r2"]
            )
        except Exception as e:
            raise ModelInferenceError(model_key, str(e))

    def predict_all(self, input_data: FeedNutritionInput) -> FeedNutritionMultiTargetResponse:
        """Predict all 7 nutritional fractions simultaneously."""
        predictions: Dict[str, NutritionalTargetPrediction] = {}
        for target_key in TARGET_METADATA.keys():
            predictions[target_key] = self.predict_target(target_key, input_data)

        return FeedNutritionMultiTargetResponse(
            feed_category=input_data.feed_category,
            detailed_feed_category=input_data.detailed_feed_category,
            predictions=predictions
        )


feed_nutrition_service = FeedNutritionInferenceService()
