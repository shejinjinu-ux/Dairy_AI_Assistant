"""
Milk Production Yield Estimation Service
"""

import pandas as pd
from backend.app.services.model_loader import model_loader
from backend.app.schemas.milk_production import (
    MilkProductionInput,
    MilkProductionPredictionResponse
)
from backend.app.core.exceptions import ModelInferenceError


class MilkProductionInferenceService:
    """Service handling milk production regression using trained XGBoost pipeline."""

    def predict(self, input_data: MilkProductionInput) -> MilkProductionPredictionResponse:
        """
        Estimate daily milk yield in litres from animal, management, and environmental parameters.
        """
        pipeline = model_loader.load_joblib_pipeline("milk_production")

        try:
            # Convert pydantic input model to pandas DataFrame matching pipeline feature names
            df = pd.DataFrame([input_data.model_dump()])

            # Predict using full pipeline (preprocessor + XGBRegressor)
            prediction = pipeline.predict(df)
            predicted_yield = float(prediction[0])

            # Non-negative physiological yield constraint
            predicted_yield = max(0.0, predicted_yield)

            return MilkProductionPredictionResponse(
                predicted_milk_yield_litres=round(predicted_yield, 2),
                target_unit="Litres / Day",
                model_r2_score=0.946193,
                features_received=len(input_data.model_dump())
            )

        except Exception as e:
            raise ModelInferenceError("milk_production", str(e))


milk_production_service = MilkProductionInferenceService()
