"""
Unit & Regression Tests for Cattle Body Temperature Removal
Verifies zero cattle body temperature fields across schemas, endpoints, and chat services.
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.app.schemas.user_farm_cattle import Cattle, CattleCreateRequest, CattleUpdateRequest
from backend.app.schemas.chat import ChatRequest
from backend.app.schemas.nutrition import NutritionRecommendationRequest

client = TestClient(app)


def test_cattle_schema_has_no_body_temperature_field():
    """Verifies that Cattle Pydantic schemas do not have body_temperature or temperature fields."""
    cattle_fields = Cattle.model_fields.keys()
    create_fields = CattleCreateRequest.model_fields.keys()
    update_fields = CattleUpdateRequest.model_fields.keys()

    for field_name in ["body_temperature", "body_temp", "cattle_temperature", "rectal_temperature", "temperature"]:
        assert field_name not in cattle_fields, f"Field '{field_name}' should not exist in Cattle schema."
        assert field_name not in create_fields, f"Field '{field_name}' should not exist in CattleCreateRequest schema."
        assert field_name not in update_fields, f"Field '{field_name}' should not exist in CattleUpdateRequest schema."


def test_nutrition_schema_cattle_context_has_no_body_temperature():
    """Verifies that nutrition request model has zero body temperature fields."""
    nutrition_fields = NutritionRecommendationRequest.model_fields.keys()
    assert "body_temperature" not in nutrition_fields
    assert "body_temp" not in nutrition_fields


def test_chat_system_does_not_request_body_temperature():
    """Verifies that chat prompts do not ask farmers for body temperature."""
    res = client.post(
        "/api/v1/chat",
        json={"message": "My cow gives 15 litres of milk and weighs 400 kg. What should I feed?"}
    )
    assert res.status_code == 200
    reply_text = res.json()["reply"].lower()
    assert "body temperature" not in reply_text
    assert "body temp" not in reply_text
