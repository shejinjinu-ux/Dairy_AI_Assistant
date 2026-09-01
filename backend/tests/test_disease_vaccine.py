"""
Unit & Integration Tests for Disease Diagnosis & Vaccination Guidance
Verifies disease prediction outputs disease name, explanation, recommended vaccine,
vaccination timing, estimated cost, and veterinary disclaimer.
"""

import io
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.app.services.disease_service import disease_service

client = TestClient(app)


def _generate_test_image_bytes() -> bytes:
    """Generates valid PNG image bytes for inference testing."""
    img = Image.new("RGB", (300, 300), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_disease_service_prediction_includes_vaccine_info():
    """Verifies that disease prediction enriches response with vaccine recommendations & costs."""
    img_bytes = _generate_test_image_bytes()
    result = disease_service.predict(img_bytes)

    assert result.predicted_class in ["FMD", "IBK", "LSD", "Normal"]
    assert result.disease_name_full
    assert result.explanation
    assert result.recommended_vaccine
    assert result.vaccination_timing
    assert result.estimated_cost
    assert "Estimated information only. Consult a qualified veterinarian" in result.veterinary_disclaimer


def test_disease_api_endpoint_response_structure():
    """Tests POST /api/v1/predict/disease endpoint response payload."""
    img_bytes = _generate_test_image_bytes()
    res = client.post(
        "/api/v1/predict/disease",
        files={"file": ("test_cow.png", img_bytes, "image/png")}
    )
    assert res.status_code == 200
    data = res.json()
    assert "predicted_class" in data
    assert "disease_name_full" in data
    assert "explanation" in data
    assert "recommended_vaccine" in data
    assert "vaccination_timing" in data
    assert "estimated_cost" in data
    assert "veterinary_disclaimer" in data
