"""
Test Suite for Source-Backed Indian Veterinary Vaccine Pricing
Validates authoritative price sources, NADCP free vaccination (₹0),
per-dose mathematical calculations, unavailable price fallbacks, and stale price detection.
"""

from datetime import date, timedelta
from fastapi.testclient import TestClient

from backend.main import app
from backend.app.schemas.user_farm_cattle import Cattle
from backend.app.services.vaccination_service import (
    vaccination_service,
    calculate_per_dose_price,
    is_price_stale,
    UNAVAILABLE_PRICE_MESSAGE,
    STANDARD_VACCINATION_SCHEDULE
)

client = TestClient(app)


def test_source_backed_price_configuration():
    """Validates that every vaccine price record contains authoritative source details, date, and price type."""
    sched = vaccination_service.get_vaccination_schedule_config()

    for key, item in sched.items():
        assert "recommended_vaccine" in item
        assert "price_type" in item
        assert item["price_type"] in [
            "GOVERNMENT_PROGRAMME_FREE",
            "GOVERNMENT_PROCUREMENT",
            "MANUFACTURER_LIST",
            "RETAIL_MARKET",
            "UNAVAILABLE"
        ]

        if item["price_type"] != "UNAVAILABLE":
            assert item.get("source_name") is not None
            assert len(item["source_name"]) > 5
            assert item.get("source_date") is not None
            assert "source_url" in item
            assert item.get("calculated_per_dose_inr") is not None
            assert item["calculated_per_dose_inr"] > 0
        else:
            assert item.get("farmer_cost_display") == UNAVAILABLE_PRICE_MESSAGE


def test_price_per_dose_calculation():
    """Validates that price per dose is calculated strictly as Total Pack Price / Pack Size in Doses."""
    # Standard pack calculation
    assert calculate_per_dose_price(1800.0, 100) == 18.00
    assert calculate_per_dose_price(650.0, 50) == 13.00
    assert calculate_per_dose_price(600.0, 50) == 12.00
    assert calculate_per_dose_price(220.0, 10) == 22.00
    assert calculate_per_dose_price(1200.0, 100) == 12.00
    assert calculate_per_dose_price(400.0, 50) == 8.00

    # Edge cases / None values
    assert calculate_per_dose_price(None, 50) is None
    assert calculate_per_dose_price(500.0, None) is None
    assert calculate_per_dose_price(500.0, 0) is None


def test_free_government_vaccination_nadcp():
    """
    Validates that government-funded vaccines (FMD, Brucellosis) under NADCP
    represent farmer cost as ₹0 only with government programme eligibility note.
    """
    fmd_detail = vaccination_service.get_vaccine_price_detail("FMD")
    assert fmd_detail.price_type == "GOVERNMENT_PROGRAMME_FREE"
    assert fmd_detail.farmer_cost_inr == 0.0
    assert "₹0" in fmd_detail.farmer_cost_display
    assert "Government Programme / Farmer Cost" in fmd_detail.farmer_cost_display
    assert fmd_detail.source_url == "https://dahd.nic.in/schemes/programmes/nadcp"
    assert fmd_detail.eligibility_notes is not None
    assert "National Animal Disease Control Programme" in fmd_detail.eligibility_notes

    # Institutional procurement benchmark is preserved separately and labeled as Government Procurement Price
    assert fmd_detail.calculated_per_dose_inr == 18.0
    assert "Government Procurement Price" in fmd_detail.procurement_cost_display
    assert "Market Price" not in fmd_detail.procurement_cost_display

    bruc_detail = vaccination_service.get_vaccine_price_detail("BRUCELLOSIS")
    assert bruc_detail.price_type == "GOVERNMENT_PROGRAMME_FREE"
    assert bruc_detail.farmer_cost_inr == 0.0
    assert "₹0" in bruc_detail.farmer_cost_display
    assert "Government Programme / Farmer Cost" in bruc_detail.farmer_cost_display
    assert bruc_detail.calculated_per_dose_inr == 22.0
    assert "Government Procurement Price" in bruc_detail.procurement_cost_display
    assert bruc_detail.eligibility_notes is not None


def test_procurement_prices_not_labeled_as_market_prices():
    """
    Validates that procurement benchmarks (₹18 FMD, ₹22 Brucellosis, ₹13 HS, ₹12 BQ)
    are strictly labeled as Government Procurement Price and never as Market/Retail Price.
    """
    hs = vaccination_service.get_vaccine_price_detail("HS")
    assert hs.calculated_per_dose_inr == 13.0
    assert "Government Procurement Price" in hs.procurement_cost_display
    assert "Market Price" not in hs.procurement_cost_display
    assert "Retail price unavailable" in hs.retail_price_display

    bq = vaccination_service.get_vaccine_price_detail("BQ")
    assert bq.calculated_per_dose_inr == 12.0
    assert "Government Procurement Price" in bq.procurement_cost_display
    assert "Market Price" not in bq.procurement_cost_display
    assert "Retail price unavailable" in bq.retail_price_display


def test_unavailable_price_fallback():
    """
    Validates that when an exact retail price cannot be verified,
    the system outputs the exact required unavailable price string without fabricated numbers.
    """
    # Theileriosis configured as unavailable due to specialized cold chain
    theil_detail = vaccination_service.get_vaccine_price_detail("THEILERIOSIS")
    assert theil_detail.price_type == "UNAVAILABLE"
    assert theil_detail.farmer_cost_inr is None
    assert theil_detail.farmer_cost_display == UNAVAILABLE_PRICE_MESSAGE
    assert theil_detail.cost_per_dose_display == UNAVAILABLE_PRICE_MESSAGE

    # Nonexistent condition fallback
    unknown_detail = vaccination_service.get_vaccine_price_detail("UNKNOWN_CONDITION")
    assert unknown_detail.price_type == "UNAVAILABLE"
    assert unknown_detail.farmer_cost_display == UNAVAILABLE_PRICE_MESSAGE


def test_stale_price_detection():
    """Validates that price records older than threshold (e.g. 2 years) are flagged as stale."""
    recent_date = (date.today() - timedelta(days=100)).isoformat()
    old_date = (date.today() - timedelta(days=800)).isoformat()

    assert not is_price_stale(recent_date, threshold_days=730)
    assert is_price_stale(old_date, threshold_days=730)
    assert is_price_stale("2020-01-01", threshold_days=730)
    assert not is_price_stale(None)


def test_vaccination_api_returns_source_and_date():
    """Verifies that the /api/v1/cattle/{tag_id}/vaccinations endpoint returns source, date, and price type."""
    user_id = "pricing_tester"
    tag_id = "COW-PRICE-01"

    # Register cow
    headers = {"Authorization": f"Bearer {user_id}", "X-User-ID": user_id}
    client.post(
        "/api/v1/cattle",
        json={
            "tag_id": tag_id,
            "species": "Cattle",
            "gender": "Female",
            "breed": "Gir",
            "body_weight_kg": 390.0,
            "age_months": 24.0
        },
        headers=headers
    )

    res = client.get(f"/api/v1/cattle/{tag_id}/vaccinations", headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert len(data) >= 5

    fmd_item = next(item for item in data if item["disease_target"] == "FMD")
    assert fmd_item["price_type"] == "GOVERNMENT_PROGRAMME_FREE"
    assert "₹0" in fmd_item["farmer_cost_display"]
    assert fmd_item["source_name"] is not None
    assert fmd_item["source_date"] is not None
    assert fmd_item["source_url"] == "https://dahd.nic.in/schemes/programmes/nadcp"

    hs_item = next(item for item in data if item["disease_target"] == "HS")
    assert hs_item["price_type"] == "GOVERNMENT_PROCUREMENT"
    assert hs_item["calculated_per_dose_inr"] == 13.0
    assert hs_item["source_name"] is not None


def test_disease_diagnosis_returns_source_backed_pricing():
    """Verifies that disease prediction endpoint includes source-backed pricing details and source citation."""
    from PIL import Image
    import io

    # Create dummy image for diagnosis test
    img = Image.new("RGB", (300, 300), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    image_bytes = buf.getvalue()

    res = client.post(
        "/api/v1/predict/disease",
        files={"file": ("bovine.jpg", image_bytes, "image/jpeg")}
    )
    assert res.status_code == 200
    pred = res.json()

    assert "recommended_vaccine" in pred
    assert "estimated_cost" in pred
    assert "veterinary_disclaimer" in pred
    # If diagnosed as FMD, LSD, or Normal, source_name and price_type must be present
    if pred["predicted_class"] in ["FMD", "LSD", "Normal"]:
        assert pred["price_type"] is not None
        assert pred["source_name"] is not None
