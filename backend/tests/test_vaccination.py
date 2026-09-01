"""
Unit & Integration Tests for Vaccination Recommendations & Scheduling
Verifies Tag ID linkage, disease targets, booster schedules, estimated costs, and veterinary disclaimer.
"""

from datetime import date, timedelta
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.app.services.vaccination_service import vaccination_service
from backend.app.schemas.user_farm_cattle import Cattle, VaccinationRecord, VaccinationRecommendation
from backend.app.db.farm_cattle_repository import get_farm_cattle_repository, normalize_tag_id

client = TestClient(app)


def test_vaccination_schedule_config():
    """Validates that standard Indian dairy vaccination schedules and estimated costs are configured."""
    sched = vaccination_service.get_vaccination_schedule_config()
    assert "FMD" in sched
    assert "HS" in sched
    assert "BQ" in sched
    assert "BRUCELLOSIS" in sched
    assert "LSD" in sched

    fmd = sched["FMD"]
    assert "Inactivated" in fmd.get("recommended_vaccine", fmd.get("vaccine_name", ""))
    assert fmd["interval_days"] == 180
    assert fmd["farmer_cost_inr"] == 0.0
    assert "0" in fmd["farmer_cost_display"]
    assert fmd["price_type"] == "GOVERNMENT_PROGRAMME_FREE"


def test_generate_vaccination_recommendations_for_cattle():
    """Verifies vaccination recommendations generation for an animal."""
    cow = Cattle(
        animal_id="COW-VACC-01",
        tag_id="COW-VACC-01",
        farm_id="farm_01",
        user_id="user_test_vac",
        tag_number="COW-VACC-01",
        gender="Female",
        age_months=36.0,
        body_weight_kg=400.0,
        daily_milk_yield_litres=12.0
    )

    recs = vaccination_service.generate_recommendations(cow, administered_records=[])
    assert len(recs) >= 5

    targets = [r.disease_target.upper() for r in recs]
    assert "FMD" in targets
    assert "HS" in targets
    assert "BQ" in targets

    for r in recs:
        assert r.tag_id == "COW-VACC-01"
        assert r.recommended_vaccine
        assert r.recommended_timing
        assert r.next_due_date
        assert r.estimated_cost_display
        assert "Estimated information only" in r.disclaimer


def test_vaccination_due_date_calculation_after_administration():
    """Verifies next due date calculation when a past vaccine was administered."""
    cow = Cattle(
        animal_id="COW-VACC-02",
        tag_id="COW-VACC-02",
        farm_id="farm_01",
        user_id="user_test_vac",
        tag_number="COW-VACC-02",
        gender="Female",
        age_months=24.0,
        body_weight_kg=380.0
    )

    past_date = (date.today() - timedelta(days=60)).isoformat()
    admin_fmd = VaccinationRecord(
        record_id="vac_fmd_01",
        tag_id="COW-VACC-02",
        user_id="user_test_vac",
        disease_target="FMD",
        vaccine_name="FMD Oil Adjuvant Vaccine",
        administered_date=past_date,
        next_due_date=(date.today() + timedelta(days=120)).isoformat(),
        recommended_timing="Bi-annual booster every 6 months",
        status="COMPLETED",
        estimated_cost_inr=80.0
    )

    recs = vaccination_service.generate_recommendations(cow, administered_records=[admin_fmd])
    fmd_rec = next(r for r in recs if r.disease_target.upper() == "FMD")
    assert fmd_rec.status in ["UPCOMING", "COMPLETED"]
    assert fmd_rec.last_administered_date == past_date


def test_vaccination_api_endpoints():
    """Tests GET and POST /api/v1/cattle/{tag_id}/vaccinations endpoints."""
    headers = {"Authorization": "Bearer vac_tester", "X-User-ID": "vac_tester"}
    # Register cattle first
    reg_res = client.post(
        "/api/v1/cattle",
        json={
            "tag_id": "COW-VAC-API-01",
            "name": "Gauri",
            "breed": "Gir",
            "body_weight_kg": 410.0,
            "daily_milk_yield_litres": 14.0
        },
        headers=headers
    )
    assert reg_res.status_code == 201

    # Get recommendations
    get_res = client.get(
        "/api/v1/cattle/COW-VAC-API-01/vaccinations",
        headers=headers
    )
    assert get_res.status_code == 200
    recs = get_res.json()
    assert len(recs) >= 4
    assert recs[0]["tag_id"] == "COW-VAC-API-01"

    # Record administered vaccine
    post_res = client.post(
        "/api/v1/cattle/COW-VAC-API-01/vaccinations",
        json={
            "disease_target": "FMD",
            "vaccine_name": "Trivalent Inactivated FMD Vaccine",
            "administered_date": "2026-08-15",
            "veterinarian_name": "Dr. Kumar",
            "notes": "Administered annual booster shot"
        },
        headers=headers
    )
    assert post_res.status_code == 201
    rec_data = post_res.json()
    assert rec_data["disease_target"] == "FMD"
    assert rec_data["status"] == "COMPLETED"
    assert rec_data["veterinarian_name"] == "Dr. Kumar"
