"""
Unit & Integration Tests for Lactation Tracking & Days in Milk (DIM)
Verifies formula DIM = Current Date - Calving Date, stages (Early, Mid, Late, Dry),
and calving event recalculations.
"""

from datetime import date, timedelta
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.app.services.lactation_service import lactation_service
from backend.app.schemas.user_farm_cattle import Cattle

client = TestClient(app)


def test_lactation_service_stage_determinations():
    """Verifies that lactation stages map correctly to DIM ranges."""
    today = date.today()

    # Early: 1-100 days
    calving_early = (today - timedelta(days=45)).isoformat()
    cow_early = Cattle(
        animal_id="COW-LAC-01", tag_id="COW-LAC-01", farm_id="f1", user_id="u1",
        calving_date=calving_early, current_lactation_status="Lactating"
    )
    res_early = lactation_service.calculate_lactation_status(cow_early)
    assert res_early.days_in_milk == 45.0
    assert res_early.lactation_stage == "Early"

    # Mid: 101-200 days
    calving_mid = (today - timedelta(days=150)).isoformat()
    cow_mid = Cattle(
        animal_id="COW-LAC-02", tag_id="COW-LAC-02", farm_id="f1", user_id="u1",
        calving_date=calving_mid, current_lactation_status="Lactating"
    )
    res_mid = lactation_service.calculate_lactation_status(cow_mid)
    assert res_mid.days_in_milk == 150.0
    assert res_mid.lactation_stage == "Mid"

    # Late: 201-305 days
    calving_late = (today - timedelta(days=250)).isoformat()
    cow_late = Cattle(
        animal_id="COW-LAC-03", tag_id="COW-LAC-03", farm_id="f1", user_id="u1",
        calving_date=calving_late, current_lactation_status="Lactating"
    )
    res_late = lactation_service.calculate_lactation_status(cow_late)
    assert res_late.days_in_milk == 250.0
    assert res_late.lactation_stage == "Late"

    # Dry: >305 days or status Dry
    calving_dry = (today - timedelta(days=320)).isoformat()
    cow_dry = Cattle(
        animal_id="COW-LAC-04", tag_id="COW-LAC-04", farm_id="f1", user_id="u1",
        calving_date=calving_dry, current_lactation_status="Dry"
    )
    res_dry = lactation_service.calculate_lactation_status(cow_dry)
    assert res_dry.days_in_milk == 320.0
    assert res_dry.lactation_stage == "Dry"


def test_calving_event_resets_lactation_cycle():
    """Verifies that recording a new calving event resets DIM and restarts Early Lactation."""
    user_id = "calving_tester"
    tag_id = "COW-CALF-01"

    # Register cow with old calving date
    old_calving = (date.today() - timedelta(days=340)).isoformat()
    headers = {"Authorization": f"Bearer {user_id}", "X-User-ID": user_id}
    client.post(
        "/api/v1/cattle",
        json={
            "tag_id": tag_id,
            "calving_date": old_calving,
            "parity": 1,
            "current_lactation_status": "Dry"
        },
        headers=headers
    )

    # Verify initial status is Dry
    init_res = client.get(f"/api/v1/cattle/{tag_id}/lactation", headers=headers)
    assert init_res.status_code == 200
    assert init_res.json()["lactation_stage"] == "Dry"

    # Record new calving event (e.g. 5 days ago)
    new_calving = (date.today() - timedelta(days=5)).isoformat()
    calv_res = client.post(
        f"/api/v1/cattle/{tag_id}/calving",
        json={
            "calving_date": new_calving,
            "parity": 2,
            "calf_gender": "Female",
            "notes": "Healthy female calf born"
        },
        headers=headers
    )
    assert calv_res.status_code == 200
    status_data = calv_res.json()
    assert status_data["days_in_milk"] == 5.0
    assert status_data["lactation_stage"] == "Early"
    assert status_data["parity"] == 2
    assert status_data["current_status"] == "Lactating"
