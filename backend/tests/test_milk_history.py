"""
Unit & Integration Tests for Milk Production Recording & Automatic Milk History Sync
Verifies Tag ID linkage, Morning + Evening yield aggregation, automatic persistent history sync,
and retention without manual sync steps.
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.app.schemas.user_farm_cattle import MilkRecord, MilkRecordCreateRequest
from backend.app.db.farm_cattle_repository import get_farm_cattle_repository

client = TestClient(app)


def test_milk_recording_automatic_history_sync():
    """Verifies that recording milk automatically appears in persistent Milk History."""
    user_id = "milk_farmer_01"
    tag_id = "COW-MILK-101"
    headers = {"Authorization": f"Bearer {user_id}", "X-User-ID": user_id}

    # Register cattle
    reg_res = client.post(
        "/api/v1/cattle",
        json={
            "tag_id": tag_id,
            "name": "Nandini",
            "breed": "Sahiwal",
            "body_weight_kg": 430.0,
            "daily_milk_yield_litres": 10.0
        },
        headers=headers
    )
    assert reg_res.status_code == 201

    # Record Day 1 Milk
    rec1_res = client.post(
        f"/api/v1/cattle/{tag_id}/milk",
        json={
            "date": "2026-08-25",
            "morning_yield_litres": 6.5,
            "evening_yield_litres": 5.5,
            "fat_percentage": 4.2,
            "snf_percentage": 8.6,
            "notes": "Good morning intake"
        },
        headers=headers
    )
    assert rec1_res.status_code == 201
    rec1_data = rec1_res.json()
    assert rec1_data["total_yield_litres"] == 12.0
    assert rec1_data["tag_id"] == tag_id

    # Record Day 2 Milk
    rec2_res = client.post(
        f"/api/v1/cattle/{tag_id}/milk",
        json={
            "date": "2026-08-26",
            "morning_yield_litres": 7.0,
            "evening_yield_litres": 6.0,
            "fat_percentage": 4.3,
            "snf_percentage": 8.7
        },
        headers=headers
    )
    assert rec2_res.status_code == 201
    assert rec2_res.json()["total_yield_litres"] == 13.0

    # Fetch persistent Milk History - Both entries must be automatically present
    hist_res = client.get(
        f"/api/v1/cattle/{tag_id}/milk-history",
        headers=headers
    )
    assert hist_res.status_code == 200
    hist_data = hist_res.json()

    assert hist_data["tag_id"] == tag_id
    assert hist_data["total_records"] == 2
    assert hist_data["average_daily_yield_litres"] == 12.5  # (12.0 + 13.0) / 2
    assert len(hist_data["records"]) == 2
    assert hist_data["records"][0]["date"] == "2026-08-26"  # Most recent first
    assert hist_data["records"][1]["date"] == "2026-08-25"


def test_milk_recording_updates_cattle_daily_yield():
    """Verifies that recording milk updates the current cattle profile's daily yield."""
    user_id = "milk_farmer_02"
    tag_id = "COW-MILK-102"
    headers = {"Authorization": f"Bearer {user_id}", "X-User-ID": user_id}

    client.post(
        "/api/v1/cattle",
        json={"tag_id": tag_id, "daily_milk_yield_litres": 8.0},
        headers=headers
    )

    # Record new yield
    client.post(
        f"/api/v1/cattle/{tag_id}/milk",
        json={
            "date": "2026-08-28",
            "morning_yield_litres": 8.5,
            "evening_yield_litres": 7.5
        },
        headers=headers
    )

    # Check cattle profile
    cow_res = client.get(f"/api/v1/cattle/{tag_id}", headers=headers)
    assert cow_res.status_code == 200
    assert cow_res.json()["daily_milk_yield_litres"] == 16.0
