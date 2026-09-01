"""
Unit & Integration Tests for Globally Unique Cattle Tag ID Enforcement
Verifies uniqueness across all users, case/whitespace normalization, 409 Conflict on duplicates,
and cross-tenant isolation.
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.app.schemas.user_farm_cattle import Cattle
from backend.app.db.farm_cattle_repository import get_farm_cattle_repository, DuplicateTagIdError

client = TestClient(app)


def test_tag_id_uniqueness_across_different_users():
    """Verifies that User B cannot register a Tag ID already registered by User A."""
    tag_id = "COW-UNIQUE-999"

    # User 1 registers COW-UNIQUE-999
    res1 = client.post(
        "/api/v1/cattle",
        json={"tag_id": tag_id, "name": "Cow A", "breed": "Gir"},
        headers={"Authorization": "Bearer user_alpha", "X-User-ID": "user_alpha"}
    )
    assert res1.status_code == 201

    # User 2 attempts to register the exact same Tag ID
    res2 = client.post(
        "/api/v1/cattle",
        json={"tag_id": tag_id, "name": "Cow B", "breed": "Murrah"},
        headers={"Authorization": "Bearer user_beta", "X-User-ID": "user_beta"}
    )
    assert res2.status_code == 409
    assert "Tag ID already exists. Please use a unique Tag ID." in res2.json()["detail"]


def test_tag_id_normalization_case_and_whitespace():
    """Verifies that variations in casing and whitespace are recognized as duplicate."""
    tag_id_base = "TAG-SPACES-100"

    # Register with base
    res1 = client.post(
        "/api/v1/cattle",
        json={"tag_id": "  tag-spaces-100  ", "name": "Ganga"},
        headers={"Authorization": "Bearer user_norm_01", "X-User-ID": "user_norm_01"}
    )
    assert res1.status_code == 201
    assert res1.json()["tag_id"] == "TAG-SPACES-100"

    # Duplicate with different casing/whitespace from another user
    res2 = client.post(
        "/api/v1/cattle",
        json={"tag_id": "TAG-SPACES-100", "name": "Yamuna"},
        headers={"Authorization": "Bearer user_norm_02", "X-User-ID": "user_norm_02"}
    )
    assert res2.status_code == 409


def test_cross_tenant_isolation_tag_id():
    """Verifies that User B cannot view User A's cattle even if they know the Tag ID."""
    tag_id = "COW-PRIVATE-777"

    # User Alpha creates cattle
    client.post(
        "/api/v1/cattle",
        json={"tag_id": tag_id, "name": "Secret Cow", "daily_milk_yield_litres": 20.0},
        headers={"Authorization": "Bearer user_alpha", "X-User-ID": "user_alpha"}
    )

    # User Beta tries to read User Alpha's cattle
    get_res = client.get(
        f"/api/v1/cattle/{tag_id}",
        headers={"Authorization": "Bearer user_beta", "X-User-ID": "user_beta"}
    )
    assert get_res.status_code == 404
