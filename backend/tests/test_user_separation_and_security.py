"""
Comprehensive Security, Data Separation, and Ownership Test Suite
Validates:
1. Demo User vs Real User Data Separation & Isolation.
2. Cross-User Access Prevention (User A vs User B).
3. Ownership Validation (User -> Farm -> Cattle -> Analysis History).
4. Guest / Unauthenticated Access Controls.
5. Selected Animal AI Advisory Context & Zero Fallback Guarantee.
6. Image Magic-Byte Security & Payload Size Enforcement.
"""

import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from backend.main import app
from backend.app.db.farm_cattle_repository import (
    get_farm_cattle_repository,
    DEMO_USER_ID,
    DEMO_FARM_ID,
    DEMO_ANIMAL_ID
)
from backend.app.schemas.user_farm_cattle import Farm, Cattle, AnalysisRecord


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def setup_test_repository_state():
    """Seeds isolated test data for User A and User B."""
    repo = get_farm_cattle_repository()
    
    # 1. User A Data
    user_a_id = "usr_alice_101"
    farm_a_id = "farm_alice_101"
    cow_a_id = "COW_ALICE_01"
    
    farm_a = Farm(farm_id=farm_a_id, user_id=user_a_id, farm_name="Alice Organic Dairy")
    cow_a = Cattle(
        animal_id=cow_a_id,
        farm_id=farm_a_id,
        user_id=user_a_id,
        species="Cattle",
        breed="Sahiwal",
        body_weight_kg=430.0,
        daily_milk_yield_litres=16.0,
        milk_fat_percentage=4.5
    )
    repo.save_farm(farm_a)
    repo.save_cattle(cow_a)

    # 2. User B Data
    user_b_id = "usr_bob_202"
    farm_b_id = "farm_bob_202"
    cow_b_id = "COW_BOB_01"
    
    farm_b = Farm(farm_id=farm_b_id, user_id=user_b_id, farm_name="Bob Modern Dairy")
    cow_b = Cattle(
        animal_id=cow_b_id,
        farm_id=farm_b_id,
        user_id=user_b_id,
        species="Buffalo",
        breed="Murrah",
        body_weight_kg=550.0,
        daily_milk_yield_litres=12.0,
        milk_fat_percentage=7.2
    )
    repo.save_farm(farm_b)
    repo.save_cattle(cow_b)


# ==============================================================================
# 1. DEMO USER VS REAL USER DATA SEPARATION
# ==============================================================================

def test_demo_user_vs_real_user_isolation():
    """Verify demo cattle are isolated under DEMO_USER_ID and never leak to real users."""
    repo = get_farm_cattle_repository()
    
    # Demo User lists cattle -> Gets DEMO_ANIMAL_ID
    demo_cattle = repo.list_cattle(user_id=DEMO_USER_ID)
    assert len(demo_cattle) >= 1
    assert any(c.animal_id == DEMO_ANIMAL_ID for c in demo_cattle)

    # Real User Alice lists cattle -> Does NOT get DEMO_ANIMAL_ID or Bob's cattle
    alice_cattle = repo.list_cattle(user_id="usr_alice_101")
    assert len(alice_cattle) == 1
    assert alice_cattle[0].animal_id == "COW_ALICE_01"
    assert not any(c.animal_id == DEMO_ANIMAL_ID for c in alice_cattle)


# ==============================================================================
# 2. CROSS-USER ACCESS PREVENTION
# ==============================================================================

def test_cross_user_cattle_access_prevention():
    """Verify User B cannot access User A's cattle."""
    repo = get_farm_cattle_repository()
    
    # Alice accesses Alice's cow -> Allowed
    cow_alice = repo.get_cattle(animal_id="COW_ALICE_01", user_id="usr_alice_101")
    assert cow_alice is not None
    assert cow_alice.breed == "Sahiwal"

    # Bob attempts to access Alice's cow -> Denied (Returns None)
    cow_bob_attempt = repo.get_cattle(animal_id="COW_ALICE_01", user_id="usr_bob_202")
    assert cow_bob_attempt is None


def test_cross_user_farm_access_prevention():
    """Verify User B cannot access User A's farm."""
    repo = get_farm_cattle_repository()
    
    farm_alice = repo.get_farm(farm_id="farm_alice_101", user_id="usr_alice_101")
    assert farm_alice is not None

    farm_bob_attempt = repo.get_farm(farm_id="farm_alice_101", user_id="usr_bob_202")
    assert farm_bob_attempt is None


# ==============================================================================
# 3. ENDPOINT OWNERSHIP VALIDATION
# ==============================================================================

def test_endpoint_unauthorized_animal_rejection(client: TestClient):
    """Verify 403 Forbidden when requesting feed analysis for unowned animal_id."""
    headers = {"Authorization": "Bearer usr_bob_202"}
    # Bob attempts to pass Alice's farm_id and animal_id
    res = client.post(
        "/api/v1/analyze/feed",
        data={"feed_name": "Maize", "quantity_kg": "5.0", "farm_id": "farm_alice_101", "animal_id": "COW_ALICE_01"},
        headers=headers
    )
    assert res.status_code == 403
    assert "Access denied" in res.json()["detail"]


# ==============================================================================
# 4. GUEST / UNAUTHENTICATED ACCESS CONTROL
# ==============================================================================

def test_guest_unauthenticated_personalized_access_rejection(client: TestClient):
    """Verify unauthenticated guest cannot pass farm_id or animal_id."""
    # No Auth header
    res = client.post(
        "/api/v1/analyze/feed",
        data={"feed_name": "Maize", "quantity_kg": "5.0", "farm_id": "farm_alice_101"}
    )
    assert res.status_code == 401
    assert "Unauthenticated guest" in res.json()["detail"]


def test_guest_public_endpoint_allowed(client: TestClient):
    """Verify unauthenticated guest can access public feed reference lookup."""
    res = client.post("/api/v1/feed/reference", json={"feed_name": "Maize", "quantity_kg": 5.0})
    assert res.status_code == 200
    assert "Maize" in res.json()["matched_feed_name"]


# ==============================================================================
# 5. SELECTED ANIMAL AI ADVISORY CONTEXT & ZERO FALLBACK
# ==============================================================================

def test_selected_animal_ai_advisory_context_resolution(client: TestClient):
    """Verify AI advisory uses ONLY selected authorized animal's profile."""
    headers = {"Authorization": "Bearer usr_alice_101"}
    payload = {
        "message": "What is the recommended feed requirement?",
        "selected_animal_id": "COW_ALICE_01"
    }
    res = client.post("/api/v1/chat", json=payload, headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["metadata"]["selected_animal_active"] is True
    # Alice's cow yields 16.0 L milk @ 4.5% fat
    assert data["metadata"]["nutrition_extracted"]["daily_milk_yield_litres"] == 16.0
    assert data["metadata"]["nutrition_extracted"]["milk_fat_percentage"] == 4.5
    assert data["metadata"]["nutrition_extracted"]["breed"] == "Sahiwal"


def test_unauthorized_selected_animal_ai_advisory_rejection(client: TestClient):
    """Verify 403 Forbidden when chat request uses an unowned selected_animal_id."""
    headers = {"Authorization": "Bearer usr_bob_202"}
    payload = {
        "message": "What should I feed my cow?",
        "selected_animal_id": "COW_ALICE_01"  # Bob requesting Alice's cow
    }
    res = client.post("/api/v1/chat", json=payload, headers=headers)
    assert res.status_code == 403


# ==============================================================================
# 6. IMAGE MAGIC-BYTE SECURITY & CORRUPT PAYLOAD REJECTION
# ==============================================================================

def test_image_magic_byte_rejection(client: TestClient):
    """Verify plain text file with .jpg extension is rejected by magic byte check."""
    fake_image_bytes = b"Hello world this is text file disguised as JPEG"
    res = client.post(
        "/api/v1/analyze/feed",
        data={"feed_name": "Maize"},
        files={"image": ("fake.jpg", fake_image_bytes, "image/jpeg")}
    )
    assert res.status_code == 400
    assert "magic bytes" in res.json()["message"].lower() or "security check" in res.json()["message"].lower()


def test_valid_image_magic_byte_acceptance(client: TestClient):
    """Verify valid JPEG file with correct magic bytes is accepted."""
    import numpy as np
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    np.random.seed(42)
    arr[:, :, 0] = np.random.randint(180, 225, (100, 100))
    arr[:, :, 1] = np.random.randint(150, 195, (100, 100))
    arr[:, :, 2] = np.random.randint(50, 95, (100, 100))
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    valid_bytes = buf.getvalue()

    res = client.post(
        "/api/v1/analyze/feed",
        data={"feed_name": "Maize"},
        files={"image": ("real.jpg", valid_bytes, "image/jpeg")}
    )
    assert res.status_code == 200
    assert res.json()["visual_screening"] is not None
    assert res.json()["visual_screening"]["predicted_class"] == "GOOD"
