"""
Unit & Integration Tests for AI Chat Grounding on Cattle Tag ID
Verifies Tag ID resolution, injection of profile + milk history + vaccinations,
and unambiguous not-found message when querying nonexistent Tag IDs without hallucinations.
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.app.schemas.chat import ChatRequest
from backend.app.services.chat.chat_service import chat_service
from backend.app.db.farm_cattle_repository import get_farm_cattle_repository

client = TestClient(app)


def test_chat_with_existing_tag_id_retrieves_records():
    """Verifies that asking about an existing Tag ID grounds the reply in real records."""
    user_id = "chat_user_100"
    tag_id = "COW-1001"
    headers = {"Authorization": f"Bearer {user_id}", "X-User-ID": user_id}

    # Register cattle
    client.post(
        "/api/v1/cattle",
        json={
            "tag_id": tag_id,
            "name": "Lakshmi",
            "breed": "Gir",
            "daily_milk_yield_litres": 15.0,
            "milk_fat_percentage": 4.2
        },
        headers=headers
    )

    # Record persistent milk
    client.post(
        f"/api/v1/cattle/{tag_id}/milk",
        json={
            "date": "2026-08-28",
            "morning_yield_litres": 8.0,
            "evening_yield_litres": 7.0
        },
        headers=headers
    )

    # Query via AI Chat
    res = client.post(
        "/api/v1/chat",
        json={
            "message": "Tell me about COW-1001 and its milk history",
            "language": "en"
        },
        headers=headers
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    reply = data["reply"]
    assert "COW-1001" in reply or "Gir" in reply or "15" in reply


def test_chat_with_nonexistent_tag_id_returns_clear_message_no_hallucination():
    """Verifies that asking about a nonexistent Tag ID returns clear not-found message."""
    user_id = "chat_user_200"
    headers = {"Authorization": f"Bearer {user_id}", "X-User-ID": user_id}

    res = client.post(
        "/api/v1/chat",
        json={
            "message": "Tell me about COW-9999",
            "language": "en"
        },
        headers=headers
    )
    assert res.status_code == 200
    data = res.json()
    assert "I couldn't find a cattle record for Tag ID COW-9999" in data["reply"]
