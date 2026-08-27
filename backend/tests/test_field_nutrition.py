"""
Unit & Integration Test Suite for Field Nutrition & Least-Cost Ration Optimization
Tests ICAR Mathematical Partitioning, Linear Programming Optimizer, API Endpoints, and Multilingual Chat Integration
"""

import uuid
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.app.schemas.nutrition import NutritionRecommendationRequest
from backend.app.schemas.chat import ChatRequest
from backend.app.services.nutrition_engine import nutrition_engine
from backend.app.services.chat.nutrition_service import nutrition_service
from backend.app.services.chat.chat_service import chat_service

client = TestClient(app)


# ==============================================================================
# 1. ICAR MATHEMATICAL PARTITIONING TESTS
# ==============================================================================

def test_icar_zebu_cow_requirements():
    """Validates ICAR formulas for Indigenous Zebu Cattle."""
    req = nutrition_engine.calculate_icar_requirements(
        species="Cattle",
        breed_type="Indigenous_Zebu",
        body_weight_kg=400.0,
        daily_milk_yield_kg=10.0,
        milk_fat_percent=4.0
    )
    # MBW = 400^0.75 = 89.44 kg
    assert pytest.approx(req.metabolic_body_weight_kg, rel=1e-2) == 89.44
    # 4% FCM = (0.4 + 0.15*4.0)*10 = 10.0 kg
    assert req.fat_corrected_milk_4pct_kg == 10.0
    # Maint DMI = 0.022*400 = 8.8 kg, Lact DMI = 0.33*10 = 3.3 kg -> Total = 12.1 kg
    assert req.req_dmi_kg_per_day == 12.1
    # TDN = 0.034*89.44 + 0.320*10 = 3.04 + 3.20 = 6.24 kg
    assert req.req_tdn_kg_per_day >= 6.0 and req.req_tdn_kg_per_day <= 6.5
    # CP = 4.2*89.44 + 85*10 = 375.6 + 850 = 1225.6 g
    assert req.req_cp_g_per_day >= 1200 and req.req_cp_g_per_day <= 1250


def test_icar_buffalo_requirements():
    """Validates ICAR formulas for Water Buffaloes."""
    req = nutrition_engine.calculate_icar_requirements(
        species="Buffalo",
        breed_type="Murrah",
        body_weight_kg=550.0,
        daily_milk_yield_kg=12.0,
        milk_fat_percent=7.0
    )
    # MBW = 550^0.75 = 113.48 kg
    assert pytest.approx(req.metabolic_body_weight_kg, rel=1e-2) == 113.48
    # 4% FCM = (0.4 + 0.15*7.0)*12 = 1.45*12 = 17.4 kg
    assert pytest.approx(req.fat_corrected_milk_4pct_kg, rel=1e-2) == 17.4
    # Buffalo TDN coefficient = 0.340 per kg FCM
    assert req.req_tdn_kg_per_day >= 9.0


def test_icar_pregnancy_allowance():
    """Validates pregnancy nutrient bumps for advanced pregnancy."""
    non_preg = nutrition_engine.calculate_icar_requirements(
        species="Cattle",
        breed_type="Indigenous_Zebu",
        body_weight_kg=450.0,
        daily_milk_yield_kg=0.0,
        milk_fat_percent=4.0,
        pregnancy_month=0
    )
    preg = nutrition_engine.calculate_icar_requirements(
        species="Cattle",
        breed_type="Indigenous_Zebu",
        body_weight_kg=450.0,
        daily_milk_yield_kg=0.0,
        milk_fat_percent=4.0,
        pregnancy_month=8
    )
    assert preg.req_tdn_kg_per_day > non_preg.req_tdn_kg_per_day
    assert preg.req_cp_g_per_day > non_preg.req_cp_g_per_day


# ==============================================================================
# 2. LINEAR PROGRAMMING OPTIMIZER TESTS
# ==============================================================================

def test_optimizer_normal_cow():
    """Tests optimization for 400kg cow yielding 10kg milk @ 4% fat."""
    req = NutritionRecommendationRequest(
        species="cow",
        breed="gir",
        body_weight_kg=400.0,
        daily_milk_yield_kg=10.0,
        milk_fat_percent=4.0
    )
    res = nutrition_engine.optimize_ration(req)
    assert res.success is True
    assert res.status == "optimized"
    assert res.total_daily_cost_inr > 0
    assert len(res.recommended_ration) >= 2


def test_optimizer_high_yielding_crossbred():
    """Tests optimization for 500kg crossbred yielding 25kg milk @ 3.8% fat."""
    req = NutritionRecommendationRequest(
        species="cow",
        breed="crossbred_hf",
        body_weight_kg=500.0,
        daily_milk_yield_kg=25.0,
        milk_fat_percent=3.8
    )
    res = nutrition_engine.optimize_ration(req)
    assert res.success is True
    assert res.status == "optimized"
    assert res.nutrient_balance["crude_protein"].percentage_fulfilled >= 99.0
    assert res.nutrient_balance["tdn"].percentage_fulfilled >= 99.0


def test_optimizer_custom_feed_selection():
    """Tests optimization restricted to a subset of feeds with custom prices."""
    req = NutritionRecommendationRequest(
        species="cow",
        body_weight_kg=450.0,
        daily_milk_yield_kg=12.0,
        milk_fat_percent=4.2,
        available_feeds=["IN_GF_004", "IN_DR_013", "IN_PC_023", "IN_CF_035", "IN_CF_037"],
        feed_prices={"IN_GF_004": 2.0, "IN_DR_013": 5.0}
    )
    res = nutrition_engine.optimize_ration(req)
    assert res.success is True
    for item in res.recommended_ration:
        assert item.feed_id in req.available_feeds


def test_optimizer_missing_parameters():
    """Tests that missing critical parameters are gracefully reported without crashing."""
    req = NutritionRecommendationRequest(
        species="cow",
        body_weight_kg=None,
        daily_milk_yield_kg=10.0,
        milk_fat_percent=4.0
    )
    res = nutrition_engine.optimize_ration(req)
    assert res.success is False
    assert res.status == "missing_parameters"
    assert "body_weight_kg" in res.missing_critical_parameters


def test_optimizer_invalid_parameters():
    """Tests parameter boundary validation."""
    req = NutritionRecommendationRequest(
        species="cow",
        body_weight_kg=20.0,
        daily_milk_yield_kg=10.0,
        milk_fat_percent=4.0
    )
    res = nutrition_engine.optimize_ration(req)
    assert res.success is False
    assert res.status == "invalid_parameters"


# ==============================================================================
# 3. FASTAPI ENDPOINT TESTS
# ==============================================================================

def test_api_nutrition_recommend():
    """Tests POST /api/v1/nutrition/recommend."""
    payload = {
        "species": "cow",
        "breed": "sahiwal",
        "body_weight_kg": 420.0,
        "daily_milk_yield_kg": 12.0,
        "milk_fat_percent": 4.5
    }
    response = client.post("/api/v1/nutrition/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["total_daily_cost_inr"] > 0
    assert len(data["recommended_ration"]) > 0


def test_api_nutrition_feeds():
    """Tests GET /api/v1/nutrition/feeds."""
    response = client.get("/api/v1/nutrition/feeds")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 20


def test_api_nutrition_standards():
    """Tests GET /api/v1/nutrition/standards."""
    response = client.get("/api/v1/nutrition/standards")
    assert response.status_code == 200
    data = response.json()
    assert "standard" in data
    assert "ICAR" in data["standard"]


# ==============================================================================
# 4. CHAT INTEGRATION TESTS
# ==============================================================================

def test_chat_nutrition_english_full_query():
    """Tests chat handling of complete English nutrition request."""
    sess_id = f"sess_test_en_full_{uuid.uuid4().hex[:8]}"
    req = ChatRequest(
        message="I have a 450 kg Gir cow producing 14 liters of milk per day with 4.5% fat. What is the recommended least-cost feed ration?",
        session_id=sess_id,
        language="en"
    )
    res = chat_service.process_message(req)
    assert res.success is True
    assert res.intent == "nutrition"
    assert res.metadata["nutrition_model_active"] is True
    assert len(res.metadata["nutrition_missing"]) == 0
    assert "Rs." in res.reply or "₹" in res.reply


def test_chat_nutrition_tamil_full_query():
    """Tests chat handling of complete Tamil nutrition request with explicit fat %."""
    sess_id = f"sess_test_ta_full_{uuid.uuid4().hex[:8]}"
    req = ChatRequest(
        message="என் 400 கிலோ பசு மாடு தினமும் 10 லிட்டர் 4% கொழுப்பு பால் தருகிறது, சமச்சீர் தீவன ரேஷன் என்ன?",
        session_id=sess_id,
        language="ta"
    )
    res = chat_service.process_message(req)
    assert res.success is True
    assert res.intent == "nutrition"
    assert res.language == "ta"
    assert "தீவன" in res.reply or "கிலோ" in res.reply or "ரேஷன்" in res.reply
    assert "ரூ." in res.reply or "Rs." in res.reply


def test_chat_nutrition_hindi_full_query():
    """Tests chat handling of complete Hindi nutrition request with explicit fat."""
    sess_id = f"sess_test_hi_full_{uuid.uuid4().hex[:8]}"
    req = ChatRequest(
        message="मेरी गाय का वजन 400 किलो है और वह 12 लीटर 4% फैट दूध देती है, संतुलित आहार बताएं",
        session_id=sess_id,
        language="hi"
    )
    res = chat_service.process_message(req)
    assert res.success is True
    assert res.intent == "nutrition"
    assert res.language == "hi"
    assert "आहार" in res.reply or "किलो" in res.reply or "दूध" in res.reply


def test_chat_nutrition_missing_weight_prompts_politely():
    """Tests that missing weight in chat asks the farmer for missing parameters."""
    sess_id = f"sess_test_missing_wt_{uuid.uuid4().hex[:8]}"
    req = ChatRequest(
        message="What balanced ration and nutrition requirement should I give for my milking cow?",
        session_id=sess_id,
        language="en"
    )
    res = chat_service.process_message(req)
    assert res.success is True
    assert res.intent == "nutrition"
    assert "body weight" in res.reply.lower() or "milk yield" in res.reply.lower()


def test_chat_nutrition_tamil_missing_parameters():
    """Tests that missing parameters in Tamil asks the farmer in Tamil."""
    sess_id = f"sess_test_ta_missing_{uuid.uuid4().hex[:8]}"
    req = ChatRequest(
        message="என் மாட்டின் தீவன தேவை மற்றும் ஊட்டச்சத்து ரேஷன் அளவு என்ன?",
        session_id=sess_id,
        language="ta"
    )
    res = chat_service.process_message(req)
    assert res.success is True
    assert res.intent == "nutrition"
    assert "எடை" in res.reply or "பால்" in res.reply


def test_chat_nutrition_missing_fat_asks_farmer_for_fat():
    """Tests that omitted fat % does NOT invent 4% and prompts farmer for fat %."""
    sess_id = f"sess_test_missing_fat_{uuid.uuid4().hex[:8]}"
    req = ChatRequest(
        message="En maadu 420 kg irukku, daily 15 litre paal kudukuthu. Enna feed kudukanum?",
        session_id=sess_id,
        language="ta"
    )
    res = chat_service.process_message(req)
    assert res.success is True
    assert res.intent == "nutrition"
    assert res.metadata["nutrition_missing"] == ["milk_fat_percentage"]
    assert res.metadata["nutrition_model_active"] is False
    assert "கொழுப்பு" in res.reply or "fat" in res.reply.lower()


def test_chat_nutrition_multiturn_fat_clarification():
    """Tests multi-turn conversation where farmer provides missing fat % in next turn."""
    sess_id = f"sess_test_fat_clarification_{uuid.uuid4().hex[:8]}"

    # Turn 1: Farmer gives weight and milk yield, but forgets fat %
    req1 = ChatRequest(
        message="En maadu 420 kg irukku, daily 15 litre paal kudukuthu. Enna feed kudukanum?",
        session_id=sess_id,
        language="ta"
    )
    res1 = chat_service.process_message(req1)
    assert res1.success is True
    assert res1.metadata["nutrition_missing"] == ["milk_fat_percentage"]
    assert "கொழுப்பு" in res1.reply or "fat" in res1.reply.lower()

    # Turn 2: Farmer supplies 4% fat
    req2 = ChatRequest(
        message="4%",
        session_id=sess_id,
        language="ta"
    )
    res2 = chat_service.process_message(req2)
    assert res2.success is True
    assert res2.metadata["nutrition_missing"] == []
    assert res2.metadata["nutrition_model_active"] is True
    assert "ரூ." in res2.reply or "Rs." in res2.reply


def test_chat_nutrition_no_hidden_default_fat():
    """Directly verifies that entity parser never invents or defaults 4% fat."""
    parsed = nutrition_service.parse_nutrition_entities("En maadu 420 kg irukku, daily 15 litre paal kudukuthu.")
    assert parsed.body_weight_kg == 420.0
    assert parsed.daily_milk_yield_litres == 15.0
    assert parsed.milk_fat_percentage is None
    assert parsed.pregnancy_status is None
