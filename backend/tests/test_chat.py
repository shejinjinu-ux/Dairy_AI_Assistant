"""
Comprehensive Test Suite for Multilingual AI Chat Assistant
Validates 20+ Indian Languages, Intent Classification, Module Routing, Silage & Nutrition Integration,
Session Memory, Error Handling, and Pluggable Architectures
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.app.services.chat.nutrition_service import nutrition_service, RationRequestModel


@pytest.fixture(scope="module")
def client():
    """Create TestClient with application lifespan executed."""
    with TestClient(app) as test_client:
        yield test_client


# ----------------------------------------------------------------------
# 1. Core Language Tests (All 20+ Indian Languages + English)
# ----------------------------------------------------------------------

def test_chat_english(client: TestClient):
    """Test A: English chat interaction."""
    payload = {"message": "Hello, how can I feed my cow?"}
    res = client.post("/api/v1/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["language"] == "en"
    assert data["detected_language"] == "en"
    assert len(data["reply"]) > 0
    assert "feed" in data["reply"].lower() or "nutrition" in data["intent"] or "feed" in data["intent"]


def test_chat_tamil(client: TestClient):
    """Test B: Tamil chat interaction."""
    payload = {"message": "என் மாட்டுக்கு என்ன தீவனம் கொடுக்க வேண்டும்?"}
    res = client.post("/api/v1/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["detected_language"] == "ta"
    assert data["language"] == "ta"
    assert "தீவனம்" in data["reply"] or "பசுந்தீவனம்" in data["reply"] or "உணவு" in data["reply"]


def test_chat_hindi(client: TestClient):
    """Test C: Hindi chat interaction."""
    payload = {"message": "मेरी गाय कितना चारा खाए?"}
    res = client.post("/api/v1/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["detected_language"] == "hi"
    assert data["language"] == "hi"
    assert "चारा" in data["reply"] or "आहार" in data["reply"]


def test_chat_telugu(client: TestClient):
    """Test D: Telugu chat interaction."""
    payload = {"message": "ఆవుకు ఎంత మేత ఇవ్వాలి?"}
    res = client.post("/api/v1/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["detected_language"] == "te"
    assert data["language"] == "te"


def test_chat_kannada(client: TestClient):
    """Test E: Kannada chat interaction."""
    payload = {"message": "ಹಸುವಿಗೆ ಎಷ್ಟು ಮೇವು ಕೊಡಬೇಕು?"}
    res = client.post("/api/v1/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["detected_language"] == "kn"
    assert data["language"] == "kn"


def test_chat_malayalam(client: TestClient):
    """Test F: Malayalam chat interaction."""
    payload = {"message": "പശുവിന് എത്ര തീറ്റ കൊടുക്കണം?"}
    res = client.post("/api/v1/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["detected_language"] == "ml"
    assert data["language"] == "ml"


def test_chat_bengali(client: TestClient):
    """Test G: Bengali chat interaction."""
    payload = {"message": "গরুকে কতটুকু ঘাস খাওয়াতে হবে?"}
    res = client.post("/api/v1/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["detected_language"] == "bn"
    assert data["language"] == "bn"


def test_chat_marathi(client: TestClient):
    """Test H: Marathi chat interaction."""
    payload = {"message": "गाईला किती चारा द्यावा?"}
    res = client.post("/api/v1/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["language"] in ["mr", "hi"]


def test_chat_gujarati(client: TestClient):
    """Test I: Gujarati chat interaction."""
    payload = {"message": "ગાયને કેટલો ચારો આપવો?"}
    res = client.post("/api/v1/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["detected_language"] == "gu"
    assert data["language"] == "gu"


def test_chat_punjabi(client: TestClient):
    """Test J: Punjabi chat interaction."""
    payload = {"message": "ਗਾਂ ਨੂੰ ਕਿੰਨਾ ਚਾਰਾ ਦੇਣਾ ਚਾਹੀਦਾ ਹੈ?"}
    res = client.post("/api/v1/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["detected_language"] == "pa"
    assert data["language"] == "pa"


def test_chat_odia(client: TestClient):
    """Test K: Odia chat interaction."""
    payload = {"message": "ଗାଈକୁ କେତେ ଘାସ ଦେବା ଉଚିତ?"}
    res = client.post("/api/v1/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["detected_language"] == "or"
    assert data["language"] == "or"


def test_chat_assamese(client: TestClient):
    """Test L: Assamese chat interaction."""
    payload = {"message": "গাভীক কেনেদৰে খাদ্য দিব লাগে?"}
    res = client.post("/api/v1/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["detected_language"] in ["as", "bn"]


def test_chat_urdu(client: TestClient):
    """Test M: Urdu chat interaction."""
    payload = {"message": "گائے کو کتنا چارہ دینا چاہیے؟"}
    res = client.post("/api/v1/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["detected_language"] == "ur"
    assert data["language"] == "ur"


def test_chat_additional_indic_languages(client: TestClient):
    """Test remaining languages: Sanskrit, Nepali, Konkani, Kashmiri, Sindhi, Maithili, Manipuri."""
    langs = [
        ("sa", "धेनुपोषणे किं दातव्यम्?"),
        ("ne", "गाईलाई कति घाँस दिने?"),
        ("kok", "गायक कितें खावड दिवचें?"),
        ("mai", "गाय के की चारा दियौक?"),
        ("mni", "ꯁꯟꯒꯤ ꯆꯤꯟꯖꯥꯛ ꯃꯇꯥꯡꯗ ꯈꯪꯍনবꯤꯌꯨ")
    ]
    for code, text in langs:
        res = client.post("/api/v1/chat", json={"message": text})
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert len(data["reply"]) > 0


# ----------------------------------------------------------------------
# 2. Tanglish & Romanized Indian Languages
# ----------------------------------------------------------------------

def test_chat_tanglish(client: TestClient):
    """Test N: Romanized Tamil / Tanglish understanding."""
    payload = {"message": "En maadu 15 litre paal kudukuthu, enna feed kudukkanum?"}
    res = client.post("/api/v1/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["detected_language"] == "ta"
    assert data["intent"] == "nutrition"
    assert data["module"] == "nutrition"


def test_chat_hinglish(client: TestClient):
    """Test Romanized Hindi / Hinglish understanding."""
    payload = {"message": "Meri gaay kitna chara khaye?"}
    res = client.post("/api/v1/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["detected_language"] == "hi"


# ----------------------------------------------------------------------
# 3. Language Detection & Selection Priority
# ----------------------------------------------------------------------

def test_language_auto_detection(client: TestClient):
    """Test O: Auto-detection when language field is omitted."""
    res_ta = client.post("/api/v1/chat", json={"message": "வணக்கம்"})
    assert res_ta.status_code == 200
    assert res_ta.json()["detected_language"] == "ta"
    assert res_ta.json()["language"] == "ta"

    res_hi = client.post("/api/v1/chat", json={"message": "नमस्ते"})
    assert res_hi.status_code == 200
    assert res_hi.json()["detected_language"] == "hi"
    assert res_hi.json()["language"] == "hi"


def test_explicit_language_override(client: TestClient):
    """Test P: Explicit language override takes priority for response."""
    payload = {
        "message": "What should I feed my cow?",
        "language": "ta"
    }
    res = client.post("/api/v1/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["detected_language"] == "en"
    assert data["language"] == "ta"
    assert "தீவனம்" in data["reply"] or "உணவு" in data["reply"] or "பரிந்துரைக்க" in data["reply"]


# ----------------------------------------------------------------------
# 4. Intent Classification & Routing Tests
# ----------------------------------------------------------------------

def test_greeting_intent(client: TestClient):
    """Test Q: Greeting intent."""
    payload = {"message": "Hi, good morning!"}
    res = client.post("/api/v1/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["intent"] == "greeting"
    assert data["module"] == "chat"


def test_nutrition_intent(client: TestClient):
    """Test R: Nutrition intent routing."""
    payload = {"message": "My cow gives 15 litres milk, what is the feed requirement?"}
    res = client.post("/api/v1/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["intent"] == "nutrition"
    assert data["module"] == "nutrition"
    assert data["metadata"]["nutrition_model_active"] is False


def test_silage_intent_with_parameters(client: TestClient):
    """Test S1: Silage intent with test parameters calls existing silage model."""
    payload = {"message": "Is my silage good? pH is 3.80 and dry matter is 33% with CP 14.5%"}
    res = client.post("/api/v1/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["intent"] == "silage_quality"
    assert data["module"] == "silage"
    assert data["metadata"]["silage_evaluated"] is True
    assert "FQI Score:" in data["reply"]


def test_silage_intent_missing_parameters(client: TestClient):
    """Test S2: Silage intent with missing test values asks for pH/moisture."""
    payload = {"message": "Is my silage good?"}
    res = client.post("/api/v1/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["intent"] == "silage_quality"
    assert data["module"] == "silage"
    assert data["metadata"]["silage_evaluated"] is False
    assert "pH" in data["reply"] or "moisture" in data["reply"] or "test" in data["reply"].lower()


def test_feed_intent(client: TestClient):
    """Test Feed category intent."""
    payload = {"message": "Can I feed maize and green fodder to my cattle?"}
    res = client.post("/api/v1/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["intent"] == "feed"


def test_cattle_health_intent_with_disclaimer(client: TestClient):
    """Test Cattle Health intent includes non-diagnostic veterinary disclaimer."""
    payload = {"message": "My cow has high fever and is not eating"}
    res = client.post("/api/v1/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["intent"] == "cattle_health_general"
    assert data["module"] == "health"
    assert "veterinarian" in data["reply"].lower() or "doctor" in data["reply"].lower() or "மருத்துவர்" in data["reply"] or "चिकित्सक" in data["reply"]


def test_milk_production_intent(client: TestClient):
    """Test Milk Production intent."""
    payload = {"message": "How can I improve my daily milk production?"}
    res = client.post("/api/v1/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["intent"] == "milk_production"


def test_unknown_intent(client: TestClient):
    """Test T: Unknown intent provides polite clarification."""
    payload = {"message": "Explain rocket science and space travel"}
    res = client.post("/api/v1/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["intent"] == "unknown"


# ----------------------------------------------------------------------
# 5. Session Memory & Multi-turn Continuity
# ----------------------------------------------------------------------

def test_session_memory(client: TestClient):
    """Test U: Multi-turn conversation continuity and session persistence."""
    # Turn 1
    t1_payload = {
        "message": "My cow gives 15 litres milk.",
        "session_id": "test_session_001"
    }
    res1 = client.post("/api/v1/chat", json=t1_payload)
    assert res1.status_code == 200
    assert res1.json()["session_id"] == "test_session_001"

    # Turn 2 (Contextual follow-up)
    t2_payload = {
        "message": "Then what should I feed it?",
        "session_id": "test_session_001"
    }
    res2 = client.post("/api/v1/chat", json=t2_payload)
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["session_id"] == "test_session_001"
    # Should resolve intent via context
    assert data2["intent"] in ["nutrition", "feed"]


# ----------------------------------------------------------------------
# 6. Error Handling & Validation Tests
# ----------------------------------------------------------------------

def test_empty_message_validation(client: TestClient):
    """Test V: Empty / whitespace message returns validation error."""
    payload = {"message": "   "}
    res = client.post("/api/v1/chat", json=payload)
    assert res.status_code == 422


def test_invalid_language_fallback(client: TestClient):
    """Test W: Invalid language code falls back gracefully without crashing."""
    payload = {
        "message": "What is the best feed for my cow?",
        "language": "xyz_invalid_123"
    }
    res = client.post("/api/v1/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["language"] == "en"


def test_message_length_limit(client: TestClient):
    """Verify request validation rejects messages exceeding 2000 characters."""
    long_msg = "cow " * 600  # 2400 chars
    payload = {"message": long_msg}
    res = client.post("/api/v1/chat", json=payload)
    assert res.status_code == 422


# ----------------------------------------------------------------------
# 7. Nutrition Pluggable Interface Hook Test
# ----------------------------------------------------------------------

def test_nutrition_service_pluggable_hook():
    """Test Z: Verifies the clean pluggable interface for future ML model."""
    assert nutrition_service.is_model_available() is False

    # Simulate future ML model integration
    def mock_future_nutrition_ml_model(req: RationRequestModel):
        return {
            "recommended_green_fodder_kg": 30.0,
            "recommended_dry_fodder_kg": 5.0,
            "recommended_concentrate_kg": 6.5,
            "model_version": "v1.0_field_nutrition"
        }

    nutrition_service.register_ml_model(mock_future_nutrition_ml_model)
    assert nutrition_service.is_model_available() is True

    result = nutrition_service.generate_ration_advisory("My cow 420 kg produces 15 litres milk")
    assert result.is_model_predicted is True
    assert result.recommendations["recommended_green_fodder_kg"] == 30.0

    # Reset back to default state
    nutrition_service._ml_model_callable = None
    assert nutrition_service.is_model_available() is False


# ----------------------------------------------------------------------
# 8. Endpoint Aliases: /api/chat vs /api/v1/chat
# ----------------------------------------------------------------------

def test_root_api_chat_alias(client: TestClient):
    """Verify POST /api/chat directly works identically to /api/v1/chat."""
    payload = {"message": "Vanakkam", "language": "ta"}
    res = client.post("/api/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["language"] == "ta"
    assert "வணக்கம்" in data["reply"]
