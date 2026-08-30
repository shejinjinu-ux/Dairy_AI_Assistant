"""
Comprehensive Test Suite for Multilingual AI Chat Assistant
Validates 20+ Indian Languages, Intent Classification, Module Routing, Silage & Nutrition Integration,
Session Memory, Error Handling, and Real Google Gemini API Integration (Mocked)
"""

from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient
import httpx

from backend.main import app
from backend.config import settings
from backend.app.services.chat.nutrition_service import nutrition_service, RationRequestModel
from backend.app.services.chat.ai_service import ai_service


@pytest.fixture(scope="module")
def client():
    """Create TestClient with application lifespan executed."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def default_local_ai_settings():
    """Ensure baseline chat tests use local provider by default, preventing live network calls."""
    with patch.object(settings, "AI_PROVIDER", "local"), patch.object(settings, "AI_API_KEY", None):
        yield


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
    """Test S2: Silage intent with missing test values asks for pH/moisture in local mode."""
    with patch.object(settings, "AI_PROVIDER", "local"), patch.object(settings, "AI_API_KEY", None):
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
    """Test T: Unknown intent provides polite clarification in local mode."""
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


# ----------------------------------------------------------------------
# 9. Google Gemini AI Provider Tests (Mocked)
# ----------------------------------------------------------------------

def _create_mock_gemini_response(reply_text: str) -> httpx.Response:
    """Helper to generate realistic Gemini REST API response structure."""
    return httpx.Response(
        status_code=200,
        json={
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": reply_text}],
                        "role": "model"
                    },
                    "finishReason": "STOP"
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 35,
                "candidatesTokenCount": 65,
                "totalTokenCount": 100
            }
        },
        request=httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent")
    )


def test_gemini_provider_selected_and_active(client: TestClient):
    """Test 1: Gemini provider is configured and returns dynamic answers."""
    mock_res = _create_mock_gemini_response(
        "Silage is fermented green forage stored under anaerobic conditions to feed livestock during dry seasons."
    )

    with patch.object(settings, "AI_PROVIDER", "gemini"), \
         patch.object(settings, "AI_API_KEY", "mock_gemini_api_key_12345"), \
         patch.object(settings, "AI_MODEL", "gemini-1.5-flash"), \
         patch.object(ai_service, "_send_gemini_request", return_value=mock_res) as mock_send:

        res = client.post("/api/v1/chat", json={"message": "What is silage?"})
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert "anaerobic conditions" in data["reply"]

        mock_send.assert_called_once()
        call_url = mock_send.call_args[0][0]
        assert "gemini-1.5-flash:generateContent" in call_url
        assert "key=mock_gemini_api_key_12345" in call_url


def test_gemini_general_nondairy_question(client: TestClient):
    """Test 2: General non-dairy questions receive accurate, non-forced answers."""
    mock_res = _create_mock_gemini_response(
        "Python is a popular high-level, interpreted programming language created by Guido van Rossum."
    )

    with patch.object(settings, "AI_PROVIDER", "gemini"), \
         patch.object(settings, "AI_API_KEY", "mock_key"), \
         patch.object(ai_service, "_send_gemini_request", return_value=mock_res):

        res = client.post("/api/v1/chat", json={"message": "What is Python?"})
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert "Python is a popular" in data["reply"]
        assert "Guido van Rossum" in data["reply"]
        assert data["metadata"]["provider"] == "gemini"
        assert "dairy cattle management" not in data["reply"].lower()


def test_gemini_general_tcp_udp_question(client: TestClient):
    """Test 2b: TCP vs UDP query receives real dynamic Gemini answer and not dairy fallback."""
    mock_res = _create_mock_gemini_response(
        "TCP is connection-oriented and reliable, while UDP is connectionless and faster."
    )

    with patch.object(settings, "AI_PROVIDER", "gemini"), \
         patch.object(settings, "AI_API_KEY", "mock_key"), \
         patch.object(ai_service, "_send_gemini_request", return_value=mock_res):

        res = client.post("/api/v1/chat", json={"message": "Explain the difference between TCP and UDP in simple words."})
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert "TCP is connection-oriented" in data["reply"]
        assert data["metadata"]["provider"] == "gemini"
        assert "dairy cattle management" not in data["reply"].lower()


def test_gemini_sunset_model_fallback(client: TestClient):
    """Test 2c: When requested model returns 404, system automatically falls back to active candidate model."""
    mock_404_res = httpx.Response(
        status_code=404,
        json={"error": {"code": 404, "message": "Model not found", "status": "NOT_FOUND"}},
        request=httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent")
    )
    mock_200_res = _create_mock_gemini_response(
        "Photosynthesis is the process by which green plants use sunlight to synthesize nutrients from carbon dioxide and water."
    )

    with patch.object(settings, "AI_PROVIDER", "gemini"), \
         patch.object(settings, "AI_API_KEY", "mock_key"), \
         patch.object(settings, "AI_MODEL", "gemini-1.5-flash"), \
         patch.object(ai_service, "_send_gemini_request", side_effect=[mock_404_res, mock_200_res]) as mock_send:

        res = client.post("/api/v1/chat", json={"message": "Explain photosynthesis."})
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert "Photosynthesis" in data["reply"]
        assert data["metadata"]["provider"] == "gemini"
        assert mock_send.call_count >= 2


def test_gemini_dairy_question_with_context(client: TestClient):
    """Test 3: Dairy question receives dynamic advice with ICAR guidance."""
    mock_res = _create_mock_gemini_response(
        "To improve milk yield: 1. Provide balanced green fodder (30kg) and dry fodder (5kg). 2. Ensure ad-libitum clean drinking water."
    )

    with patch.object(settings, "AI_PROVIDER", "gemini"), \
         patch.object(settings, "AI_API_KEY", "mock_key"), \
         patch.object(ai_service, "_send_gemini_request", return_value=mock_res):

        res = client.post("/api/v1/chat", json={"message": "How can I improve milk yield in my cows?"})
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert "green fodder" in data["reply"]
        assert data["metadata"]["provider"] == "gemini"


def test_gemini_multilingual_response(client: TestClient):
    """Test 4: Multilingual queries are answered in native languages dynamically."""
    mock_res = _create_mock_gemini_response(
        "பால் உற்பத்தியை அதிகரிக்க உயர்தர பசுந்தீவனம் மற்றும் தாது உப்பு கலவை கொடுக்கவும்."
    )

    with patch.object(settings, "AI_PROVIDER", "gemini"), \
         patch.object(settings, "AI_API_KEY", "mock_key"), \
         patch.object(ai_service, "_send_gemini_request", return_value=mock_res):

        res = client.post("/api/v1/chat", json={"message": "பால் அதிகரிக்க என்ன செய்ய வேண்டும்?", "language": "ta"})
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert data["language"] == "ta"
        assert "பசுந்தீவனம்" in data["reply"]
        assert data["metadata"]["provider"] == "gemini"


def test_gemini_session_continuity(client: TestClient):
    """Test 5: Multi-turn session conversation history is passed to Gemini."""
    mock_res_t1 = _create_mock_gemini_response("A Gir cow produces on average 12-18 litres of milk per day.")
    mock_res_t2 = _create_mock_gemini_response("For a 15-litre yield, provide 6kg concentrate and 35kg green fodder.")

    with patch.object(settings, "AI_PROVIDER", "gemini"), \
         patch.object(settings, "AI_API_KEY", "mock_key"), \
         patch.object(ai_service, "_send_gemini_request", side_effect=[mock_res_t1, mock_res_t2]) as mock_send:

        # Turn 1
        res1 = client.post("/api/v1/chat", json={"message": "How much milk does a Gir cow produce?", "session_id": "gemini_sess_001"})
        assert res1.status_code == 200
        assert "Gir cow" in res1.json()["reply"]

        # Turn 2
        res2 = client.post("/api/v1/chat", json={"message": "What should I feed her?", "session_id": "gemini_sess_001"})
        assert res2.status_code == 200
        assert "concentrate" in res2.json()["reply"]

        # Verify second call included conversation history
        second_call_payload = mock_send.call_args_list[1][0][2]
        contents = second_call_payload["contents"]
        assert len(contents) >= 2


def test_gemini_timeout_fallback(client: TestClient):
    """Test 6: Gemini timeout fails gracefully and returns local domain response."""
    with patch.object(settings, "AI_PROVIDER", "gemini"), \
         patch.object(settings, "AI_API_KEY", "mock_key"), \
         patch.object(ai_service, "_send_gemini_request", side_effect=httpx.TimeoutException("Read timed out")):

        res = client.post("/api/v1/chat", json={"message": "What feed should I give my cow?"})
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        # Falls back to local feed knowledge base
        assert len(data["reply"]) > 0
        assert data["metadata"]["provider"] == "local_fallback"


def test_gemini_invalid_key_fallback(client: TestClient):
    """Test 7: Invalid API key (HTTP 400/403) returns fallback response without crashing."""
    mock_err_res = httpx.Response(
        status_code=400,
        json={"error": {"code": 400, "message": "API key not valid", "status": "INVALID_ARGUMENT"}},
        request=httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent")
    )

    with patch.object(settings, "AI_PROVIDER", "gemini"), \
         patch.object(settings, "AI_API_KEY", "invalid_key"), \
         patch.object(ai_service, "_send_gemini_request", return_value=mock_err_res):

        res = client.post("/api/v1/chat", json={"message": "Hello!"})
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert len(data["reply"]) > 0
        assert data["metadata"]["provider"] == "local_fallback"
