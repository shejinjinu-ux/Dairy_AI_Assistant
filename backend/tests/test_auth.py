"""
Unit & Integration Tests for SMS OTP Authentication with Fast2SMS
"""

import time
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
import httpx

from backend.main import app
from backend.config import settings
from backend.app.services.fast2sms_service import (
    normalize_phone_number,
    mask_phone_number,
    otp_rate_limiter,
    secure_otp_store,
    fast2sms_service,
)
from backend.app.core.exceptions import (
    InvalidPhoneNumberError,
    OTPVerificationError,
    OTPRateLimitError,
    OTPProviderError,
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_stores():
    """Reset rate limiter and OTP store before each test."""
    otp_rate_limiter.reset()
    secure_otp_store.reset()
    yield
    otp_rate_limiter.reset()
    secure_otp_store.reset()


# ==============================================================================
# 1. Phone Normalization & Masking Tests
# ==============================================================================

def test_indian_phone_normalization_standard_10_digits():
    canonical, national = normalize_phone_number("9876543210")
    assert canonical == "+919876543210"
    assert national == "9876543210"

    canonical2, national2 = normalize_phone_number("8123456789")
    assert canonical2 == "+918123456789"
    assert national2 == "8123456789"


def test_indian_phone_normalization_with_formatting():
    assert normalize_phone_number("+91 98765 43210")[0] == "+919876543210"
    assert normalize_phone_number("+91-98765-43210")[0] == "+919876543210"
    assert normalize_phone_number("09876543210")[0] == "+919876543210"
    assert normalize_phone_number("919876543210")[0] == "+919876543210"
    assert normalize_phone_number("+919876543210")[0] == "+919876543210"


def test_invalid_phone_numbers():
    # Too short
    with pytest.raises(InvalidPhoneNumberError):
        normalize_phone_number("12345")

    # Invalid Indian prefix (starts with 1, 2, 3, 4, 5)
    with pytest.raises(InvalidPhoneNumberError):
        normalize_phone_number("1234567890")
    with pytest.raises(InvalidPhoneNumberError):
        normalize_phone_number("+911234567890")

    # Alpha characters
    with pytest.raises(InvalidPhoneNumberError):
        normalize_phone_number("98765abcdef")

    # Empty
    with pytest.raises(InvalidPhoneNumberError):
        normalize_phone_number("")


def test_phone_masking():
    assert mask_phone_number("+919876543210") == "+919876****10"
    assert mask_phone_number("") == "[MASKED]"


# ==============================================================================
# 2. Rate Limiting Unit Tests
# ==============================================================================

def test_rate_limiter_cooldown():
    phone = "+919876543210"
    otp_rate_limiter.check_and_record_send(phone, cooldown_seconds=60, max_sends=5, window_seconds=900)

    with pytest.raises(OTPRateLimitError) as exc_info:
        otp_rate_limiter.check_and_record_send(phone, cooldown_seconds=60, max_sends=5, window_seconds=900)

    assert "wait" in str(exc_info.value).lower()
    assert exc_info.value.status_code == 429


def test_rate_limiter_window_quota():
    phone = "+919876543210"
    for _ in range(5):
        otp_rate_limiter._last_send[phone] = 0.0  # bypass cooldown
        otp_rate_limiter.check_and_record_send(phone, cooldown_seconds=0, max_sends=5, window_seconds=900)

    with pytest.raises(OTPRateLimitError) as exc_info:
        otp_rate_limiter.check_and_record_send(phone, cooldown_seconds=0, max_sends=5, window_seconds=900)

    assert "too many" in str(exc_info.value).lower()


# ==============================================================================
# 3. Secure OTP Store Tests
# ==============================================================================

def test_secure_otp_store_verification():
    phone = "+919876543210"
    otp = secure_otp_store.generate_and_store_otp(phone, expiry_seconds=300)

    # Valid OTP succeeds
    assert secure_otp_store.verify_and_consume_otp(phone, otp) is True

    # Single-use: Subsequent verification fails
    with pytest.raises(OTPVerificationError):
        secure_otp_store.verify_and_consume_otp(phone, otp)


def test_secure_otp_store_invalid_code():
    phone = "+919876543210"
    secure_otp_store.generate_and_store_otp(phone, expiry_seconds=300)

    with pytest.raises(OTPVerificationError) as exc_info:
        secure_otp_store.verify_and_consume_otp(phone, "000000")
    assert "invalid" in str(exc_info.value).lower()


def test_secure_otp_store_expiration():
    phone = "+919876543210"
    otp = secure_otp_store.generate_and_store_otp(phone, expiry_seconds=1)

    # Simulate expiration
    secure_otp_store._records[phone]["expires_at"] = time.time() - 10

    with pytest.raises(OTPVerificationError) as exc_info:
        secure_otp_store.verify_and_consume_otp(phone, otp)
    assert "expired" in str(exc_info.value).lower()


def test_secure_otp_store_max_attempts():
    phone = "+919876543210"
    secure_otp_store.generate_and_store_otp(phone, expiry_seconds=300)

    # 5 failed attempts
    for _ in range(5):
        try:
            secure_otp_store.verify_and_consume_otp(phone, "111111")
        except OTPVerificationError:
            pass

    # 6th attempt triggers rate limit / code invalidation
    with pytest.raises(OTPRateLimitError):
        secure_otp_store.verify_and_consume_otp(phone, "111111")


# ==============================================================================
# 4. POST /api/v1/auth/send-otp Endpoint Tests (Fast2SMS)
# ==============================================================================

def test_send_otp_success_mocked():
    """Test successful OTP dispatch with Fast2SMS mocked."""
    mock_response = httpx.Response(
        status_code=200,
        json={
            "return": True,
            "request_id": "fast2sms_req_001",
            "message": ["SMS sent successfully"]
        },
        request=httpx.Request("POST", "https://www.fast2sms.com/dev/bulkV2")
    )

    with patch.object(settings, "FAST2SMS_API_KEY", "mock_fast2sms_key_12345"), \
         patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response) as mock_post:

        response = client.post(
            "/api/v1/auth/send-otp",
            json={"phone": "9876543210"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["phone"] == "+919876543210"
        assert data["status"] == "pending"
        assert data["cooldown_seconds"] == 60
        assert "otp" not in data  # Never expose OTP

        # Verify Fast2SMS call structure
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["headers"]["authorization"] == "mock_fast2sms_key_12345"
        assert call_kwargs["json"]["route"] == "otp"
        assert call_kwargs["json"]["numbers"] == "9876543210"
        assert len(call_kwargs["json"]["variables_values"]) == 6


def test_send_otp_unconfigured_fast2sms_production():
    """Test response when FAST2SMS_API_KEY is not configured in production mode."""
    with patch.object(settings, "ENVIRONMENT", "production"), \
         patch.object(settings, "FAST2SMS_API_KEY", None):
        response = client.post(
            "/api/v1/auth/send-otp",
            json={"phone": "9876543210"}
        )

        assert response.status_code == 503
        data = response.json()
        assert data["success"] is False
        assert data["error_type"] == "OTPProviderError"
        assert "unconfigured" in data["message"].lower()


def test_send_otp_unconfigured_fast2sms_demo_mode():
    """Test that send-otp succeeds in development/demo mode even when FAST2SMS_API_KEY is unconfigured."""
    with patch.object(settings, "ENVIRONMENT", "development"), \
         patch.object(settings, "ENABLE_DEMO_OTP", True), \
         patch.object(settings, "FAST2SMS_API_KEY", None):
        response = client.post(
            "/api/v1/auth/send-otp",
            json={"phone": "9876543210"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["phone"] == "+919876543210"
        assert data["status"] == "pending"
        assert data["cooldown_seconds"] == 60


def test_send_otp_invalid_phone_format():
    """Test rejection of invalid phone formats."""
    # 10 digits with invalid prefix
    res1 = client.post("/api/v1/auth/send-otp", json={"phone": "1234567890"})
    assert res1.status_code == 400
    assert res1.json()["error_type"] == "InvalidPhoneNumberError"

    # Too short
    res2 = client.post("/api/v1/auth/send-otp", json={"phone": "12345"})
    assert res2.status_code == 422


def test_send_otp_empty_payload():
    """Test rejection of empty phone payload."""
    response = client.post("/api/v1/auth/send-otp", json={"phone": ""})
    assert response.status_code == 422


def test_send_otp_cooldown_rate_limit():
    """Test that rapid consecutive requests return HTTP 429."""
    mock_response = httpx.Response(
        status_code=200,
        json={"return": True, "request_id": "req1", "message": ["SMS sent successfully"]},
        request=httpx.Request("POST", "https://www.fast2sms.com/dev/bulkV2")
    )

    with patch.object(settings, "FAST2SMS_API_KEY", "mock_key"), \
         patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):

        res1 = client.post("/api/v1/auth/send-otp", json={"phone": "+919876543210"})
        assert res1.status_code == 200

        res2 = client.post("/api/v1/auth/send-otp", json={"phone": "+919876543210"})
        assert res2.status_code == 429
        assert res2.json()["error_type"] == "OTPRateLimitError"


def test_send_otp_fast2sms_api_failure_production():
    """Test handling when Fast2SMS API returns return=False in production mode."""
    mock_response = httpx.Response(
        status_code=200,
        json={"return": False, "message": ["Insufficient account balance"]},
        request=httpx.Request("POST", "https://www.fast2sms.com/dev/bulkV2")
    )

    with patch.object(settings, "ENVIRONMENT", "production"), \
         patch.object(settings, "FAST2SMS_API_KEY", "mock_key"), \
         patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):

        response = client.post("/api/v1/auth/send-otp", json={"phone": "+919876543210"})
        assert response.status_code == 503
        data = response.json()
        assert data["success"] is False
        assert "Insufficient account balance" in data["message"]


def test_send_otp_fast2sms_timeout_production():
    """Test handling when Fast2SMS request times out in production mode."""
    with patch.object(settings, "ENVIRONMENT", "production"), \
         patch.object(settings, "FAST2SMS_API_KEY", "mock_key"), \
         patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=httpx.TimeoutException("Timeout")):

        response = client.post("/api/v1/auth/send-otp", json={"phone": "+919876543210"})
        assert response.status_code == 503
        data = response.json()
        assert data["success"] is False
        assert "timed out" in data["message"].lower() or "network" in data["message"].lower()


# ==============================================================================
# 5. POST /api/v1/auth/verify-otp Endpoint Tests (Fast2SMS & Demo Mode)
# ==============================================================================

def test_verify_otp_approved_success():
    """Test successful OTP verification flow."""
    phone = "+919876543210"
    otp = secure_otp_store.generate_and_store_otp(phone, expiry_seconds=300)

    response = client.post(
        "/api/v1/auth/verify-otp",
        json={"phone": phone, "otp": otp}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["verified"] is True
    assert data["phone"] == "+919876543210"
    assert data["user_id"].startswith("farmer_")
    assert data["session_id"].startswith("sess_")
    assert "otp" not in data


def test_verify_otp_invalid_code():
    """Test rejection when wrong OTP is entered."""
    phone = "+919876543210"
    secure_otp_store.generate_and_store_otp(phone, expiry_seconds=300)

    response = client.post(
        "/api/v1/auth/verify-otp",
        json={"phone": phone, "otp": "999999"}
    )

    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert data["error_type"] == "OTPVerificationError"
    assert "invalid" in data["message"].lower()


def test_verify_otp_no_active_request_production():
    """Test verification in production when no OTP was requested for that phone."""
    with patch.object(settings, "ENVIRONMENT", "production"):
        response = client.post(
            "/api/v1/auth/verify-otp",
            json={"phone": "+919876543210", "otp": "123456"}
        )

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert data["error_type"] == "OTPVerificationError"
        assert "no active otp request" in data["message"].lower()


def test_verify_otp_invalid_code_format():
    """Test schema validation rejecting non-numeric or malformed OTP codes."""
    res1 = client.post(
        "/api/v1/auth/verify-otp",
        json={"phone": "+919876543210", "otp": "abcdef"}
    )
    assert res1.status_code == 422

    res2 = client.post(
        "/api/v1/auth/verify-otp",
        json={"phone": "+919876543210", "otp": "12"}
    )
    assert res2.status_code == 422


# ==============================================================================
# 6. DEMO OTP FLOW & SECURITY TESTS (123456)
# ==============================================================================

def test_demo_otp_123456_succeeds_in_development_mode():
    """Test that Demo OTP 123456 successfully verifies in development/demo mode."""
    with patch.object(settings, "ENVIRONMENT", "development"), \
         patch.object(settings, "ENABLE_DEMO_OTP", True):

        # Request demo OTP
        send_res = client.post("/api/v1/auth/send-otp", json={"phone": "9876543210"})
        assert send_res.status_code == 200

        # Verify with 123456
        verify_res = client.post(
            "/api/v1/auth/verify-otp",
            json={"phone": "9876543210", "otp": "123456"}
        )
        assert verify_res.status_code == 200
        data = verify_res.json()
        assert data["success"] is True
        assert data["verified"] is True
        assert data["phone"] == "+919876543210"
        assert data["user_id"].startswith("farmer_")
        assert data["session_id"].startswith("sess_")
        assert data["token_type"] == "Bearer"


def test_demo_otp_wrong_code_fails_in_development_mode():
    """Test that entering incorrect OTP (e.g. 999999) fails even in development mode."""
    with patch.object(settings, "ENVIRONMENT", "development"), \
         patch.object(settings, "ENABLE_DEMO_OTP", True):

        # Request demo OTP
        client.post("/api/v1/auth/send-otp", json={"phone": "9876543210"})

        # Enter wrong code
        verify_res = client.post(
            "/api/v1/auth/verify-otp",
            json={"phone": "9876543210", "otp": "999999"}
        )
        assert verify_res.status_code == 400
        assert verify_res.json()["error_type"] == "OTPVerificationError"


def test_default_configuration_does_not_enable_demo_otp():
    """Test that the default backend configuration does NOT enable Demo OTP (secure by default)."""
    # Verify Settings defaults
    from backend.config import Settings
    fresh_settings = Settings()
    assert fresh_settings.ENABLE_DEMO_OTP is False
    assert fresh_settings.is_demo_otp_active is False

    # Attempting demo OTP verification with default settings must fail
    verify_res = client.post(
        "/api/v1/auth/verify-otp",
        json={"phone": "9876543210", "otp": "123456"}
    )
    assert verify_res.status_code == 400
    assert verify_res.json()["error_type"] == "OTPVerificationError"


def test_demo_otp_rejected_in_production_mode():
    """Test that Demo OTP 123456 is strictly rejected in production mode."""
    with patch.object(settings, "ENVIRONMENT", "production"), \
         patch.object(settings, "ENABLE_DEMO_OTP", True):

        # Ensure is_demo_otp_active is False even when ENABLE_DEMO_OTP is True
        assert settings.is_demo_otp_active is False

        # Attempt to verify with 123456 in production without real OTP issued
        verify_res = client.post(
            "/api/v1/auth/verify-otp",
            json={"phone": "9876543210", "otp": "123456"}
        )
        assert verify_res.status_code == 400
        assert verify_res.json()["error_type"] == "OTPVerificationError"


def test_demo_otp_rejected_when_flag_disabled():
    """Test that Demo OTP 123456 is rejected when ENABLE_DEMO_OTP is False."""
    with patch.object(settings, "ENVIRONMENT", "development"), \
         patch.object(settings, "ENABLE_DEMO_OTP", False):

        assert settings.is_demo_otp_active is False

        verify_res = client.post(
            "/api/v1/auth/verify-otp",
            json={"phone": "9876543210", "otp": "123456"}
        )
        assert verify_res.status_code == 400
        assert verify_res.json()["error_type"] == "OTPVerificationError"


def test_demo_otp_creates_valid_authoritative_session_and_accesses_protected_endpoints():
    """Test that Demo OTP verification creates a registered session usable with protected APIs."""
    from backend.app.core.ownership_guard import auth_session_store

    with patch.object(settings, "ENVIRONMENT", "development"), \
         patch.object(settings, "ENABLE_DEMO_OTP", True):

        # Login with Demo OTP
        verify_res = client.post(
            "/api/v1/auth/verify-otp",
            json={"phone": "9876543210", "otp": "123456"}
        )
        assert verify_res.status_code == 200
        auth_data = verify_res.json()
        session_id = auth_data["session_id"]
        user_id = auth_data["user_id"]

        # Confirm session is authoritatively registered in session store
        assert auth_session_store.is_valid_session(session_id) is True
        assert auth_session_store.get_user_id(session_id) == user_id

        # Use the Bearer session token to create a cattle record
        auth_headers = {"Authorization": f"Bearer {session_id}"}
        create_cow_res = client.post(
            "/api/v1/cattle",
            json={
                "tag_id": "TAG-DEMO-LOGIN-01",
                "name": "Kamadhenu",
                "breed": "Gir",
                "daily_milk_yield_litres": 14.5
            },
            headers=auth_headers
        )
        assert create_cow_res.status_code == 201

        # Query cattle list using the Bearer session token
        list_res = client.get("/api/v1/cattle", headers=auth_headers)
        assert list_res.status_code == 200
        cattle_list = list_res.json()
        assert any(c["tag_id"] == "TAG-DEMO-LOGIN-01" for c in cattle_list)


def test_demo_otp_session_anti_spoofing_rejected():
    """Test that client attempting X-User-ID spoofing with a demo session is rejected with HTTP 403."""
    with patch.object(settings, "ENVIRONMENT", "development"), \
         patch.object(settings, "ENABLE_DEMO_OTP", True):

        verify_res = client.post(
            "/api/v1/auth/verify-otp",
            json={"phone": "9876543210", "otp": "123456"}
        )
        session_id = verify_res.json()["session_id"]

        # Session belongs to farmer derived from +919876543210, but client passes X-User-ID: usr_attacker
        spoofed_headers = {
            "Authorization": f"Bearer {session_id}",
            "X-User-ID": "usr_attacker_999"
        }
        res = client.get("/api/v1/cattle", headers=spoofed_headers)
        assert res.status_code == 403
        assert "Security violation" in res.json()["detail"] or "does not match" in res.json()["detail"]


def test_demo_otp_user_isolation_between_phone_numbers():
    """Test that two farmers logging in via demo OTP receive isolated identities and cattle."""
    with patch.object(settings, "ENVIRONMENT", "development"), \
         patch.object(settings, "ENABLE_DEMO_OTP", True):

        # Farmer 1 Login
        res1 = client.post("/api/v1/auth/verify-otp", json={"phone": "9876543210", "otp": "123456"})
        session_1 = res1.json()["session_id"]
        user_1 = res1.json()["user_id"]

        # Farmer 2 Login
        res2 = client.post("/api/v1/auth/verify-otp", json={"phone": "8123456789", "otp": "123456"})
        session_2 = res2.json()["session_id"]
        user_2 = res2.json()["user_id"]

        # Ensure distinct user identities
        assert user_1 != user_2
        assert session_1 != session_2

        # Farmer 1 adds a cow
        client.post(
            "/api/v1/cattle",
            json={"tag_id": "TAG-FARMER1-COW", "name": "Ganga", "daily_milk_yield_litres": 15.0},
            headers={"Authorization": f"Bearer {session_1}"}
        )

        # Farmer 2 adds a cow
        client.post(
            "/api/v1/cattle",
            json={"tag_id": "TAG-FARMER2-COW", "name": "Yamuna", "daily_milk_yield_litres": 20.0},
            headers={"Authorization": f"Bearer {session_2}"}
        )

        # Farmer 1 views cattle list -> Only sees TAG-FARMER1-COW
        f1_cows = client.get("/api/v1/cattle", headers={"Authorization": f"Bearer {session_1}"}).json()
        assert any(c["tag_id"] == "TAG-FARMER1-COW" for c in f1_cows)
        assert not any(c["tag_id"] == "TAG-FARMER2-COW" for c in f1_cows)

        # Farmer 2 views cattle list -> Only sees TAG-FARMER2-COW
        f2_cows = client.get("/api/v1/cattle", headers={"Authorization": f"Bearer {session_2}"}).json()
        assert any(c["tag_id"] == "TAG-FARMER2-COW" for c in f2_cows)
        assert not any(c["tag_id"] == "TAG-FARMER1-COW" for c in f2_cows)


# ==============================================================================
# 7. Direct Route Compatibility (/api/auth)
# ==============================================================================

def test_direct_api_auth_route_mounted():
    """Verify /api/auth/send-otp is mounted alongside /api/v1/auth/send-otp."""
    mock_response = httpx.Response(
        status_code=200,
        json={"return": True, "request_id": "req_direct", "message": ["SMS sent successfully"]},
        request=httpx.Request("POST", "https://www.fast2sms.com/dev/bulkV2")
    )

    with patch.object(settings, "FAST2SMS_API_KEY", "mock_key"), \
         patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):

        res = client.post("/api/auth/send-otp", json={"phone": "+919876543210"})
        assert res.status_code == 200
        assert res.json()["success"] is True
