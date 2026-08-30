"""
Fast2SMS OTP Service & Cryptographic Verification Engine
"""

import re
import time
import hmac
import secrets
import hashlib
import logging
import threading
from typing import Dict, List, Optional, Tuple
import httpx

from backend.config import settings
from backend.app.core.exceptions import (
    InvalidPhoneNumberError,
    OTPVerificationError,
    OTPRateLimitError,
    OTPProviderError,
)

logger = logging.getLogger("dairy_ai.auth.fast2sms")


def normalize_phone_number(raw_phone: str) -> Tuple[str, str]:
    """
    Validate and normalize phone numbers into:
      1. Canonical E.164 format (e.g. +919876543210)
      2. 10-digit Indian National format for Fast2SMS API (e.g. 9876543210)

    Handles Indian standard numbering (+91, 0-prefix, spaces, dashes):
      - 9876543210 -> (+919876543210, 9876543210)
      - +91 98765 43210 -> (+919876543210, 9876543210)
      - 09876543210 -> (+919876543210, 9876543210)
      - 919876543210 -> (+919876543210, 9876543210)
      - +919876543210 -> (+919876543210, 9876543210)
    """
    if not raw_phone or not isinstance(raw_phone, str):
        raise InvalidPhoneNumberError("Phone number is required.")

    cleaned = raw_phone.strip()
    if not cleaned:
        raise InvalidPhoneNumberError("Phone number cannot be blank.")

    has_plus = cleaned.startswith("+")
    digits_only = re.sub(r"\D", "", cleaned)

    if not digits_only:
        raise InvalidPhoneNumberError("Phone number must contain numeric digits.")

    # 1. Format with leading '+'
    if has_plus:
        if digits_only.startswith("91"):
            national_part = digits_only[2:]
            if len(national_part) != 10:
                raise InvalidPhoneNumberError(
                    f"Indian mobile numbers must have exactly 10 digits after +91 (received {len(national_part)} digits).",
                    phone=raw_phone
                )
            if national_part[0] not in "6789":
                raise InvalidPhoneNumberError(
                    "Indian mobile numbers must begin with 6, 7, 8, or 9.",
                    phone=raw_phone
                )
            return f"+91{national_part}", national_part

        raise InvalidPhoneNumberError(
            "Currently only Indian mobile numbers (+91) are supported for SMS OTP delivery.",
            phone=raw_phone
        )

    # 2. National / Local formats without leading '+'
    # Standard 10-digit Indian Mobile
    if len(digits_only) == 10:
        if digits_only[0] not in "6789":
            raise InvalidPhoneNumberError(
                "Indian 10-digit mobile numbers must start with 6, 7, 8, or 9.",
                phone=raw_phone
            )
        return f"+91{digits_only}", digits_only

    # 11-digit format starting with 0 (e.g. 09876543210)
    if len(digits_only) == 11 and digits_only.startswith("0"):
        national_part = digits_only[1:]
        if national_part[0] not in "6789":
            raise InvalidPhoneNumberError(
                "Indian mobile numbers must start with 6, 7, 8, or 9.",
                phone=raw_phone
            )
        return f"+91{national_part}", national_part

    # 12-digit format starting with 91 (e.g. 919876543210)
    if len(digits_only) == 12 and digits_only.startswith("91"):
        national_part = digits_only[2:]
        if national_part[0] not in "6789":
            raise InvalidPhoneNumberError(
                "Indian mobile numbers must start with 6, 7, 8, or 9.",
                phone=raw_phone
            )
        return f"+91{national_part}", national_part

    raise InvalidPhoneNumberError(
        "Invalid phone format. Please enter a valid 10-digit Indian mobile number.",
        phone=raw_phone
    )


def mask_phone_number(phone: str) -> str:
    """Masks phone number for safe telemetry and logging (e.g. +9198765****0)."""
    if not phone or len(phone) < 6:
        return "[MASKED]"
    return f"{phone[:7]}****{phone[-2:]}"


class OTPRateLimiter:
    """Thread-safe in-memory rate limiter and cooldown tracker per phone number."""

    def __init__(self):
        self._lock = threading.Lock()
        self._last_send: Dict[str, float] = {}
        self._history: Dict[str, List[float]] = {}

    def check_and_record_send(
        self,
        phone: str,
        cooldown_seconds: int = 60,
        max_sends: int = 5,
        window_seconds: int = 900
    ) -> None:
        """
        Enforce resend cooldown and rolling window rate limits.
        Raises OTPRateLimitError if rate limits are exceeded.
        """
        now = time.time()
        with self._lock:
            # 1. Check Cooldown
            last_time = self._last_send.get(phone, 0.0)
            elapsed = now - last_time
            if elapsed < cooldown_seconds:
                remaining = int(cooldown_seconds - elapsed) + 1
                logger.warning(
                    f"Rate limit: phone {mask_phone_number(phone)} requested OTP during cooldown ({remaining}s remaining)."
                )
                raise OTPRateLimitError(
                    f"Please wait {remaining} seconds before requesting a new OTP.",
                    retry_after=remaining,
                    phone=phone
                )

            # 2. Check Window Quota
            history = [t for t in self._history.get(phone, []) if (now - t) < window_seconds]
            if len(history) >= max_sends:
                logger.warning(
                    f"Rate limit: phone {mask_phone_number(phone)} exceeded {max_sends} OTP requests in {window_seconds}s."
                )
                raise OTPRateLimitError(
                    f"Too many OTP requests. Please try again after {window_seconds // 60} minutes.",
                    retry_after=window_seconds,
                    phone=phone
                )

            # Record this send attempt
            history.append(now)
            self._history[phone] = history
            self._last_send[phone] = now

    def reset(self, phone: Optional[str] = None) -> None:
        """Reset rate limiter state (used in automated test fixtures)."""
        with self._lock:
            if phone:
                self._last_send.pop(phone, None)
                self._history.pop(phone, None)
            else:
                self._last_send.clear()
                self._history.clear()


class SecureOTPStore:
    """
    Thread-safe in-memory store for salted hashes of active OTP codes.
    Never stores plaintext OTPs. Enforces single-use consumption and attempt limits.
    """

    def __init__(self):
        self._lock = threading.Lock()
        # Structure: phone -> {"hash": str, "salt": str, "expires_at": float, "attempts": int}
        self._records: Dict[str, dict] = {}

    def generate_and_store_otp(self, phone: str, expiry_seconds: int = 300) -> str:
        """
        Generates a cryptographically secure 6-digit OTP, stores its salted SHA-256 hash,
        and returns the raw OTP code to be dispatched via SMS provider.
        """
        # Cryptographically secure 6-digit numeric OTP (100000 to 999999)
        otp_code = str(secrets.randbelow(900000) + 100000)
        salt = secrets.token_hex(16)

        otp_hash = hashlib.sha256(f"{salt}:{otp_code}:{phone}".encode("utf-8")).hexdigest()
        expires_at = time.time() + expiry_seconds

        with self._lock:
            self._records[phone] = {
                "hash": otp_hash,
                "salt": salt,
                "expires_at": expires_at,
                "attempts": 0
            }

        return otp_code

    def verify_and_consume_otp(self, phone: str, user_otp: str, max_attempts: int = 5) -> bool:
        """
        Verifies the user OTP against the stored salted hash using constant-time comparison.
        On success, consumes/deletes the record (single-use).
        """
        now = time.time()
        with self._lock:
            record = self._records.get(phone)
            if not record:
                raise OTPVerificationError(
                    "No active OTP request found for this mobile number. Please request a new OTP.",
                    phone=phone
                )

            # Check max verification attempts
            if record["attempts"] >= max_attempts:
                self._records.pop(phone, None)
                logger.warning(f"OTP invalidation: max verification attempts reached for {mask_phone_number(phone)}.")
                raise OTPRateLimitError(
                    "Too many incorrect OTP attempts. This code has been invalidated. Please request a new OTP.",
                    retry_after=60,
                    phone=phone
                )

            # Check expiration
            if now > record["expires_at"]:
                self._records.pop(phone, None)
                logger.warning(f"OTP invalidation: expired code for {mask_phone_number(phone)}.")
                raise OTPVerificationError(
                    "Verification code has expired. Please request a new OTP.",
                    phone=phone
                )

            # Compute hash for comparison
            salt = record["salt"]
            candidate_hash = hashlib.sha256(f"{salt}:{user_otp}:{phone}".encode("utf-8")).hexdigest()

            # Constant-time comparison
            if hmac.compare_digest(record["hash"], candidate_hash):
                # Consume immediately upon success
                self._records.pop(phone, None)
                return True

            # Increment failed attempt counter
            record["attempts"] += 1
            remaining_attempts = max_attempts - record["attempts"]
            logger.warning(
                f"Failed OTP attempt for {mask_phone_number(phone)} ({remaining_attempts} remaining)."
            )
            raise OTPVerificationError(
                f"Invalid verification code. Please check and try again.",
                phone=phone
            )

    def reset(self, phone: Optional[str] = None) -> None:
        """Reset OTP store state (used in test fixtures)."""
        with self._lock:
            if phone:
                self._records.pop(phone, None)
            else:
                self._records.clear()


# Singletons for rate limiting & secure storage
otp_rate_limiter = OTPRateLimiter()
secure_otp_store = SecureOTPStore()


def _extract_safe_provider_error(response: httpx.Response, otp_code: str = "") -> str:
    """
    Safely extract Fast2SMS provider error message and response body without exposing secrets or OTP.
    """
    error_msg = ""
    try:
        data = response.json()
        if isinstance(data, dict):
            msgs = data.get("message")
            if isinstance(msgs, list) and msgs:
                error_msg = ", ".join(str(m) for m in msgs)
            elif msgs:
                error_msg = str(msgs)
            elif "status_message" in data:
                error_msg = str(data["status_message"])
            else:
                error_msg = str(data)
        elif isinstance(data, list):
            error_msg = ", ".join(str(m) for m in data)
        else:
            error_msg = str(data)
    except Exception:
        error_msg = response.text.strip()[:300] if response.text else "No response body"

    # Redact sensitive data if present
    if settings.FAST2SMS_API_KEY and str(settings.FAST2SMS_API_KEY).strip() in error_msg:
        error_msg = error_msg.replace(str(settings.FAST2SMS_API_KEY).strip(), "[REDACTED_API_KEY]")
    if otp_code and otp_code in error_msg:
        error_msg = error_msg.replace(otp_code, "[REDACTED_OTP]")

    return error_msg


class Fast2SMSService:
    """
    Fast2SMS OTP Service integrating the official Bulk SMS v2 OTP Route.
    """

    FAST2SMS_API_URL = "https://www.fast2sms.com/dev/bulkV2"

    def __init__(self, http_client: Optional[httpx.AsyncClient] = None):
        self._client = http_client

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def send_otp(self, raw_phone: str) -> dict:
        """
        Generate a secure OTP and deliver it via the Fast2SMS OTP API.
        Never logs or leaks the OTP code.
        """
        canonical_phone, national_10_digits = normalize_phone_number(raw_phone)

        # Enforce rate limiting and cooldown
        otp_rate_limiter.check_and_record_send(
            phone=canonical_phone,
            cooldown_seconds=settings.OTP_COOLDOWN_SECONDS,
            max_sends=settings.OTP_MAX_SENDS_PER_WINDOW,
            window_seconds=settings.OTP_WINDOW_SECONDS
        )

        if not settings.is_fast2sms_configured:
            logger.error("FAST2SMS_API_KEY is not configured on the backend server.")
            raise OTPProviderError(
                "SMS OTP service is currently unconfigured. Please configure FAST2SMS_API_KEY "
                "in your server environment variables.",
                provider="fast2sms"
            )

        # Generate cryptographically secure OTP and store salted hash
        otp_code = secure_otp_store.generate_and_store_otp(
            phone=canonical_phone,
            expiry_seconds=settings.OTP_EXPIRY_SECONDS
        )

        headers = {
            "authorization": settings.FAST2SMS_API_KEY,
            "Content-Type": "application/json"
        }
        payload = {
            "route": "otp",
            "variables_values": otp_code,
            "numbers": national_10_digits
        }

        client = await self._get_client()
        try:
            logger.info(f"Dispatching Fast2SMS OTP to {mask_phone_number(canonical_phone)}...")
            response = await client.post(self.FAST2SMS_API_URL, headers=headers, json=payload)

            if response.status_code == 200:
                res_data = response.json()
                if res_data.get("return") is True:
                    logger.info(f"Fast2SMS OTP successfully dispatched to {mask_phone_number(canonical_phone)}.")
                    return {
                        "success": True,
                        "phone": canonical_phone,
                        "status": "pending",
                        "request_id": res_data.get("request_id")
                    }
                else:
                    # Fast2SMS returned error message list
                    error_msg = _extract_safe_provider_error(response, otp_code)
                    logger.error(f"Fast2SMS API rejected request for {mask_phone_number(canonical_phone)}: {error_msg}")
                    raise OTPProviderError(f"Fast2SMS delivery error: {error_msg}", provider="fast2sms")

            error_details = _extract_safe_provider_error(response, otp_code)

            if response.status_code in [401, 403]:
                logger.error(f"Fast2SMS authentication failure (HTTP {response.status_code}): {error_details}")
                raise OTPProviderError(f"Fast2SMS authentication failed (HTTP {response.status_code}): {error_details}", provider="fast2sms")

            if response.status_code == 429:
                logger.warning(f"Fast2SMS provider rate limit exceeded (HTTP 429): {error_details}")
                raise OTPRateLimitError(
                    f"SMS provider rate limit reached ({error_details}). Please wait a few minutes before retrying.",
                    retry_after=120,
                    phone=canonical_phone
                )

            logger.error(f"Fast2SMS error (HTTP {response.status_code}) for {mask_phone_number(canonical_phone)}: {error_details}")
            raise OTPProviderError(f"Fast2SMS error (HTTP {response.status_code}): {error_details}", provider="fast2sms")

        except (InvalidPhoneNumberError, OTPRateLimitError, OTPProviderError):
            raise
        except httpx.RequestError as exc:
            logger.error(f"Network timeout/error connecting to Fast2SMS: {exc}")
            raise OTPProviderError("Network connection to SMS provider gateway failed or timed out.", provider="fast2sms")
        except Exception as exc:
            logger.error(f"Unexpected error in Fast2SMS send_otp: {exc}")
            raise OTPProviderError(f"SMS service error: {str(exc)}", provider="fast2sms")

    async def verify_otp(self, raw_phone: str, user_otp: str) -> dict:
        """
        Verify the user-provided OTP code against the secure salted hash store.
        """
        canonical_phone, _ = normalize_phone_number(raw_phone)

        if not user_otp or not user_otp.strip():
            raise OTPVerificationError("OTP code is required.", phone=canonical_phone)

        cleaned_otp = user_otp.strip()
        if not re.match(r"^\d{4,8}$", cleaned_otp):
            raise OTPVerificationError("OTP code must be 4 to 8 numeric digits.", phone=canonical_phone)

        logger.info(f"Verifying OTP code for {mask_phone_number(canonical_phone)}...")
        secure_otp_store.verify_and_consume_otp(phone=canonical_phone, user_otp=cleaned_otp)

        logger.info(f"OTP successfully verified and consumed for {mask_phone_number(canonical_phone)}.")
        return {
            "success": True,
            "verified": True,
            "phone": canonical_phone,
            "status": "approved"
        }


# Singleton service
fast2sms_service = Fast2SMSService()
