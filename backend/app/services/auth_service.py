"""
High-Level Authentication Service for Dairy AI Assistant
"""

import hashlib
import uuid
import logging
from typing import Optional

from backend.config import settings
from backend.app.schemas.auth import (
    SendOTPRequest,
    SendOTPResponse,
    VerifyOTPRequest,
    VerifyOTPResponse
)
from backend.app.services.fast2sms_service import (
    fast2sms_service,
    Fast2SMSService,
    mask_phone_number
)

logger = logging.getLogger("dairy_ai.auth.service")


class AuthService:
    """
    Orchestrates user authentication, real SMS OTP dispatches, and session binding.
    """

    def __init__(self, verify_service: Optional[Fast2SMSService] = None):
        self.verify_service = verify_service or fast2sms_service

    async def send_otp(self, request: SendOTPRequest) -> SendOTPResponse:
        """
        Request SMS OTP generation and delivery via Fast2SMS.
        """
        result = await self.verify_service.send_otp(request.phone)
        phone = result["phone"]

        return SendOTPResponse(
            success=True,
            message="OTP sent successfully to your mobile phone via SMS.",
            phone=phone,
            status=result.get("status", "pending"),
            cooldown_seconds=settings.OTP_COOLDOWN_SECONDS
        )

    async def verify_otp(self, request: VerifyOTPRequest) -> VerifyOTPResponse:
        """
        Validate SMS OTP code and return authentication credentials.
        """
        result = await self.verify_service.verify_otp(request.phone, request.otp)
        phone = result["phone"]

        # Derive a stable, non-reversible user ID tied to verified phone
        phone_hash = hashlib.sha256(f"dairy_ai_user_{phone}".encode("utf-8")).hexdigest()[:16]
        user_id = f"farmer_{phone_hash}"
        session_id = f"sess_{uuid.uuid4().hex[:16]}"

        logger.info(f"Authenticated user {user_id} with verified phone {mask_phone_number(phone)}.")

        return VerifyOTPResponse(
            success=True,
            message="Mobile number successfully verified.",
            verified=True,
            phone=phone,
            user_id=user_id,
            session_id=session_id,
            token_type="Bearer"
        )


auth_service = AuthService()
