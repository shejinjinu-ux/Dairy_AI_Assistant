"""
Authentication & SMS OTP Endpoints
"""

import logging
from fastapi import APIRouter, status

from backend.app.schemas.auth import (
    SendOTPRequest,
    SendOTPResponse,
    VerifyOTPRequest,
    VerifyOTPResponse
)
from backend.app.services.auth_service import auth_service

logger = logging.getLogger("dairy_ai.api.auth")

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/send-otp",
    response_model=SendOTPResponse,
    status_code=status.HTTP_200_OK,
    summary="Send real SMS OTP to Indian mobile number via Fast2SMS",
    description=(
        "Validates the phone number (with native support for Indian +91 formats), "
        "enforces resend cooldowns and rate limits, and triggers a real SMS OTP via Fast2SMS. "
        "Does NOT return the OTP in the API response."
    )
)
async def send_otp(request: SendOTPRequest) -> SendOTPResponse:
    """
    Send real SMS OTP to user's phone number.
    """
    return await auth_service.send_otp(request)


@router.post(
    "/verify-otp",
    response_model=VerifyOTPResponse,
    status_code=status.HTTP_200_OK,
    summary="Verify SMS OTP code via Fast2SMS secure verification engine",
    description=(
        "Verifies the SMS OTP code against the secure salted hash store. "
        "Returns authentication and session details upon successful approval. "
        "Rejects expired, invalid, or already-used codes."
    )
)
async def verify_otp(request: VerifyOTPRequest) -> VerifyOTPResponse:
    """
    Verify user-provided OTP code.
    """
    return await auth_service.verify_otp(request)
