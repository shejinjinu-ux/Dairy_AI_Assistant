"""
Authentication & OTP Request/Response Schemas
"""

import re
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class SendOTPRequest(BaseModel):
    """Request payload to initiate SMS OTP delivery."""
    phone: str = Field(
        ...,
        min_length=7,
        max_length=25,
        description="Mobile phone number to receive real SMS OTP (Indian +91 or E.164 format).",
        examples=["+919876543210", "9876543210"]
    )

    @field_validator("phone")
    @classmethod
    def validate_phone_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Phone number must not be empty.")
        return v.strip()


class SendOTPResponse(BaseModel):
    """Response returned upon successful SMS OTP dispatch."""
    success: bool = Field(default=True, description="Indicates whether OTP was successfully dispatched.")
    message: str = Field(..., description="User-friendly status message.")
    phone: str = Field(..., description="Normalized E.164 phone number.")
    status: str = Field(default="pending", description="Verification delivery status (e.g. 'pending').")
    cooldown_seconds: int = Field(default=60, description="Cooldown period in seconds before resending.")


class VerifyOTPRequest(BaseModel):
    """Request payload to verify the received SMS OTP."""
    phone: str = Field(
        ...,
        min_length=7,
        max_length=25,
        description="Mobile phone number that received the SMS OTP.",
        examples=["+919876543210", "9876543210"]
    )
    otp: str = Field(
        ...,
        min_length=4,
        max_length=8,
        description="Numeric SMS verification code received on device.",
        examples=["123456"]
    )

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Phone number must not be empty.")
        return v.strip()

    @field_validator("otp")
    @classmethod
    def validate_otp(cls, v: str) -> str:
        cleaned = v.strip()
        if not re.match(r"^\d{4,8}$", cleaned):
            raise ValueError("OTP code must consist of 4 to 8 numeric digits.")
        return cleaned


class VerifyOTPResponse(BaseModel):
    """Response returned upon successful SMS OTP verification."""
    success: bool = Field(default=True, description="Whether OTP verification succeeded.")
    message: str = Field(..., description="Status message.")
    verified: bool = Field(default=True, description="Verification status.")
    phone: str = Field(..., description="Normalized E.164 phone number.")
    user_id: Optional[str] = Field(default=None, description="Consistent user identifier for dairy AI services.")
    session_id: Optional[str] = Field(default=None, description="Optional active session ID.")
    token_type: str = Field(default="Bearer", description="Token authentication scheme.")
