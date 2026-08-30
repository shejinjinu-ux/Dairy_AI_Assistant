"""
Core Exceptions & Exception Handlers
"""

from typing import Any, Dict, Optional
from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class AppBaseException(Exception):
    """Base application exception."""
    def __init__(self, message: str, status_code: int = 500, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class ModelNotFoundError(AppBaseException):
    """Raised when a requested model is not found in the registry."""
    def __init__(self, model_key: str):
        super().__init__(
            message=f"Model '{model_key}' was not found in the model registry.",
            status_code=status.HTTP_404_NOT_FOUND,
            details={"model_key": model_key}
        )


class ModelDisabledError(AppBaseException):
    """Raised when an experimental or disabled model is requested without activation."""
    def __init__(self, model_key: str, reason: str = "Experimental model disabled by default."):
        super().__init__(
            message=f"Model '{model_key}' is currently disabled. {reason}",
            status_code=status.HTTP_403_FORBIDDEN,
            details={
                "model_key": model_key,
                "hint": "Set ENABLE_EXPERIMENTAL_MODELS=true in backend configuration with calibrated sensor/lab telemetry to access this model."
            }
        )


class ModelInferenceError(AppBaseException):
    """Raised when an error occurs during model execution."""
    def __init__(self, model_key: str, error_detail: str):
        super().__init__(
            message=f"Inference failed for model '{model_key}': {error_detail}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"model_key": model_key, "error": error_detail}
        )


class ImageProcessingError(AppBaseException):
    """Raised when an uploaded image cannot be processed or decoded."""
    def __init__(self, detail: str):
        super().__init__(
            message=f"Invalid image input: {detail}",
            status_code=status.HTTP_400_BAD_REQUEST,
            details={"error": detail}
        )


class InvalidInputDataError(AppBaseException):
    """Raised when input tabular or sensor data fails validation."""
    def __init__(self, detail: str, missing_fields: Optional[list] = None):
        super().__init__(
            message=f"Invalid input data: {detail}",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details={"error": detail, "missing_fields": missing_fields or []}
        )


class InvalidPhoneNumberError(AppBaseException):
    """Raised when a supplied phone number fails validation or normalization."""
    def __init__(self, detail: str, phone: Optional[str] = None):
        super().__init__(
            message=f"Invalid phone number: {detail}",
            status_code=status.HTTP_400_BAD_REQUEST,
            details={"error": detail, "phone": phone}
        )


class OTPVerificationError(AppBaseException):
    """Raised when an OTP verification attempt fails, expires, or is rejected."""
    def __init__(self, detail: str, phone: Optional[str] = None):
        super().__init__(
            message=detail,
            status_code=status.HTTP_400_BAD_REQUEST,
            details={"error": detail, "phone": phone}
        )


class OTPRateLimitError(AppBaseException):
    """Raised when OTP send or verification requests exceed allowed rate limits or cooldown."""
    def __init__(self, detail: str, retry_after: int = 60, phone: Optional[str] = None):
        super().__init__(
            message=detail,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details={"error": detail, "retry_after_seconds": retry_after, "phone": phone}
        )


class OTPProviderError(AppBaseException):
    """Raised when the SMS/Fast2SMS OTP provider fails or is unconfigured."""
    def __init__(self, detail: str, provider: str = "fast2sms"):
        super().__init__(
            message=f"SMS OTP service error: {detail}",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details={"error": detail, "provider": provider}
        )


def register_exception_handlers(app: FastAPI) -> None:
    """Register uniform custom exception handlers on FastAPI application."""

    @app.exception_handler(AppBaseException)
    async def app_base_exception_handler(request: Request, exc: AppBaseException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error_type": exc.__class__.__name__,
                "message": exc.message,
                "details": jsonable_encoder(exc.details),
                "path": request.url.path
            }
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": False,
                "error_type": "RequestValidationError",
                "message": "Input validation failed against endpoint schema.",
                "details": jsonable_encoder(exc.errors()),
                "path": request.url.path
            }
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error_type": "InternalServerError",
                "message": "An unexpected error occurred during request processing.",
                "details": {"error": str(exc)},
                "path": request.url.path
            }
        )

