"""
Health & System Status Schemas
"""

from typing import Dict, Any, List
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """System Health Check Response"""
    status: str = Field(default="healthy", description="Overall service health status")
    version: str = Field(..., description="API Version")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="UTC Timestamp")
    device: str = Field(..., description="PyTorch inference device (cpu/cuda)")
    models_loaded: int = Field(..., description="Number of currently loaded models in memory cache")
    production_models_ready: int = Field(..., description="Total production models available")
    experimental_models_enabled: bool = Field(..., description="Whether experimental models are enabled")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional health diagnostics")


class PingResponse(BaseModel):
    """Lightweight ping response"""
    ping: str = "pong"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
