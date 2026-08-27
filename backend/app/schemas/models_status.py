"""
Model Registry & Status Schemas
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class ModelDescriptor(BaseModel):
    """Metadata descriptor for an individual AI/ML model."""
    key: str = Field(..., description="Unique model registry key identifier")
    status: str = Field(..., description="Deployment status: 'production' or 'experimental'")
    framework: str = Field(..., description="Framework: 'pytorch' or 'xgboost'")
    task: str = Field(..., description="Machine learning task type")
    path: str = Field(..., description="Relative filesystem path from project root")
    is_enabled: bool = Field(..., description="Whether model is currently enabled for inference")
    is_cached_in_memory: bool = Field(..., description="Whether model weights are loaded in RAM/VRAM")
    target: Optional[str] = Field(None, description="Regression target column name")
    outputs: Optional[List[str]] = Field(None, description="Output classification class names")
    classes_count: Optional[int] = Field(None, description="Number of output classes")
    accuracy: Optional[float] = Field(None, description="Benchmark accuracy score")
    r2_score: Optional[float] = Field(None, description="Benchmark R2 determination coefficient")
    f1_macro: Optional[float] = Field(None, description="Macro F1 score")


class ModelRegistryStatusResponse(BaseModel):
    """Overall Model Registry Status Response"""
    version: str = Field(..., description="Registry schema version")
    total_models: int = Field(..., description="Total models registered")
    production_count: int = Field(..., description="Total production-grade models")
    experimental_count: int = Field(..., description="Total experimental models")
    experimental_enabled: bool = Field(..., description="Current experimental model flag state")
    models: List[ModelDescriptor] = Field(..., description="List of all registered models and their states")
