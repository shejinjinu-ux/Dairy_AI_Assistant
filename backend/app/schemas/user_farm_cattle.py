"""
Pydantic Schemas for User, Farm, Cattle, and Analysis History
Enforces strict relational hierarchy: User -> Farm -> Cattle -> AnalysisRecord
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    """Authenticated user profile with demo environment flag."""
    user_id: str = Field(..., description="Unique user identifier derived from auth token/session.")
    email: Optional[str] = Field(default=None, description="User email address.")
    full_name: Optional[str] = Field(default=None, description="User display name.")
    is_demo: bool = Field(default=False, description="True if account is an isolated demo user sandbox.")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Farm(BaseModel):
    """Farm entity owned by a specific User."""
    farm_id: str = Field(..., description="Unique farm identifier.")
    user_id: str = Field(..., description="Owner user ID.")
    farm_name: str = Field(..., description="Name of the farm.")
    location: Optional[str] = Field(default=None, description="Geographic region/location.")
    is_demo: bool = Field(default=False, description="Flag indicating demo farm.")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Cattle(BaseModel):
    """Cattle/bovine entity belonging to a specific Farm and User."""
    animal_id: str = Field(..., description="Unique animal tag or identifier.")
    farm_id: str = Field(..., description="Parent farm ID.")
    user_id: str = Field(..., description="Owner user ID.")
    tag_number: Optional[str] = Field(default=None, description="Ear tag number.")
    species: str = Field(default="Cattle", description="Bovine species ('Cattle' or 'Buffalo').")
    breed: str = Field(default="Crossbred", description="Bovine breed (e.g. 'Gir', 'Sahiwal', 'Murrah', 'Holstein_Friesian').")
    age_months: Optional[float] = Field(default=None, description="Age in months.")
    body_weight_kg: float = Field(..., description="Live body weight in kg.")
    lactation_stage: Optional[str] = Field(default=None, description="Lactation phase: 'Early', 'Mid', 'Late', 'Dry'.")
    days_in_milk: Optional[float] = Field(default=None, description="Days in current milk cycle.")
    daily_milk_yield_litres: float = Field(..., description="Current daily milk yield in litres/day.")
    milk_fat_percentage: float = Field(..., description="Milk fat percentage (e.g., 4.0).")
    pregnancy_status: bool = Field(default=False, description="Pregnancy status.")
    pregnancy_month: Optional[int] = Field(default=None, description="Pregnancy month if pregnant.")
    is_demo: bool = Field(default=False, description="Flag indicating demo cattle.")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AnalysisRecord(BaseModel):
    """Persistent analysis record (feed, silage, disease, breed, NIR, milk yield)."""
    record_id: str = Field(..., description="Unique analysis record identifier.")
    user_id: str = Field(..., description="Owner user ID.")
    farm_id: Optional[str] = Field(default=None, description="Associated farm ID.")
    animal_id: Optional[str] = Field(default=None, description="Associated animal ID.")
    analysis_type: str = Field(..., description="Type of analysis: 'feed', 'silage', 'disease', 'breed', 'milk_quality_nir', 'milk_production'.")
    summary_status: str = Field(..., description="Overall status tier (e.g., 'EXCELLENT', 'GOOD', 'CAUTION', 'UNSAFE', 'FMD').")
    quality_score: Optional[float] = Field(default=None, description="Numerical score if applicable.")
    details: Dict[str, Any] = Field(default_factory=dict, description="Full raw analysis payload and metrics.")
    is_demo: bool = Field(default=False, description="Flag indicating demo record.")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class OwnershipValidationContext(BaseModel):
    """Result of user identity extraction and ownership verification."""
    is_authenticated: bool = Field(..., description="True if request has valid authenticated identity.")
    user_id: Optional[str] = Field(default=None, description="Authenticated user ID.")
    is_demo: bool = Field(default=False, description="True if authenticated user is demo user.")
    validated_farm: Optional[Farm] = Field(default=None, description="Validated Farm entity if farm_id supplied.")
    validated_cattle: Optional[Cattle] = Field(default=None, description="Validated Cattle entity if animal_id supplied.")
