"""
Pydantic Schemas for User, Farm, Cattle, Milk History, Vaccination, and Analysis History
Enforces strict relational hierarchy: User -> Farm -> Cattle -> (MilkRecords, VaccinationRecords, AnalysisRecords)
All Tag IDs are globally unique, normalized, and permanently persisted.
"""

from datetime import datetime, timezone, date
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class UserProfile(BaseModel):
    """Authenticated user profile with demo environment flag."""
    user_id: str = Field(..., description="Unique user identifier derived from auth token/session.")
    email: Optional[str] = Field(default=None, description="User email address.")
    full_name: Optional[str] = Field(default=None, description="User display name.")
    phone: Optional[str] = Field(default=None, description="User mobile phone number.")
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
    """Cattle/bovine entity belonging to a specific Farm and User with globally unique Tag ID."""
    animal_id: str = Field(..., description="Unique animal tag or identifier (synonymous with tag_id).")
    tag_id: str = Field(..., description="Globally unique animal Tag ID (e.g. 'COW-1001').")
    farm_id: str = Field(..., description="Parent farm ID.")
    user_id: str = Field(..., description="Owner user ID.")
    tag_number: Optional[str] = Field(default=None, description="Ear tag number display string.")
    name: Optional[str] = Field(default=None, description="Optional colloquial name (e.g. 'Lakshmi').")
    species: str = Field(default="Cattle", description="Bovine species ('Cattle' or 'Buffalo').")
    breed: str = Field(default="Crossbred", description="Bovine breed (e.g. 'Gir', 'Sahiwal', 'Murrah', 'Holstein_Friesian').")
    gender: str = Field(default="Female", description="Sex of animal ('Female' or 'Male').")
    age_months: Optional[float] = Field(default=None, description="Age in months.")
    date_of_birth: Optional[str] = Field(default=None, description="Date of birth in YYYY-MM-DD format if known.")
    body_weight_kg: float = Field(default=400.0, description="Live body weight in kg.")

    # Lactation & Reproduction
    calving_date: Optional[str] = Field(default=None, description="Most recent calving date (YYYY-MM-DD).")
    lactation_start_date: Optional[str] = Field(default=None, description="Lactation start date (YYYY-MM-DD).")
    parity: int = Field(default=1, ge=1, le=20, description="Lactation cycle number / parity.")
    current_lactation_status: str = Field(default="Lactating", description="Status: 'Lactating', 'Dry', 'Heifer', 'Transition'.")
    days_in_milk: Optional[float] = Field(default=None, description="Calculated days in current milk cycle.")
    lactation_stage: Optional[str] = Field(default=None, description="Calculated phase: 'Early', 'Mid', 'Late', 'Dry'.")
    daily_milk_yield_litres: float = Field(default=0.0, description="Current daily milk yield in litres/day.")
    milk_fat_percentage: float = Field(default=4.0, description="Average milk fat percentage (e.g., 4.0).")
    pregnancy_status: bool = Field(default=False, description="Pregnancy status.")
    pregnancy_month: Optional[int] = Field(default=None, description="Pregnancy month if pregnant.")

    is_demo: bool = Field(default=False, description="Flag indicating demo cattle.")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="before")
    @classmethod
    def sync_tag_and_animal_id(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Normalize whitespace and ensure tag_id and animal_id are in sync
            tid = data.get("tag_id") or data.get("animal_id") or data.get("tag_number")
            if tid:
                cleaned = str(tid).strip()
                data["tag_id"] = cleaned
                data["animal_id"] = cleaned
                if not data.get("tag_number"):
                    data["tag_number"] = cleaned
        return data


class CattleCreateRequest(BaseModel):
    """Payload to register a new cattle record with a globally unique Tag ID."""
    tag_id: str = Field(..., description="Globally unique Tag ID (e.g. 'COW-1001', 'TAG-402').", min_length=2, max_length=50)
    farm_id: Optional[str] = Field(default=None, description="Farm identifier. If omitted, assigned to user's default farm.")
    name: Optional[str] = Field(default=None, description="Optional name for the cow.")
    species: str = Field(default="Cattle", description="'Cattle' or 'Buffalo'")
    breed: str = Field(default="Crossbred", description="Breed name (e.g., 'Gir', 'Sahiwal', 'Murrah', 'Jersey', 'Holstein_Friesian')")
    gender: str = Field(default="Female", description="'Female' or 'Male'")
    age_months: Optional[float] = Field(default=36.0, ge=0.0, le=300.0, description="Age in months")
    date_of_birth: Optional[str] = Field(default=None, description="Date of birth (YYYY-MM-DD)")
    body_weight_kg: float = Field(default=420.0, ge=30.0, le=1500.0, description="Live body weight in kg")
    calving_date: Optional[str] = Field(default=None, description="Most recent calving date (YYYY-MM-DD)")
    parity: int = Field(default=1, ge=1, le=20, description="Lactation cycle number")
    current_lactation_status: str = Field(default="Lactating", description="'Lactating' or 'Dry'")
    daily_milk_yield_litres: float = Field(default=10.0, ge=0.0, le=100.0, description="Daily milk yield in litres")
    milk_fat_percentage: float = Field(default=4.0, ge=1.5, le=15.0, description="Milk fat percentage")
    pregnancy_status: bool = Field(default=False)
    pregnancy_month: Optional[int] = Field(default=None, ge=1, le=10)


class CattleUpdateRequest(BaseModel):
    """Payload to update an existing cattle record."""
    name: Optional[str] = None
    species: Optional[str] = None
    breed: Optional[str] = None
    age_months: Optional[float] = None
    body_weight_kg: Optional[float] = None
    calving_date: Optional[str] = None
    parity: Optional[int] = None
    current_lactation_status: Optional[str] = None
    daily_milk_yield_litres: Optional[float] = None
    milk_fat_percentage: Optional[float] = None
    pregnancy_status: Optional[bool] = None
    pregnancy_month: Optional[int] = None


class MilkRecord(BaseModel):
    """Persistent daily milk production record for a specific Tag ID."""
    record_id: str = Field(..., description="Unique milk record identifier.")
    tag_id: str = Field(..., description="Animal Tag ID associated with this milk production.")
    user_id: str = Field(..., description="Owner user ID.")
    farm_id: Optional[str] = Field(default=None, description="Farm identifier.")
    date: str = Field(..., description="Recording date in YYYY-MM-DD format.")
    morning_yield_litres: float = Field(default=0.0, ge=0.0, le=60.0, description="Morning milking yield in litres.")
    evening_yield_litres: float = Field(default=0.0, ge=0.0, le=60.0, description="Evening milking yield in litres.")
    total_yield_litres: float = Field(..., ge=0.0, le=120.0, description="Total daily milk yield in litres (Morning + Evening).")
    fat_percentage: Optional[float] = Field(default=None, ge=1.0, le=15.0, description="Fat % measurement if available.")
    snf_percentage: Optional[float] = Field(default=None, ge=4.0, le=15.0, description="SNF % measurement if available.")
    notes: Optional[str] = Field(default=None, description="Optional farmer observations.")
    is_demo: bool = Field(default=False, description="Flag indicating demo record.")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MilkRecordCreateRequest(BaseModel):
    """Input payload to record daily milk production for an animal."""
    date: Optional[str] = Field(default=None, description="Date in YYYY-MM-DD format. If omitted, defaults to today.")
    morning_yield_litres: float = Field(..., ge=0.0, le=60.0, description="Morning milking volume in litres.")
    evening_yield_litres: float = Field(..., ge=0.0, le=60.0, description="Evening milking volume in litres.")
    fat_percentage: Optional[float] = Field(default=None, ge=1.0, le=15.0, description="Optional fat percentage.")
    snf_percentage: Optional[float] = Field(default=None, ge=4.0, le=15.0, description="Optional SNF percentage.")
    notes: Optional[str] = Field(default=None, description="Optional notes (e.g., 'Normal milking', 'Slight reduction').")


class MilkHistoryResponse(BaseModel):
    """Complete milk history output for a specific cattle Tag ID."""
    tag_id: str = Field(..., description="Animal Tag ID.")
    total_records: int = Field(..., description="Total number of historical entries.")
    average_daily_yield_litres: float = Field(..., description="Average daily milk yield across recorded history.")
    latest_yield_litres: Optional[float] = Field(default=None, description="Most recent recorded total yield.")
    records: List[MilkRecord] = Field(default_factory=list, description="Chronological list of milk production entries.")


class VaccinationRecord(BaseModel):
    """Persistent record of an administered vaccine for an animal."""
    record_id: str = Field(..., description="Unique vaccination record identifier.")
    tag_id: str = Field(..., description="Animal Tag ID.")
    user_id: str = Field(..., description="Owner user ID.")
    disease_target: str = Field(..., description="Target disease (e.g. 'FMD', 'HS', 'BQ', 'Brucellosis', 'Anthrax', 'LSD').")
    vaccine_name: str = Field(..., description="Specific commercial or generic vaccine name.")
    administered_date: str = Field(..., description="Date vaccination was administered (YYYY-MM-DD).")
    next_due_date: str = Field(..., description="Calculated next booster/vaccination due date (YYYY-MM-DD).")
    recommended_timing: str = Field(..., description="Schedule description (e.g., 'Bi-annual booster every 6 months').")
    status: str = Field(default="COMPLETED", description="'COMPLETED', 'DUE', 'UPCOMING', 'OVERDUE'.")
    estimated_cost_inr: Optional[float] = Field(default=None, description="Configured estimated cost in INR.")
    batch_number: Optional[str] = Field(default=None, description="Vaccine vial batch/lot number.")
    veterinarian_name: Optional[str] = Field(default=None, description="Administering veterinarian or technician.")
    notes: Optional[str] = Field(default=None, description="Optional clinical notes.")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class VaccinePriceDetail(BaseModel):
    """Authoritative source-backed veterinary vaccine pricing detail."""
    disease_target: str = Field(..., description="Disease prevented (e.g., 'FMD', 'HS', 'BQ', 'Brucellosis', 'Anthrax', 'LSD').")
    vaccine_name: str = Field(..., description="Formulation / chemical description.")
    brand_name: Optional[str] = Field(default=None, description="Commercial brand name if available (e.g. 'Raksha-Ovac', 'Bruvax', 'Raksha-HS').")
    manufacturer: Optional[str] = Field(default=None, description="Manufacturer name (e.g. 'Indian Immunologicals Ltd', 'Hester Biosciences').")
    pack_size_doses: Optional[int] = Field(default=None, description="Standard pack vial volume in doses (e.g., 50 or 100 doses).")
    total_pack_price_inr: Optional[float] = Field(default=None, description="Total institutional pack price in INR.")
    calculated_per_dose_inr: Optional[float] = Field(default=None, description="Calculated single dose price (Pack Price / Pack Size).")
    cost_per_dose_display: str = Field(default="₹0", description="Formatted single dose cost string.")
    procurement_cost_inr: Optional[float] = Field(default=None, description="Government institutional procurement cost per dose in INR.")
    procurement_cost_display: Optional[str] = Field(default=None, description="Formatted Government Procurement Price string.")
    retail_price_inr: Optional[float] = Field(default=None, description="Separately verified private retail price in INR if available.")
    retail_price_display: str = Field(
        default="Retail price unavailable — check local veterinary pharmacy / Animal Husbandry Department.",
        description="Private Retail Price display string."
    )
    price_type: str = Field(
        default="GOVERNMENT_PROGRAMME_FREE",
        description="Type of price: 'GOVERNMENT_PROGRAMME_FREE', 'GOVERNMENT_PROCUREMENT', 'MANUFACTURER_LIST', 'RETAIL_MARKET', 'UNAVAILABLE'."
    )
    farmer_cost_inr: Optional[float] = Field(default=0.0, description="Net cost to eligible farmer under government programme in INR.")
    farmer_cost_display: str = Field(default="₹0 (Government Programme / Farmer Cost)", description="Clear farmer-facing cost representation.")
    state_market: Optional[str] = Field(default="All India", description="Geographical jurisdiction or market (e.g. 'All India (NADCP)', 'State Animal Husbandry').")
    source_name: Optional[str] = Field(default=None, description="Authoritative source name (e.g. 'Department of Animal Husbandry & Dairying (DAHD)').")
    source_url: Optional[str] = Field(default=None, description="Authoritative URL reference if available.")
    source_date: Optional[str] = Field(default=None, description="Date or year of price bulletin/contract (YYYY-MM-DD or YYYY-MM).")
    is_stale: bool = Field(default=False, description="Flag indicating if source price is older than review threshold.")
    notes: Optional[str] = Field(default=None, description="Logistical or storage notes (e.g., cold chain requirements).")
    eligibility_notes: Optional[str] = Field(default=None, description="Government programme eligibility criteria and conditions.")


class VaccinationRecommendation(BaseModel):
    """Dynamic veterinary vaccination recommendation for an animal."""
    tag_id: str = Field(..., description="Target animal Tag ID.")
    disease_target: str = Field(..., description="Disease prevented (e.g., 'FMD', 'HS', 'BQ', 'Brucellosis', 'Anthrax', 'LSD').")
    recommended_vaccine: str = Field(..., description="Recommended vaccine formulation.")
    recommended_timing: str = Field(..., description="Optimal administration schedule.")
    next_due_date: str = Field(..., description="Calculated next recommended due date (YYYY-MM-DD).")
    status: str = Field(..., description="'DUE', 'UPCOMING', 'COMPLETED', 'OVERDUE'.")
    estimated_cost_inr: Optional[float] = Field(default=None, description="Benchmark or procurement cost in INR.")
    estimated_cost_display: str = Field(..., description="Human-readable cost display (e.g. '₹0 (Government Programme / Farmer Cost)').")
    brand_name: Optional[str] = Field(default=None, description="Brand name if applicable.")
    manufacturer: Optional[str] = Field(default=None, description="Manufacturer name if applicable.")
    pack_size_doses: Optional[int] = Field(default=None, description="Pack size in doses.")
    total_pack_price_inr: Optional[float] = Field(default=None, description="Pack price in INR.")
    calculated_per_dose_inr: Optional[float] = Field(default=None, description="Per-dose cost in INR.")
    procurement_cost_inr: Optional[float] = Field(default=None, description="Procurement price per dose.")
    procurement_cost_display: Optional[str] = Field(default=None, description="Government Procurement Price display.")
    retail_price_inr: Optional[float] = Field(default=None, description="Private Retail Price.")
    retail_price_display: Optional[str] = Field(default=None, description="Private Retail Price display.")
    price_type: Optional[str] = Field(default=None, description="Price classification type.")
    farmer_cost_inr: Optional[float] = Field(default=None, description="Farmer cost in INR.")
    farmer_cost_display: Optional[str] = Field(default=None, description="Farmer cost representation.")
    state_market: Optional[str] = Field(default=None, description="State or market.")
    source_name: Optional[str] = Field(default=None, description="Source citation.")
    source_url: Optional[str] = Field(default=None, description="Source URL reference.")
    source_date: Optional[str] = Field(default=None, description="Source publication/contract date.")
    is_stale: bool = Field(default=False, description="Staleness status.")
    eligibility_notes: Optional[str] = Field(default=None, description="Government programme eligibility details.")
    price_detail: Optional[VaccinePriceDetail] = Field(default=None, description="Full structured price detail object.")
    last_administered_date: Optional[str] = Field(default=None, description="Date of last administration if recorded.")
    notes: Optional[str] = Field(default=None, description="Practical administration notes.")
    disclaimer: str = Field(
        default="Estimated information only. Consult a qualified veterinarian for diagnosis and vaccination decisions.",
        description="Mandatory veterinary disclaimer."
    )


class VaccinationRecordCreateRequest(BaseModel):
    """Input payload to record an administered vaccination."""
    disease_target: str = Field(..., description="Target disease (e.g., 'FMD', 'HS', 'BQ', 'Brucellosis', 'Anthrax', 'LSD').")
    vaccine_name: str = Field(..., description="Vaccine name administered.")
    administered_date: Optional[str] = Field(default=None, description="Date administered (YYYY-MM-DD). Defaults to today.")
    next_due_date: Optional[str] = Field(default=None, description="Optional override for next due date (YYYY-MM-DD).")
    estimated_cost_inr: Optional[float] = Field(default=None, description="Actual or estimated cost in INR.")
    batch_number: Optional[str] = Field(default=None, description="Vaccine batch number.")
    veterinarian_name: Optional[str] = Field(default=None, description="Veterinarian or VAW name.")
    notes: Optional[str] = Field(default=None, description="Optional notes.")


class LactationStatusResponse(BaseModel):
    """Calculated lactation status and DIM metrics for an animal."""
    tag_id: str = Field(..., description="Animal Tag ID.")
    calving_date: Optional[str] = Field(default=None, description="Recorded calving date (YYYY-MM-DD).")
    lactation_start_date: Optional[str] = Field(default=None, description="Lactation start date (YYYY-MM-DD).")
    parity: int = Field(default=1, description="Lactation number / parity.")
    days_in_milk: Optional[int] = Field(default=None, description="Calculated Days in Milk (DIM).")
    lactation_stage: str = Field(..., description="Stage: 'Early', 'Mid', 'Late', 'Dry'.")
    current_status: str = Field(..., description="Status: 'Lactating', 'Dry', 'Transition'.")
    stage_description: str = Field(..., description="Clinical description of current lactation phase.")
    estimated_peak_yield_day: Optional[int] = Field(default=45, description="Expected peak yield timeline in days post-calving.")
    suggested_dry_off_date: Optional[str] = Field(default=None, description="Recommended dry-off date (approx. 305 days post-calving).")


class CalvingEventRequest(BaseModel):
    """Input payload to record a new calving event for an animal."""
    calving_date: str = Field(..., description="Date of calving event (YYYY-MM-DD).")
    parity: Optional[int] = Field(default=None, description="New lactation parity number. If omitted, increments previous parity.")
    calf_gender: Optional[str] = Field(default=None, description="Optional newborn calf gender ('Female' or 'Male').")
    notes: Optional[str] = Field(default=None, description="Optional delivery notes.")


class AnalysisRecord(BaseModel):
    """Persistent analysis record (feed, silage, disease, breed, NIR, milk yield)."""
    record_id: str = Field(..., description="Unique analysis record identifier.")
    user_id: str = Field(..., description="Owner user ID.")
    farm_id: Optional[str] = Field(default=None, description="Associated farm ID.")
    animal_id: Optional[str] = Field(default=None, description="Associated animal ID / Tag ID.")
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
    validated_cattle: Optional[Cattle] = Field(default=None, description="Validated Cattle entity if animal_id/tag_id supplied.")
