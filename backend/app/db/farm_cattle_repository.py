"""
Farm, Cattle, Milk History, Vaccination, and Analysis Repository
Provides persistent storage, global Tag ID uniqueness, and strict tenant isolation
for Users, Farms, Cattle, Milk Production History, and Health Records.
Adheres to strict relational hierarchy: User -> Farm -> Cattle -> (MilkRecords, VaccinationRecords, AnalysisRecords).
Supports In-Memory and Supabase PostgreSQL backends.
"""

import abc
import logging
import threading
import uuid
from datetime import datetime, timezone, date
from typing import Any, Dict, List, Optional

from backend.config import settings
from backend.app.schemas.user_farm_cattle import (
    UserProfile,
    Farm,
    Cattle,
    MilkRecord,
    MilkHistoryResponse,
    VaccinationRecord,
    VaccinationRecommendation,
    AnalysisRecord
)
from backend.app.core.exceptions import AppBaseException

logger = logging.getLogger("dairy_ai.db.farm_cattle_repository")

DEMO_USER_ID = "user_demo"
DEMO_FARM_ID = "farm_demo_01"
DEMO_TAG_ID = "COW_DEMO_01"
DEMO_ANIMAL_ID = DEMO_TAG_ID


def normalize_tag_id(tag_id: str) -> str:
    """
    Consistently normalizes cattle Tag IDs:
    - Trims leading and trailing whitespace
    - Uppercases tag characters for uniform lookup
    """
    if not tag_id:
        return ""
    return str(tag_id).strip().upper()


class DuplicateTagIdError(AppBaseException):
    """Raised when an attempt is made to register a Tag ID that already exists in the database."""

    def __init__(self, tag_id: str):
        self.tag_id = tag_id
        super().__init__(
            message="Tag ID already exists. Please use a unique Tag ID.",
            status_code=409,
            details={"tag_id": tag_id, "error_type": "DUPLICATE_TAG_ID"}
        )


class FarmCattleRepository(abc.ABC):
    """Abstract Repository Interface for Farms, Cattle, Milk History, and Vaccinations."""

    @abc.abstractmethod
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Fetch user profile."""
        pass

    @abc.abstractmethod
    def save_user_profile(self, profile: UserProfile) -> UserProfile:
        """Create or update user profile."""
        pass

    @abc.abstractmethod
    def get_farm(self, farm_id: str, user_id: str) -> Optional[Farm]:
        """Fetch farm owned by user_id."""
        pass

    @abc.abstractmethod
    def list_farms(self, user_id: str) -> List[Farm]:
        """List all farms owned by user_id."""
        pass

    @abc.abstractmethod
    def save_farm(self, farm: Farm) -> Farm:
        """Create or update a farm."""
        pass

    @abc.abstractmethod
    def get_cattle(self, animal_id_or_tag: str, user_id: str, farm_id: Optional[str] = None) -> Optional[Cattle]:
        """Fetch cattle owned by user_id by Tag ID / animal ID."""
        pass

    @abc.abstractmethod
    def list_cattle(self, user_id: str, farm_id: Optional[str] = None) -> List[Cattle]:
        """List all cattle owned by user_id."""
        pass

    @abc.abstractmethod
    def save_cattle(self, cattle: Cattle) -> Cattle:
        """Create or update cattle record with global Tag ID uniqueness enforcement."""
        pass

    @abc.abstractmethod
    def delete_cattle(self, tag_id: str, user_id: str) -> bool:
        """Delete cattle record owned by user."""
        pass

    # Milk Production History
    @abc.abstractmethod
    def save_milk_record(self, record: MilkRecord) -> MilkRecord:
        """Save and automatically sync daily milk record into persistent Milk History."""
        pass

    @abc.abstractmethod
    def list_milk_records(self, tag_id: str, user_id: str, limit: int = 100) -> List[MilkRecord]:
        """List persistent milk production history for cattle Tag ID."""
        pass

    @abc.abstractmethod
    def get_milk_summary(self, tag_id: str, user_id: str) -> MilkHistoryResponse:
        """Get summary analytics and chronological history for cattle Tag ID."""
        pass

    # Vaccination Records
    @abc.abstractmethod
    def save_vaccination_record(self, record: VaccinationRecord) -> VaccinationRecord:
        """Save persistent administered vaccination record."""
        pass

    @abc.abstractmethod
    def list_vaccination_records(self, tag_id: str, user_id: str) -> List[VaccinationRecord]:
        """List all vaccination records for cattle Tag ID."""
        pass

    # Analysis History
    @abc.abstractmethod
    def save_analysis_record(self, record: AnalysisRecord) -> AnalysisRecord:
        """Save analysis result (feed, silage, disease, NIR, yield)."""
        pass

    @abc.abstractmethod
    def get_latest_analysis(
        self,
        user_id: str,
        animal_id: Optional[str] = None,
        analysis_type: Optional[str] = None,
        limit: int = 5
    ) -> List[AnalysisRecord]:
        """Fetch latest authorized analysis records for a user and optional animal_id."""
        pass


class InMemoryFarmCattleRepository(FarmCattleRepository):
    """Thread-safe In-Memory Farm, Cattle, Milk History, and Vaccination repository."""

    def __init__(self):
        self._lock = threading.RLock()
        self._users: Dict[str, UserProfile] = {}
        self._farms: Dict[str, Farm] = {}
        self._cattle: Dict[str, Cattle] = {}  # Key: normalized tag_id
        # Global uniqueness map: normalized_tag_id -> (user_id, animal_id)
        self._global_tag_registry: Dict[str, Dict[str, str]] = {}
        self._milk_records: List[MilkRecord] = []
        self._vaccination_records: List[VaccinationRecord] = []
        self._analysis_records: List[AnalysisRecord] = []
        self._seed_demo_data()

    def _seed_demo_data(self):
        """Seed isolated demo environment data under DEMO_USER_ID."""
        now = datetime.now(timezone.utc)
        demo_user = UserProfile(user_id=DEMO_USER_ID, email="demo@dairynova.ai", full_name="Demo Farmer", is_demo=True, created_at=now)
        demo_farm = Farm(farm_id=DEMO_FARM_ID, user_id=DEMO_USER_ID, farm_name="DairyNova Demo Farm", location="Anand, Gujarat", is_demo=True, created_at=now)

        normalized_demo_tag = normalize_tag_id(DEMO_TAG_ID)
        demo_cow = Cattle(
            animal_id=normalized_demo_tag,
            tag_id=normalized_demo_tag,
            farm_id=DEMO_FARM_ID,
            user_id=DEMO_USER_ID,
            tag_number=normalized_demo_tag,
            name="Lakshmi",
            species="Cattle",
            breed="Gir",
            gender="Female",
            age_months=48.0,
            body_weight_kg=420.0,
            calving_date="2026-06-01",
            lactation_start_date="2026-06-01",
            parity=2,
            current_lactation_status="Lactating",
            days_in_milk=90.0,
            lactation_stage="Mid",
            daily_milk_yield_litres=15.0,
            milk_fat_percentage=4.2,
            is_demo=True,
            created_at=now,
            updated_at=now
        )
        self._users[DEMO_USER_ID] = demo_user
        self._farms[DEMO_FARM_ID] = demo_farm
        self._cattle[normalized_demo_tag] = demo_cow
        self._global_tag_registry[normalized_demo_tag] = {
            "user_id": DEMO_USER_ID,
            "animal_id": normalized_demo_tag
        }

        # Seed initial persistent milk records for demo cow
        self._milk_records.extend([
            MilkRecord(
                record_id="milk_demo_01",
                tag_id=normalized_demo_tag,
                user_id=DEMO_USER_ID,
                farm_id=DEMO_FARM_ID,
                date="2026-08-29",
                morning_yield_litres=7.5,
                evening_yield_litres=7.0,
                total_yield_litres=14.5,
                fat_percentage=4.2,
                snf_percentage=8.5,
                is_demo=True,
                created_at=now
            ),
            MilkRecord(
                record_id="milk_demo_02",
                tag_id=normalized_demo_tag,
                user_id=DEMO_USER_ID,
                farm_id=DEMO_FARM_ID,
                date="2026-08-30",
                morning_yield_litres=8.0,
                evening_yield_litres=7.0,
                total_yield_litres=15.0,
                fat_percentage=4.2,
                snf_percentage=8.6,
                is_demo=True,
                created_at=now
            ),
        ])

        # Seed demo vaccination
        self._vaccination_records.append(
            VaccinationRecord(
                record_id="vac_demo_01",
                tag_id=normalized_demo_tag,
                user_id=DEMO_USER_ID,
                disease_target="FMD",
                vaccine_name="Inactivated Trivalent FMD Vaccine",
                administered_date="2026-03-01",
                next_due_date="2026-09-01",
                recommended_timing="Bi-annual booster every 6 months",
                status="DUE",
                estimated_cost_inr=80.0,
                created_at=now
            )
        )

    # User Profiles
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        with self._lock:
            p = self._users.get(user_id)
            return p.model_copy(deep=True) if p else None

    def save_user_profile(self, profile: UserProfile) -> UserProfile:
        with self._lock:
            self._users[profile.user_id] = profile.model_copy(deep=True)
            return self._users[profile.user_id]

    # Farms
    def get_farm(self, farm_id: str, user_id: str) -> Optional[Farm]:
        with self._lock:
            farm = self._farms.get(farm_id)
            if farm and farm.user_id == user_id:
                return farm.model_copy(deep=True)
            return None

    def list_farms(self, user_id: str) -> List[Farm]:
        with self._lock:
            return [f.model_copy(deep=True) for f in self._farms.values() if f.user_id == user_id]

    def save_farm(self, farm: Farm) -> Farm:
        with self._lock:
            self._farms[farm.farm_id] = farm.model_copy(deep=True)
            return self._farms[farm.farm_id]

    def get_cattle(
        self,
        animal_id_or_tag: str = "",
        user_id: str = "",
        farm_id: Optional[str] = None,
        animal_id: Optional[str] = None
    ) -> Optional[Cattle]:
        target = animal_id_or_tag or animal_id or ""
        if not target:
            return None
        norm_tag = normalize_tag_id(target)
        with self._lock:
            cow = self._cattle.get(norm_tag)
            if cow and cow.user_id == user_id:
                if farm_id and cow.farm_id != farm_id:
                    return None
                return cow.model_copy(deep=True)
            return None

    def list_cattle(self, user_id: str, farm_id: Optional[str] = None) -> List[Cattle]:
        with self._lock:
            res = []
            for c in self._cattle.values():
                if c.user_id == user_id:
                    if farm_id is None or c.farm_id == farm_id:
                        res.append(c.model_copy(deep=True))
            return res

    def save_cattle(self, cattle: Cattle) -> Cattle:
        """
        Saves or updates a cattle record.
        Enforces GLOBAL uniqueness of Tag ID across all users and farms.
        """
        norm_tag = normalize_tag_id(cattle.tag_id or cattle.animal_id)
        if not norm_tag:
            raise ValueError("Tag ID cannot be empty or blank.")

        with self._lock:
            # Check global uniqueness across the entire application database
            existing_entry = self._global_tag_registry.get(norm_tag)
            if existing_entry:
                # If Tag ID is registered to a different user, reject
                if existing_entry["user_id"] != cattle.user_id:
                    logger.warning(
                        f"Global Tag ID conflict: User '{cattle.user_id}' attempted to register Tag ID '{norm_tag}' already owned by user '{existing_entry['user_id']}'."
                    )
                    raise DuplicateTagIdError(norm_tag)
                # If Tag ID is registered to a different animal under same user, reject
                if existing_entry["animal_id"] != norm_tag and existing_entry["animal_id"] != cattle.animal_id:
                    raise DuplicateTagIdError(norm_tag)

            # Update normalized fields
            cattle.tag_id = norm_tag
            cattle.animal_id = norm_tag
            cattle.tag_number = norm_tag
            cattle.updated_at = datetime.now(timezone.utc)

            # Auto-calculate DIM and lactation stage if calving date exists
            if cattle.calving_date:
                try:
                    calving_dt = datetime.strptime(cattle.calving_date, "%Y-%m-%d").date()
                    delta_days = (date.today() - calving_dt).days
                    dim = max(0, delta_days)
                    cattle.days_in_milk = float(dim)
                    if cattle.current_lactation_status.lower() == "dry" or dim > 305:
                        cattle.lactation_stage = "Dry"
                        cattle.current_lactation_status = "Dry"
                    elif dim <= 100:
                        cattle.lactation_stage = "Early"
                        cattle.current_lactation_status = "Lactating"
                    elif dim <= 200:
                        cattle.lactation_stage = "Mid"
                        cattle.current_lactation_status = "Lactating"
                    else:
                        cattle.lactation_stage = "Late"
                        cattle.current_lactation_status = "Lactating"
                except ValueError:
                    pass

            copy_cow = cattle.model_copy(deep=True)
            self._cattle[norm_tag] = copy_cow
            self._global_tag_registry[norm_tag] = {
                "user_id": cattle.user_id,
                "animal_id": norm_tag
            }
            logger.info(f"Persistently saved cattle Tag ID '{norm_tag}' for user '{cattle.user_id}'.")
            return copy_cow

    def delete_cattle(self, tag_id: str, user_id: str) -> bool:
        norm_tag = normalize_tag_id(tag_id)
        with self._lock:
            cow = self._cattle.get(norm_tag)
            if cow and cow.user_id == user_id:
                del self._cattle[norm_tag]
                self._global_tag_registry.pop(norm_tag, None)
                # Cleanup associated records
                self._milk_records = [r for r in self._milk_records if r.tag_id != norm_tag or r.user_id != user_id]
                self._vaccination_records = [r for r in self._vaccination_records if r.tag_id != norm_tag or r.user_id != user_id]
                return True
            return False

    # Milk Production History with Automatic Real-Time Persistence
    def save_milk_record(self, record: MilkRecord) -> MilkRecord:
        """
        Saves daily milk entry and automatically syncs it into persistent Milk History.
        Updates cattle daily yield metric.
        """
        norm_tag = normalize_tag_id(record.tag_id)
        record.tag_id = norm_tag
        if not record.record_id:
            record.record_id = f"milk_{uuid.uuid4().hex[:12]}"

        # Calculate total yield automatically if needed
        calculated_total = round(record.morning_yield_litres + record.evening_yield_litres, 2)
        record.total_yield_litres = calculated_total

        with self._lock:
            copy_rec = record.model_copy(deep=True)
            self._milk_records.append(copy_rec)

            # Update cattle latest daily yield if cow exists
            cow = self._cattle.get(norm_tag)
            if cow and cow.user_id == record.user_id:
                cow.daily_milk_yield_litres = calculated_total
                if record.fat_percentage:
                    cow.milk_fat_percentage = record.fat_percentage
                cow.updated_at = datetime.now(timezone.utc)

            logger.info(f"Automatically persisted milk record '{copy_rec.record_id}' ({calculated_total} L) for Tag ID '{norm_tag}'.")
            return copy_rec

    def list_milk_records(self, tag_id: str, user_id: str, limit: int = 100) -> List[MilkRecord]:
        norm_tag = normalize_tag_id(tag_id)
        with self._lock:
            records = [
                r.model_copy(deep=True) for r in self._milk_records
                if r.tag_id == norm_tag and r.user_id == user_id
            ]
            # Return chronologically sorted (or most recent first depending on application)
            records.sort(key=lambda x: x.date, reverse=True)
            return records[:limit]

    def get_milk_summary(self, tag_id: str, user_id: str) -> MilkHistoryResponse:
        records = self.list_milk_records(tag_id, user_id, limit=365)
        total_recs = len(records)
        if total_recs > 0:
            avg_yield = round(sum(r.total_yield_litres for r in records) / total_recs, 2)
            latest_yield = records[0].total_yield_litres
        else:
            avg_yield = 0.0
            latest_yield = None

        return MilkHistoryResponse(
            tag_id=normalize_tag_id(tag_id),
            total_records=total_recs,
            average_daily_yield_litres=avg_yield,
            latest_yield_litres=latest_yield,
            records=records
        )

    # Vaccination Records
    def save_vaccination_record(self, record: VaccinationRecord) -> VaccinationRecord:
        norm_tag = normalize_tag_id(record.tag_id)
        record.tag_id = norm_tag
        if not record.record_id:
            record.record_id = f"vac_{uuid.uuid4().hex[:12]}"
        with self._lock:
            copy_rec = record.model_copy(deep=True)
            self._vaccination_records.append(copy_rec)
            logger.info(f"Persisted vaccination record for Tag ID '{norm_tag}' ({record.disease_target}).")
            return copy_rec

    def list_vaccination_records(self, tag_id: str, user_id: str) -> List[VaccinationRecord]:
        norm_tag = normalize_tag_id(tag_id)
        with self._lock:
            records = [
                r.model_copy(deep=True) for r in self._vaccination_records
                if r.tag_id == norm_tag and r.user_id == user_id
            ]
            records.sort(key=lambda x: x.administered_date, reverse=True)
            return records

    # Analysis History
    def save_analysis_record(self, record: AnalysisRecord) -> AnalysisRecord:
        if record.animal_id:
            record.animal_id = normalize_tag_id(record.animal_id)
        with self._lock:
            copy_rec = record.model_copy(deep=True)
            self._analysis_records.append(copy_rec)
            return copy_rec

    def get_latest_analysis(
        self,
        user_id: str,
        animal_id: Optional[str] = None,
        analysis_type: Optional[str] = None,
        limit: int = 5
    ) -> List[AnalysisRecord]:
        norm_animal = normalize_tag_id(animal_id) if animal_id else None
        with self._lock:
            filtered = []
            for r in reversed(self._analysis_records):
                if r.user_id == user_id:
                    if norm_animal and r.animal_id != norm_animal:
                        continue
                    if analysis_type and r.analysis_type != analysis_type:
                        continue
                    filtered.append(r.model_copy(deep=True))
                    if len(filtered) >= limit:
                        break
            return filtered


# Singleton Repository Instance
_farm_cattle_repo: Optional[FarmCattleRepository] = None
_fc_lock = threading.Lock()


def get_farm_cattle_repository() -> FarmCattleRepository:
    global _farm_cattle_repo
    if _farm_cattle_repo is None:
        with _fc_lock:
            if _farm_cattle_repo is None:
                _farm_cattle_repo = InMemoryFarmCattleRepository()
    return _farm_cattle_repo


def set_farm_cattle_repository(repo: FarmCattleRepository) -> None:
    global _farm_cattle_repo
    with _fc_lock:
        _farm_cattle_repo = repo
