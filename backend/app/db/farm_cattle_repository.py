"""
Farm, Cattle, and Analysis History Repository
Provides persistent storage and strict tenant isolation for Users, Farms, Cattle, and Analysis Records.
Adheres to strict relational hierarchy: User -> Farm -> Cattle -> AnalysisRecord.
Supports dual In-Memory and Supabase PostgreSQL backends.
"""

import abc
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import httpx

from backend.config import settings
from backend.app.schemas.user_farm_cattle import UserProfile, Farm, Cattle, AnalysisRecord

logger = logging.getLogger("dairy_ai.db.farm_cattle_repository")

DEMO_USER_ID = "user_demo"
DEMO_FARM_ID = "farm_demo_01"
DEMO_ANIMAL_ID = "COW_DEMO_01"


class FarmCattleRepository(abc.ABC):
    """Abstract Repository Interface for Farms, Cattle, and Analysis Records."""

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
    def get_cattle(self, animal_id: str, user_id: str, farm_id: Optional[str] = None) -> Optional[Cattle]:
        """Fetch cattle owned by user_id and belonging to farm_id."""
        pass

    @abc.abstractmethod
    def list_cattle(self, user_id: str, farm_id: Optional[str] = None) -> List[Cattle]:
        """List all cattle owned by user_id."""
        pass

    @abc.abstractmethod
    def save_cattle(self, cattle: Cattle) -> Cattle:
        """Create or update cattle record."""
        pass

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
    """Thread-safe In-Memory Farm, Cattle, and Analysis repository with isolated demo data."""

    def __init__(self):
        self._lock = threading.RLock()
        self._users: Dict[str, UserProfile] = {}
        self._farms: Dict[str, Farm] = {}
        self._cattle: Dict[str, Cattle] = {}
        self._records: List[AnalysisRecord] = []
        self._seed_demo_data()

    def _seed_demo_data(self):
        """Seed isolated demo environment data under DEMO_USER_ID."""
        now = datetime.now(timezone.utc)
        demo_user = UserProfile(user_id=DEMO_USER_ID, email="demo@dairynova.ai", full_name="Demo Farmer", is_demo=True, created_at=now)
        demo_farm = Farm(farm_id=DEMO_FARM_ID, user_id=DEMO_USER_ID, farm_name="DairyNova Demo Farm", location="Anand, Gujarat", is_demo=True, created_at=now)
        demo_cow = Cattle(
            animal_id=DEMO_ANIMAL_ID,
            farm_id=DEMO_FARM_ID,
            user_id=DEMO_USER_ID,
            tag_number="TAG_DEMO_01",
            species="Cattle",
            breed="Gir",
            age_months=48.0,
            body_weight_kg=420.0,
            lactation_stage="Mid",
            days_in_milk=90.0,
            daily_milk_yield_litres=14.5,
            milk_fat_percentage=4.2,
            is_demo=True,
            created_at=now
        )
        self._users[DEMO_USER_ID] = demo_user
        self._farms[DEMO_FARM_ID] = demo_farm
        self._cattle[DEMO_ANIMAL_ID] = demo_cow

    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        with self._lock:
            p = self._users.get(user_id)
            return p.model_copy(deep=True) if p else None

    def save_user_profile(self, profile: UserProfile) -> UserProfile:
        with self._lock:
            self._users[profile.user_id] = profile.model_copy(deep=True)
            return self._users[profile.user_id]

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

    def get_cattle(self, animal_id: str, user_id: str, farm_id: Optional[str] = None) -> Optional[Cattle]:
        with self._lock:
            cow = self._cattle.get(animal_id)
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
        with self._lock:
            self._cattle[cattle.animal_id] = cattle.model_copy(deep=True)
            return self._cattle[cattle.animal_id]

    def save_analysis_record(self, record: AnalysisRecord) -> AnalysisRecord:
        with self._lock:
            copy_rec = record.model_copy(deep=True)
            self._records.append(copy_rec)
            return copy_rec

    def get_latest_analysis(
        self,
        user_id: str,
        animal_id: Optional[str] = None,
        analysis_type: Optional[str] = None,
        limit: int = 5
    ) -> List[AnalysisRecord]:
        with self._lock:
            filtered = []
            for r in reversed(self._records):
                if r.user_id == user_id:
                    if animal_id and r.animal_id != animal_id:
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
