"""
Lactation Tracking & Timeline Calculation Service
Calculates Days in Milk (DIM), lactation stages, dry period transitions,
and nutritional timelines based on calving events.
"""

import logging
from datetime import datetime, date, timedelta, timezone
from typing import Dict, Optional, Tuple, Any

from backend.app.schemas.user_farm_cattle import Cattle, LactationStatusResponse

logger = logging.getLogger("dairy_ai.services.lactation")

# Configurable Lactation Stage Thresholds (in Days in Milk)
STAGE_THRESHOLDS: Dict[str, Tuple[int, int]] = {
    "Early": (1, 100),
    "Mid": (101, 200),
    "Late": (201, 305),
}

DRY_PERIOD_DAYS_THRESHOLD = 305


class LactationService:
    """Production Lactation Timeline & Phase Tracking Service."""

    def calculate_lactation_status(
        self,
        cattle: Cattle,
        reference_date: Optional[date] = None
    ) -> LactationStatusResponse:
        """
        Calculates DIM and determines lactation stage from recorded calving date.
        Recalculates timeline dynamically whenever queried.
        """
        ref_dt = reference_date or date.today()
        calving_str = cattle.calving_date or cattle.lactation_start_date

        if not calving_str:
            # Fallback if no calving date is recorded yet
            status = cattle.current_lactation_status or "Lactating"
            stage = cattle.lactation_stage or ("Dry" if status == "Dry" else "Mid")
            dim = int(cattle.days_in_milk or 90) if status != "Dry" else None
            return LactationStatusResponse(
                tag_id=cattle.tag_id,
                calving_date=None,
                lactation_start_date=None,
                parity=cattle.parity or 1,
                days_in_milk=dim,
                lactation_stage=stage,
                current_status=status,
                stage_description=self._get_stage_description(stage),
                estimated_peak_yield_day=45,
                suggested_dry_off_date=None
            )

        try:
            calving_dt = datetime.strptime(calving_str, "%Y-%m-%d").date()
        except ValueError:
            calving_dt = ref_dt

        # Calculate Days in Milk
        delta_days = (ref_dt - calving_dt).days
        dim = max(0, delta_days)

        # Determine Stage & Status
        if cattle.current_lactation_status.lower() == "dry":
            stage = "Dry"
            status = "Dry"
        elif dim > DRY_PERIOD_DAYS_THRESHOLD:
            stage = "Dry"
            status = "Dry"
        elif dim <= STAGE_THRESHOLDS["Early"][1]:
            stage = "Early"
            status = "Lactating"
        elif dim <= STAGE_THRESHOLDS["Mid"][1]:
            stage = "Mid"
            status = "Lactating"
        else:
            stage = "Late"
            status = "Lactating"

        dry_off_dt = calving_dt + timedelta(days=305)

        return LactationStatusResponse(
            tag_id=cattle.tag_id,
            calving_date=calving_dt.isoformat(),
            lactation_start_date=calving_dt.isoformat(),
            parity=cattle.parity or 1,
            days_in_milk=float(dim),
            lactation_stage=stage,
            current_status=status,
            stage_description=self._get_stage_description(stage),
            estimated_peak_yield_day=45,
            suggested_dry_off_date=dry_off_dt.isoformat()
        )

    def recalculate_cattle_lactation(self, cattle: Cattle, new_calving_date: str, new_parity: Optional[int] = None) -> Cattle:
        """
        Updates cattle with a new calving event, restarting DIM from 0/1 and resetting stage to Early.
        """
        calving_dt = datetime.strptime(new_calving_date, "%Y-%m-%d").date()
        today_dt = date.today()
        dim = max(0, (today_dt - calving_dt).days)

        if dim <= STAGE_THRESHOLDS["Early"][1]:
            stage = "Early"
            status = "Lactating"
        elif dim <= STAGE_THRESHOLDS["Mid"][1]:
            stage = "Mid"
            status = "Lactating"
        elif dim <= STAGE_THRESHOLDS["Late"][1]:
            stage = "Late"
            status = "Lactating"
        else:
            stage = "Dry"
            status = "Dry"

        cattle.calving_date = new_calving_date
        cattle.lactation_start_date = new_calving_date
        if new_parity is not None:
            cattle.parity = new_parity
        else:
            cattle.parity = (cattle.parity or 1) + 1
        cattle.days_in_milk = float(dim)
        cattle.lactation_stage = stage
        cattle.current_lactation_status = status
        cattle.updated_at = datetime.now(timezone.utc)
        return cattle

    def _get_stage_description(self, stage: str) -> str:
        descriptions = {
            "Early": "Early Lactation (1-100 days post-calving): High metabolic demand, negative energy balance recovery, milk yield ascending to peak.",
            "Mid": "Mid Lactation (101-200 days post-calving): Peak yield plateau and stable production, dry matter intake at maximum.",
            "Late": "Late Lactation (201-305 days post-calving): Gradual yield decline, body condition score replenishment, preparing for dry period.",
            "Dry": "Dry Period (>305 days post-calving / Resting): Mammary involution and regeneration, fetal growth support prior to next calving."
        }
        return descriptions.get(stage, "Standard dairy lactation cycle phase.")


lactation_service = LactationService()
