"""
Cattle Management, Milk Production History, Vaccination, and Lactation Endpoints
Provides globally unique Tag ID registration, persistent daily milk recording,
dynamic lactation phase calculation, and veterinary vaccination scheduling.
"""

import logging
import uuid
from datetime import datetime, timezone, date, timedelta
from typing import List, Optional

from fastapi import APIRouter, Request, HTTPException, status, Query, Depends

from backend.app.schemas.user_farm_cattle import (
    Cattle,
    CattleCreateRequest,
    CattleUpdateRequest,
    MilkRecord,
    MilkRecordCreateRequest,
    MilkHistoryResponse,
    VaccinationRecord,
    VaccinationRecommendation,
    VaccinationRecordCreateRequest,
    LactationStatusResponse,
    CalvingEventRequest
)
from backend.app.db.farm_cattle_repository import (
    get_farm_cattle_repository,
    normalize_tag_id,
    DuplicateTagIdError,
    DEMO_USER_ID,
    DEMO_FARM_ID
)
from backend.app.services.vaccination_service import vaccination_service
from backend.app.services.lactation_service import lactation_service
from backend.app.core.ownership_guard import ownership_guard

logger = logging.getLogger("dairy_ai.api.cattle")

router = APIRouter(prefix="/cattle", tags=["Cattle, Milk History & Lactation Management"])


def _get_auth_user_id(request: Request) -> str:
    """Derives authenticated user ID or raises 401 Unauthorized."""
    user_id = ownership_guard.extract_authenticated_user_id(request)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please provide a valid Authorization Bearer token or session."
        )
    return user_id


@router.post(
    "",
    response_model=Cattle,
    status_code=status.HTTP_201_CREATED,
    summary="Register New Cattle with Globally Unique Tag ID"
)
async def register_cattle(request: Request, payload: CattleCreateRequest):
    """
    Registers a new cattle or buffalo with a globally unique Tag ID.
    Enforces uniqueness across the entire database.
    """
    user_id = _get_auth_user_id(request)
    repo = get_farm_cattle_repository()
    norm_tag = normalize_tag_id(payload.tag_id)

    farm_id = payload.farm_id or DEMO_FARM_ID
    # Ensure user default farm exists
    farm = repo.get_farm(farm_id, user_id)
    if not farm:
        repo.save_farm(
            repo.save_farm(
                type("FarmObj", (), {
                    "farm_id": farm_id,
                    "user_id": user_id,
                    "farm_name": "My Dairy Farm",
                    "location": None,
                    "is_demo": user_id == DEMO_USER_ID,
                    "created_at": datetime.now(timezone.utc),
                    "model_copy": lambda self, deep=True: self
                })()
            )
        )

    # Initial lactation calculation
    dim = None
    stage = None
    if payload.calving_date:
        try:
            calving_dt = datetime.strptime(payload.calving_date, "%Y-%m-%d").date()
            dim = float(max(0, (date.today() - calving_dt).days))
            if payload.current_lactation_status.lower() == "dry" or dim > 305:
                stage = "Dry"
            elif dim <= 100:
                stage = "Early"
            elif dim <= 200:
                stage = "Mid"
            else:
                stage = "Late"
        except ValueError:
            pass

    cow = Cattle(
        animal_id=norm_tag,
        tag_id=norm_tag,
        farm_id=farm_id,
        user_id=user_id,
        tag_number=norm_tag,
        name=payload.name,
        species=payload.species,
        breed=payload.breed,
        gender=payload.gender,
        age_months=payload.age_months,
        date_of_birth=payload.date_of_birth,
        body_weight_kg=payload.body_weight_kg,
        calving_date=payload.calving_date,
        lactation_start_date=payload.calving_date,
        parity=payload.parity,
        current_lactation_status=payload.current_lactation_status,
        days_in_milk=dim,
        lactation_stage=stage,
        daily_milk_yield_litres=payload.daily_milk_yield_litres,
        milk_fat_percentage=payload.milk_fat_percentage,
        pregnancy_status=payload.pregnancy_status,
        pregnancy_month=payload.pregnancy_month,
        is_demo=user_id == DEMO_USER_ID,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )

    try:
        saved = repo.save_cattle(cow)
        return saved
    except DuplicateTagIdError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Tag ID already exists. Please use a unique Tag ID."
        )


@router.get(
    "",
    response_model=List[Cattle],
    status_code=status.HTTP_200_OK,
    summary="List Authenticated User's Cattle"
)
async def list_cattle(request: Request, farm_id: Optional[str] = Query(None)):
    """Lists all cattle owned by the authenticated user."""
    user_id = _get_auth_user_id(request)
    repo = get_farm_cattle_repository()
    return repo.list_cattle(user_id=user_id, farm_id=farm_id)


@router.get(
    "/{tag_id}",
    response_model=Cattle,
    status_code=status.HTTP_200_OK,
    summary="Get Cattle Record by Tag ID"
)
async def get_cattle_by_tag(request: Request, tag_id: str):
    """Retrieves a single cattle record by Tag ID for the authenticated user."""
    user_id = _get_auth_user_id(request)
    repo = get_farm_cattle_repository()
    norm_tag = normalize_tag_id(tag_id)
    cow = repo.get_cattle(norm_tag, user_id=user_id)
    if not cow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cattle with Tag ID '{norm_tag}' not found or access denied."
        )
    return cow


@router.put(
    "/{tag_id}",
    response_model=Cattle,
    status_code=status.HTTP_200_OK,
    summary="Update Cattle Profile"
)
async def update_cattle(request: Request, tag_id: str, payload: CattleUpdateRequest):
    """Updates profile attributes of an existing cattle record."""
    user_id = _get_auth_user_id(request)
    repo = get_farm_cattle_repository()
    norm_tag = normalize_tag_id(tag_id)
    cow = repo.get_cattle(norm_tag, user_id=user_id)
    if not cow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cattle with Tag ID '{norm_tag}' not found or access denied."
        )

    # Apply updates
    update_dict = payload.model_dump(exclude_unset=True)
    for k, v in update_dict.items():
        if hasattr(cow, k) and v is not None:
            setattr(cow, k, v)

    # If calving date updated, recalculate lactation
    if "calving_date" in update_dict and update_dict["calving_date"]:
        cow = lactation_service.recalculate_cattle_lactation(cow, update_dict["calving_date"], update_dict.get("parity"))

    saved = repo.save_cattle(cow)
    return saved


@router.delete(
    "/{tag_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete Cattle Record"
)
async def delete_cattle(request: Request, tag_id: str):
    """Deletes a cattle record owned by the authenticated user."""
    user_id = _get_auth_user_id(request)
    repo = get_farm_cattle_repository()
    norm_tag = normalize_tag_id(tag_id)
    success = repo.delete_cattle(norm_tag, user_id=user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cattle with Tag ID '{norm_tag}' not found or access denied."
        )
    return {"success": True, "message": f"Cattle with Tag ID '{norm_tag}' deleted successfully."}


# ==============================================================================
# 2. Milk Production Recording & Persistent History Sync
# ==============================================================================

@router.post(
    "/{tag_id}/milk",
    response_model=MilkRecord,
    status_code=status.HTTP_201_CREATED,
    summary="Record Milk Production (Automatically Persists to Milk History)"
)
async def record_milk_production(request: Request, tag_id: str, payload: MilkRecordCreateRequest):
    """
    Records morning and evening milk production for an animal.
    The record AUTOMATICALLY syncs into persistent Milk History without any manual step.
    """
    user_id = _get_auth_user_id(request)
    repo = get_farm_cattle_repository()
    norm_tag = normalize_tag_id(tag_id)

    cow = repo.get_cattle(norm_tag, user_id=user_id)
    if not cow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cannot record milk: Cattle with Tag ID '{norm_tag}' not found in your herd."
        )

    rec_date = payload.date or date.today().isoformat()
    total_l = round(payload.morning_yield_litres + payload.evening_yield_litres, 2)

    milk_rec = MilkRecord(
        record_id=f"milk_{uuid.uuid4().hex[:12]}",
        tag_id=norm_tag,
        user_id=user_id,
        farm_id=cow.farm_id,
        date=rec_date,
        morning_yield_litres=payload.morning_yield_litres,
        evening_yield_litres=payload.evening_yield_litres,
        total_yield_litres=total_l,
        fat_percentage=payload.fat_percentage or cow.milk_fat_percentage,
        snf_percentage=payload.snf_percentage,
        notes=payload.notes,
        is_demo=user_id == DEMO_USER_ID,
        created_at=datetime.now(timezone.utc)
    )

    saved_rec = repo.save_milk_record(milk_rec)
    return saved_rec


@router.get(
    "/{tag_id}/milk-history",
    response_model=MilkHistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Persistent Milk Production History"
)
async def get_milk_history(request: Request, tag_id: str):
    """Retrieves full persistent milk production history for a cattle Tag ID."""
    user_id = _get_auth_user_id(request)
    repo = get_farm_cattle_repository()
    norm_tag = normalize_tag_id(tag_id)

    cow = repo.get_cattle(norm_tag, user_id=user_id)
    if not cow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cattle with Tag ID '{norm_tag}' not found or access denied."
        )

    return repo.get_milk_summary(norm_tag, user_id=user_id)


# ==============================================================================
# 3. Vaccination Recommendations & Records
# ==============================================================================

@router.get(
    "/{tag_id}/vaccinations",
    response_model=List[VaccinationRecommendation],
    status_code=status.HTTP_200_OK,
    summary="Get Vaccination Schedule & Recommendations for Cattle"
)
async def get_cattle_vaccinations(request: Request, tag_id: str):
    """
    Returns personalized veterinary vaccination recommendations,
    next due dates, statuses, and estimated costs linked to the cattle Tag ID.
    """
    user_id = _get_auth_user_id(request)
    repo = get_farm_cattle_repository()
    norm_tag = normalize_tag_id(tag_id)

    cow = repo.get_cattle(norm_tag, user_id=user_id)
    if not cow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cattle with Tag ID '{norm_tag}' not found or access denied."
        )

    administered = repo.list_vaccination_records(norm_tag, user_id=user_id)
    recommendations = vaccination_service.generate_recommendations(cow, administered)
    return recommendations


@router.post(
    "/{tag_id}/vaccinations",
    response_model=VaccinationRecord,
    status_code=status.HTTP_201_CREATED,
    summary="Record Administered Vaccine"
)
async def record_administered_vaccine(
    request: Request,
    tag_id: str,
    payload: VaccinationRecordCreateRequest
):
    """Records that a vaccine was administered to the animal."""
    user_id = _get_auth_user_id(request)
    repo = get_farm_cattle_repository()
    norm_tag = normalize_tag_id(tag_id)

    cow = repo.get_cattle(norm_tag, user_id=user_id)
    if not cow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cattle with Tag ID '{norm_tag}' not found or access denied."
        )

    admin_date = payload.administered_date or date.today().isoformat()
    sched = vaccination_service.get_vaccination_schedule_config().get(payload.disease_target.upper(), {})
    timing = sched.get("recommended_timing", "Standard booster protocol")
    est_cost = payload.estimated_cost_inr or sched.get("estimated_cost_inr", 60.0)

    if payload.next_due_date:
        next_due = payload.next_due_date
    else:
        interval = sched.get("interval_days", 180)
        if interval:
            try:
                dt_obj = datetime.strptime(admin_date, "%Y-%m-%d").date()
                next_due = (dt_obj + timedelta(days=interval)).isoformat()
            except ValueError:
                next_due = (date.today() + timedelta(days=interval)).isoformat()
        else:
            next_due = "LIFETIME_PROTECTED"

    record = VaccinationRecord(
        record_id=f"vac_{uuid.uuid4().hex[:12]}",
        tag_id=norm_tag,
        user_id=user_id,
        disease_target=payload.disease_target,
        vaccine_name=payload.vaccine_name,
        administered_date=admin_date,
        next_due_date=next_due,
        recommended_timing=timing,
        status="COMPLETED",
        estimated_cost_inr=est_cost,
        batch_number=payload.batch_number,
        veterinarian_name=payload.veterinarian_name,
        notes=payload.notes,
        created_at=datetime.now(timezone.utc)
    )

    saved = repo.save_vaccination_record(record)
    return saved


# ==============================================================================
# 4. Lactation Tracking & Calving Events
# ==============================================================================

@router.get(
    "/{tag_id}/lactation",
    response_model=LactationStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Lactation Timeline & Days in Milk (DIM)"
)
async def get_lactation_status(request: Request, tag_id: str):
    """Calculates Days in Milk (DIM) and determines current lactation stage."""
    user_id = _get_auth_user_id(request)
    repo = get_farm_cattle_repository()
    norm_tag = normalize_tag_id(tag_id)

    cow = repo.get_cattle(norm_tag, user_id=user_id)
    if not cow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cattle with Tag ID '{norm_tag}' not found or access denied."
        )

    return lactation_service.calculate_lactation_status(cow)


@router.post(
    "/{tag_id}/calving",
    response_model=LactationStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Record New Calving Event (Recalculates Lactation Cycle)"
)
async def record_calving_event(request: Request, tag_id: str, payload: CalvingEventRequest):
    """
    Records a new calving event for the cattle, resetting DIM and restarting Early Lactation.
    """
    user_id = _get_auth_user_id(request)
    repo = get_farm_cattle_repository()
    norm_tag = normalize_tag_id(tag_id)

    cow = repo.get_cattle(norm_tag, user_id=user_id)
    if not cow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cattle with Tag ID '{norm_tag}' not found or access denied."
        )

    updated_cow = lactation_service.recalculate_cattle_lactation(cow, payload.calving_date, payload.parity)
    repo.save_cattle(updated_cow)

    return lactation_service.calculate_lactation_status(updated_cow)
