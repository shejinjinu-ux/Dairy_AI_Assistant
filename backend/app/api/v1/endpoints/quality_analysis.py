"""
Combined Feed and Silage Quality Analysis Endpoints
Aggregates Reference Nutrition, ML Inference, Visual Mould Screening, and Risk Analysis into unified APIs.
"""
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile, Request, status, HTTPException

from backend.app.schemas.quality_analysis import (
    CombinedFeedAnalysisResponse,
    CombinedSilageAnalysisResponse
)
from backend.app.schemas.feed_reference import FeedReferenceRequest
from backend.app.schemas.feed_nutrition import FeedNutritionInput
from backend.app.schemas.silage import SilageInput
from backend.app.schemas.user_farm_cattle import AnalysisRecord
from backend.app.services.feed_reference_service import feed_reference_service
from backend.app.services.feed_nutrition_service import feed_nutrition_service
from backend.app.services.feed_scoring import calculate_feed_quality_score, g_per_kg_to_percentage
from backend.app.services.silage_service import silage_service
from backend.app.services.visual_mould_service import visual_mould_service
from backend.app.services.risk_assessment_service import risk_assessment_service
from backend.app.core.exceptions import AppBaseException
from backend.app.core.file_validator import validate_image_file
from backend.app.core.ownership_guard import ownership_guard
from backend.app.db.farm_cattle_repository import get_farm_cattle_repository

logger = logging.getLogger("dairy_ai.api.quality_analysis")

router = APIRouter(prefix="/analyze", tags=["Combined Quality Analysis & Screening"])


@router.post(
    "/feed",
    response_model=CombinedFeedAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Comprehensive Combined Feed Quality Analysis (Reference + ML + Vision + Risk)"
)
async def analyze_feed(
    request: Request,
    feed_name: str = Form(default="Maize", description="Feed ingredient name (e.g. 'Maize', 'Hybrid Napier', 'Wheat Bran')"),
    quantity_kg: Optional[float] = Form(default=1.0, description="Feed quantity in kg"),
    dry_matter_g_per_kg: Optional[float] = Form(default=None, description="Optional proximal dry matter in g/kg"),
    crude_fibre_g_per_kg_dm: Optional[float] = Form(default=None, description="Optional proximal crude fibre in g/kg DM"),
    ndf_g_per_kg_dm: Optional[float] = Form(default=None, description="Optional proximal NDF in g/kg DM"),
    adf_g_per_kg_dm: Optional[float] = Form(default=None, description="Optional proximal ADF in g/kg DM"),
    starch_g_per_kg_dm: Optional[float] = Form(default=None, description="Optional proximal starch in g/kg DM"),
    farm_id: Optional[str] = Form(default=None, description="Optional farm ID"),
    animal_id: Optional[str] = Form(default=None, description="Optional animal ID"),
    image: Optional[UploadFile] = File(default=None, description="Optional feed sample image for visual mould screening")
):
    """
    Consolidated Feed Analysis Endpoint:
    Combines reference nutrition lookup, proximal ML predictions, visual mould screening,
    dynamic quality scoring, and contamination hazard analysis.
    """
    try:
        # 0. Validate Ownership Context if farm_id/animal_id provided
        auth_ctx = ownership_guard.validate_request_ownership(
            request=request, farm_id=farm_id, animal_id=animal_id, require_auth=False
        )

        # 1. Reference Nutrition
        ref_req = FeedReferenceRequest(feed_name=feed_name, quantity_kg=quantity_kg or 1.0)
        ref_response = feed_reference_service.calculate_nutrition(ref_req)
        matched_cat = ref_response.category

        # 2. Optional ML Inference if proximal parameters supplied
        ml_response = None
        if dry_matter_g_per_kg is not None or ndf_g_per_kg_dm is not None:
            ml_input = FeedNutritionInput(
                feed_category="Forages" if "roughage" in matched_cat.lower() else "Concentrates",
                detailed_feed_category=ref_response.matched_feed_name,
                dry_matter_g_per_kg=dry_matter_g_per_kg or (ref_response.per_kg.dry_matter_g or 880.0),
                crude_fibre_g_per_kg_dm=crude_fibre_g_per_kg_dm or (ref_response.per_kg.crude_fibre_g or 100.0),
                ndf_g_per_kg_dm=ndf_g_per_kg_dm or (ref_response.per_kg.ndf_g or 300.0),
                adf_g_per_kg_dm=adf_g_per_kg_dm or (ref_response.per_kg.adf_g or 150.0),
                starch_g_per_kg_dm=starch_g_per_kg_dm or (ref_response.per_kg.starch_g or 200.0)
            )
            ml_response = feed_nutrition_service.predict_all(ml_input)

        # 3. Secure Visual Mould Screening if image supplied
        visual_response = None
        if image is not None and image.filename:
            img_bytes = await image.read()
            if len(img_bytes) > 0:
                # Magic byte security check
                validate_image_file(img_bytes, filename=image.filename)
                visual_response = visual_mould_service.predict_feed_visual(img_bytes)

        # 4. Calculate Dynamic Composite Score
        if ml_response and ml_response.quality_score is not None:
            base_score = ml_response.quality_score
            why_items = list(ml_response.why or [])
            action_items = list(ml_response.recommended_action or [])
        else:
            dm_g = ref_response.per_kg.dry_matter_g or 880.0
            cp_g = ref_response.per_kg.crude_protein_g or 100.0
            cf_g = ref_response.per_kg.crude_fibre_g or 100.0
            ndf_g = ref_response.per_kg.ndf_g or 300.0
            adf_g = ref_response.per_kg.adf_g or 150.0
            adl_g = ref_response.per_kg.adl_g
            starch_g = ref_response.per_kg.starch_g

            base_score, _, why_items, action_items = calculate_feed_quality_score(
                feed_category=matched_cat,
                dry_matter_g_per_kg=dm_g,
                crude_protein_g_per_kg_dm=cp_g,
                crude_fibre_g_per_kg_dm=cf_g,
                ndf_g_per_kg_dm=ndf_g,
                adf_g_per_kg_dm=adf_g,
                adl_g_per_kg_dm=adl_g,
                starch_g_per_kg_dm=starch_g
            )

        # Apply Visual Mould Penalty if image screening performed and was valid
        if visual_response is not None and visual_response.success:
            if visual_response.predicted_class == "SPOILED":
                base_score = max(10.0, base_score - 45.0)
                why_items.insert(0, "CRITICAL: Visual screening detected severe surface spoilage and decomposition.")
                action_items.insert(0, "DO NOT feed spoiled feed to dairy animals.")
            elif visual_response.predicted_class == "MOULD_RISK":
                base_score = max(20.0, base_score - 25.0)
                why_items.insert(0, "WARNING: Visual screening detected fungal cluster spots and mould risk.")
                action_items.insert(0, "Isolate batch and avoid feeding mouldy portions.")
            else:
                why_items.append("Visual screening confirmed clean surface with no visible fungal patches.")
        elif visual_response is not None and not visual_response.success:
            why_items.append(f"Visual screening note: {visual_response.message or 'Image not recognized as cattle feed; visual quality penalty skipped.'}")

        final_score = round(max(0.0, min(100.0, base_score)), 1)
        if final_score >= 85.0:
            status_tier = "EXCELLENT"
        elif final_score >= 70.0:
            status_tier = "GOOD"
        elif final_score >= 50.0:
            status_tier = "FAIR"
        else:
            status_tier = "POOR"

        # 5. Risk Assessment
        dm_pct = ref_response.nutrient_percentages_dm.get("dry_matter_percent", 88.0)
        risk_obj = risk_assessment_service.assess_feed_risk(
            visual_screening=visual_response,
            moisture_pct=(100.0 - dm_pct) if dm_pct else 12.0,
            feed_category=matched_cat
        )

        # 6. Save Persistent Analysis History if User Authenticated
        rec_id = None
        now_iso = None
        if auth_ctx.is_authenticated and auth_ctx.user_id:
            rec_id = f"rec_{uuid.uuid4().hex[:12]}"
            now_iso = datetime.now(timezone.utc).isoformat()
            history_rec = AnalysisRecord(
                record_id=rec_id,
                user_id=auth_ctx.user_id,
                farm_id=farm_id,
                animal_id=animal_id,
                analysis_type="feed",
                summary_status=status_tier,
                quality_score=final_score,
                details={
                    "feed_name": feed_name,
                    "category": matched_cat,
                    "quantity_kg": quantity_kg,
                    "quality_score": final_score,
                    "status": status_tier
                },
                is_demo=auth_ctx.is_demo,
                created_at=datetime.now(timezone.utc)
            )
            get_farm_cattle_repository().save_analysis_record(history_rec)

        return CombinedFeedAnalysisResponse(
            success=True,
            feed_name=feed_name,
            category=matched_cat,
            quantity_kg=quantity_kg,
            quality_score=final_score,
            status=status_tier,
            nutrition_reference=ref_response,
            nutrition_ml_predictions=ml_response,
            visual_screening=visual_response,
            risk_analysis=risk_obj,
            why=why_items,
            recommended_action=action_items,
            farm_id=farm_id,
            animal_id=animal_id,
            record_id=rec_id,
            persisted_at=now_iso,
            disclaimer="Screening and reference analysis. Laboratory chemical and microbiological assay required for definitive safety certification."
        )

    except HTTPException:
        raise
    except AppBaseException:
        raise
    except Exception as e:
        logger.error(f"Error in combined feed analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Combined feed analysis failed: {str(e)}"
        )


@router.post(
    "/silage",
    response_model=CombinedSilageAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Comprehensive Combined Silage Quality Analysis (ML + Chemistry + Vision + Risk)"
)
async def analyze_silage(
    request: Request,
    pH: float = Form(default=3.85, ge=2.5, le=9.0, description="Silage pH"),
    dm_s: float = Form(default=31.8, description="Silage dry matter percentage (%)"),
    cp_s: float = Form(default=14.0, description="Silage crude protein percentage (% DM)"),
    lactic_ac_s: float = Form(default=6.2, description="Silage lactic acid (% DM)"),
    acetic_ac_s: float = Form(default=1.8, description="Silage acetic acid (% DM)"),
    butyric_ac_s: float = Form(default=0.05, description="Silage butyric acid (% DM)"),
    ammonia_s: float = Form(default=6.5, description="Silage ammonia-N (% of total N)"),
    starch_s: float = Form(default=21.0, description="Silage starch (% DM)"),
    ndf_s: float = Form(default=46.5, description="Silage NDF (% DM)"),
    adf_s: float = Form(default=27.9, description="Silage ADF (% DM)"),
    farm_id: Optional[str] = Form(default=None, description="Optional farm ID"),
    animal_id: Optional[str] = Form(default=None, description="Optional animal ID"),
    image: Optional[UploadFile] = File(default=None, description="Optional image of silage bunker face / sample")
):
    """
    Consolidated Silage Analysis Endpoint:
    Combines XGBoost Quality Class ('ea'/'la'), FQI Regressor score, chemical fermentation analysis,
    visual mould screening, dynamic quality scoring, and clostridial hazard analysis.
    """
    try:
        # 0. Validate Ownership Context if farm_id/animal_id provided
        auth_ctx = ownership_guard.validate_request_ownership(
            request=request, farm_id=farm_id, animal_id=animal_id, require_auth=False
        )

        # 1. Prepare Silage Input Model
        silage_in = SilageInput(
            pH=pH,
            dm_s=dm_s,
            cp_s=cp_s,
            lactic_ac_s=lactic_ac_s,
            acetic_ac_s=acetic_ac_s,
            butyric_ac_s=butyric_ac_s,
            ammonia_s=ammonia_s,
            starch_s=starch_s,
            ndf_s=ndf_s,
            adf_s=adf_s
        )

        # 2. Run Comprehensive Silage ML & Screening Layer
        comprehensive_ml = silage_service.predict_comprehensive(silage_in)
        screening_res = comprehensive_ml.screening_result

        # 3. Secure Visual Screening if image provided
        visual_res = None
        if image is not None and image.filename:
            img_bytes = await image.read()
            if len(img_bytes) > 0:
                validate_image_file(img_bytes, filename=image.filename)
                visual_res = visual_mould_service.predict_silage_visual(img_bytes)

        # 4. Integrate Dynamic Score & Visual Penalties
        score = screening_res.composite_quality_score if screening_res else comprehensive_ml.fermentation_quality_index.predicted_fqi
        why_list = list(screening_res.why if screening_res else [])
        action_list = list(screening_res.recommended_action if screening_res else [])

        if visual_res is not None and visual_res.success:
            if visual_res.predicted_class == "SPOILED":
                score = max(0.0, score - 40.0)
                why_list.insert(0, "CRITICAL: Visual screening shows severe surface spoilage / slimy decomposition.")
                action_list.insert(0, "Discard spoiled silage immediately.")
            elif visual_res.predicted_class == "MOULD_RISK":
                score = max(15.0, score - 20.0)
                why_list.insert(0, "WARNING: Surface mould patches detected on silage bunker face.")
                action_list.insert(0, "Discard top 5-10 cm layer of mouldy silage before feeding.")
            elif visual_res.predicted_class == "POOR_FERMENTATION":
                score = max(25.0, score - 15.0)
                why_list.append("Visual texture indicates possible aerobic deterioration or heating.")
            else:
                why_list.append("Visual screening confirmed normal, well-preserved silage appearance.")
        elif visual_res is not None and not visual_res.success:
            why_list.append(f"Visual screening note: {visual_res.message or 'Image not recognized as silage; visual quality penalty skipped.'}")

        final_score = round(max(0.0, min(100.0, score)), 1)
        if final_score >= 75.0 and (visual_res is None or visual_res.predicted_class == "GOOD"):
            status_tier = "GOOD"
        elif final_score >= 45.0:
            status_tier = "CAUTION"
        else:
            status_tier = "UNSAFE"

        # 5. Risk Assessment
        risk_obj = risk_assessment_service.assess_silage_risk(
            visual_screening=visual_res,
            ph=pH,
            butyric_acid_pct=butyric_ac_s,
            ammonia_n_pct=ammonia_s
        )

        fermentation_metrics = {
            "pH": pH,
            "dry_matter_percent": dm_s,
            "moisture_percent": round(100.0 - dm_s, 2),
            "crude_protein_percent_dm": cp_s,
            "lactic_acid_percent_dm": lactic_ac_s,
            "acetic_acid_percent_dm": acetic_ac_s,
            "butyric_acid_percent_dm": butyric_ac_s,
            "ammonia_n_percent_total_n": ammonia_s,
            "fqi_score": comprehensive_ml.fermentation_quality_index.predicted_fqi,
            "fao_quality_class": comprehensive_ml.quality_classification.predicted_class
        }

        # 6. Save Persistent Analysis History if User Authenticated
        rec_id = None
        now_iso = None
        if auth_ctx.is_authenticated and auth_ctx.user_id:
            rec_id = f"rec_{uuid.uuid4().hex[:12]}"
            now_iso = datetime.now(timezone.utc).isoformat()
            history_rec = AnalysisRecord(
                record_id=rec_id,
                user_id=auth_ctx.user_id,
                farm_id=farm_id,
                animal_id=animal_id,
                analysis_type="silage",
                summary_status=status_tier,
                quality_score=final_score,
                details={
                    "quality_score": final_score,
                    "status": status_tier,
                    "fao_class": comprehensive_ml.quality_classification.predicted_class,
                    "fqi_score": comprehensive_ml.fermentation_quality_index.predicted_fqi
                },
                is_demo=auth_ctx.is_demo,
                created_at=datetime.now(timezone.utc)
            )
            get_farm_cattle_repository().save_analysis_record(history_rec)

        return CombinedSilageAnalysisResponse(
            success=True,
            quality_score=final_score,
            status=status_tier,
            fermentation_ml=comprehensive_ml,
            visual_screening=visual_res,
            risk_analysis=risk_obj,
            fermentation_metrics=fermentation_metrics,
            why=why_list,
            recommended_action=action_list,
            farm_id=farm_id,
            animal_id=animal_id,
            record_id=rec_id,
            persisted_at=now_iso,
            disclaimer="Silage screening analysis based on proximal fermentation metrics. Laboratory confirmation required for comprehensive microbiological analysis."
        )

    except HTTPException:
        raise
    except AppBaseException:
        raise
    except Exception as e:
        logger.error(f"Error in combined silage analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Combined silage analysis failed: {str(e)}"
        )
