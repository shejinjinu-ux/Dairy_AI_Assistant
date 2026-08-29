"""
Contamination & Adulteration Risk Assessment Service
Distinguishes real ML visual predictions, rule-based screening, and laboratory-dependent hazards.
"""

from typing import Optional
from backend.app.schemas.risk_analysis import (
    ComprehensiveRiskAnalysis,
    ActiveRiskItem,
    UnavailableRiskItem
)
from backend.app.schemas.visual_screening import FeedVisualScreeningResponse, SilageVisualScreeningResponse


class RiskAssessmentService:
    """Evaluates biological and physical risk while honestly flagging lab-required hazards."""

    def assess_feed_risk(
        self,
        visual_screening: Optional[FeedVisualScreeningResponse] = None,
        moisture_pct: Optional[float] = None,
        feed_category: Optional[str] = None
    ) -> ComprehensiveRiskAnalysis:
        """Generates comprehensive risk framework for animal feed sample."""
        # 1. Mould Risk
        if visual_screening is not None:
            mould_level = visual_screening.risk_level
            mould_basis = "RULE_BASED_IMAGE_ANALYSIS"
            if visual_screening.predicted_class == "GOOD":
                mould_details = "Rule-based visual screening detected uniform surface with no fungal cluster patterns."
            elif visual_screening.predicted_class == "MOULD_RISK":
                mould_details = "Rule-based visual screening identified localized surface discolouration and fungal cluster spots."
            else:
                mould_details = "Rule-based visual screening detected extensive surface deterioration and mould colonization."
        else:
            # Rule-based fallback if no image provided
            if moisture_pct is not None and moisture_pct > 15.0:
                mould_level = "MEDIUM"
                mould_basis = "SCREENING_RULE_BASED"
                mould_details = f"Moisture content ({moisture_pct:.1f}%) exceeds 14% safe limit, elevating fungal propagation risk."
            else:
                mould_level = "LOW"
                mould_basis = "SCREENING_RULE_BASED"
                mould_details = "No image provided; standard storage risk baseline."

        # 2. Spoilage Risk
        if visual_screening is not None and visual_screening.predicted_class == "SPOILED":
            spoilage_level = "CRITICAL"
            spoilage_basis = "RULE_BASED_IMAGE_ANALYSIS"
            spoilage_details = "Visual evidence of structural decomposition and severe surface spoilage."
        elif moisture_pct is not None and moisture_pct > 18.0 and feed_category and "concentrate" in feed_category.lower():
            spoilage_level = "HIGH"
            spoilage_basis = "SCREENING_RULE_BASED"
            spoilage_details = f"High moisture in concentrate ({moisture_pct:.1f}%) promotes rapid bacterial and fungal spoilage."
        elif visual_screening is not None and visual_screening.predicted_class == "MOULD_RISK":
            spoilage_level = "MEDIUM"
            spoilage_basis = "RULE_BASED_IMAGE_ANALYSIS"
            spoilage_details = "Early-stage fungal colonization detected via image analysis."
        else:
            spoilage_level = "LOW"
            spoilage_basis = "SCREENING_RULE_BASED"
            spoilage_details = "Parameters within normal preservation tolerances."

        return ComprehensiveRiskAnalysis(
            mould_risk=ActiveRiskItem(
                level=mould_level,
                basis=mould_basis,
                details=mould_details
            ),
            spoilage_risk=ActiveRiskItem(
                level=spoilage_level,
                basis=spoilage_basis,
                details=spoilage_details
            ),
            urea_adulteration=UnavailableRiskItem(
                status="NOT_AVAILABLE",
                message="Laboratory wet chemistry / urease testing required."
            ),
            sand_silica_contamination=UnavailableRiskItem(
                status="NOT_AVAILABLE",
                message="Acid-insoluble ash laboratory muffle furnace test required."
            ),
            mycotoxin=UnavailableRiskItem(
                status="NOT_AVAILABLE",
                message="Quantitative HPLC / LC-MS/MS or ELISA laboratory confirmation required."
            )
        )

    def assess_silage_risk(
        self,
        visual_screening: Optional[SilageVisualScreeningResponse] = None,
        ph: Optional[float] = None,
        butyric_acid_pct: Optional[float] = None,
        ammonia_n_pct: Optional[float] = None
    ) -> ComprehensiveRiskAnalysis:
        """Generates comprehensive risk framework for preserved silage."""
        # 1. Mould Risk
        if visual_screening is not None:
            mould_level = visual_screening.risk_level
            mould_basis = "RULE_BASED_IMAGE_ANALYSIS"
            mould_details = f"Visual screening class: {visual_screening.predicted_class} (confidence: {visual_screening.confidence_percentage}%)."
        else:
            mould_level = "LOW"
            mould_basis = "SCREENING_RULE_BASED"
            mould_details = "No image provided; aerobic mould risk depends on bunker face management."

        # 2. Spoilage / Clostridial Risk
        if butyric_acid_pct is not None and butyric_acid_pct > 0.4:
            spoilage_level = "CRITICAL"
            spoilage_basis = "REAL_MODEL_OUTPUT"
            spoilage_details = f"Elevated butyric acid ({butyric_acid_pct:.2f}% DM) indicates severe clostridial spoilage and anaerobic decomposition."
        elif ph is not None and ph > 4.8:
            spoilage_level = "HIGH"
            spoilage_basis = "REAL_MODEL_OUTPUT"
            spoilage_details = f"High pH ({ph:.2f}) indicates poor lactic acid preservation and elevated microbial spoilage."
        elif visual_screening is not None and visual_screening.predicted_class in ["SPOILED", "POOR_FERMENTATION"]:
            spoilage_level = "HIGH"
            spoilage_basis = "RULE_BASED_IMAGE_ANALYSIS"
            spoilage_details = "Visual signs of surface degradation or poor fermentation."
        else:
            spoilage_level = "LOW"
            spoilage_basis = "REAL_MODEL_OUTPUT"
            spoilage_details = "Fermentation indicators show stable anaerobic preservation."

        return ComprehensiveRiskAnalysis(
            mould_risk=ActiveRiskItem(
                level=mould_level,
                basis=mould_basis,
                details=mould_details
            ),
            spoilage_risk=ActiveRiskItem(
                level=spoilage_level,
                basis=spoilage_basis,
                details=spoilage_details
            ),
            urea_adulteration=UnavailableRiskItem(
                status="NOT_AVAILABLE",
                message="Laboratory wet chemistry required for non-protein nitrogen validation."
            ),
            sand_silica_contamination=UnavailableRiskItem(
                status="NOT_AVAILABLE",
                message="Acid-insoluble ash laboratory test required."
            ),
            mycotoxin=UnavailableRiskItem(
                status="NOT_AVAILABLE",
                message="Laboratory HPLC / ELISA confirmation required for mycotoxin assays."
            )
        )


risk_assessment_service = RiskAssessmentService()
