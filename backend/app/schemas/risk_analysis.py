"""
Contamination & Adulteration Risk Analysis Schemas
"""

from typing import Dict, Optional
from pydantic import BaseModel, Field


class ActiveRiskItem(BaseModel):
    """Risk item supported by an active model or rule-based screening engine."""
    level: str = Field(..., description="Risk tier: 'LOW', 'MEDIUM', 'HIGH', or 'CRITICAL'")
    basis: str = Field(..., description="Evidence basis: 'REAL_MODEL_OUTPUT' or 'SCREENING_RULE_BASED'")
    details: str = Field(..., description="Explanation of observed indicators")


class UnavailableRiskItem(BaseModel):
    """Contamination/adulteration target that strictly requires physical lab sensors."""
    status: str = Field(default="NOT_AVAILABLE", description="Availability status ('NOT_AVAILABLE')")
    message: str = Field(..., description="Reason and required physical testing instrument")


class ComprehensiveRiskAnalysis(BaseModel):
    """Consolidated risk assessment covering biological, chemical, and physical hazards."""
    mould_risk: ActiveRiskItem = Field(..., description="Visual mould risk evaluation")
    spoilage_risk: ActiveRiskItem = Field(..., description="Microbial & aerobic spoilage risk evaluation")
    urea_adulteration: UnavailableRiskItem = Field(
        default_factory=lambda: UnavailableRiskItem(
            status="NOT_AVAILABLE",
            message="Laboratory wet chemistry / urease testing or NIR sensor confirmation required."
        ),
        description="Urea / non-protein nitrogen adulteration status"
    )
    sand_silica_contamination: UnavailableRiskItem = Field(
        default_factory=lambda: UnavailableRiskItem(
            status="NOT_AVAILABLE",
            message="Acid-insoluble ash (AIA) laboratory muffle furnace test required."
        ),
        description="Sand / silica / acid-insoluble ash contamination status"
    )
    mycotoxin: UnavailableRiskItem = Field(
        default_factory=lambda: UnavailableRiskItem(
            status="NOT_AVAILABLE",
            message="Quantitative HPLC / LC-MS/MS or ELISA laboratory confirmation required."
        ),
        description="Mycotoxins (Aflatoxin B1, DON, Zearalenone, Fumonisin) status"
    )
    disclaimer: str = Field(
        default="Screening risk evaluation. Never assume absence of invisible chemical adulterants without laboratory validation.",
        description="Risk assessment disclaimer"
    )
