"""
Feed Reference Database & Quantity Analysis Schemas
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class FeedReferenceRequest(BaseModel):
    """Input payload for feed reference lookup and quantity-based calculation."""
    feed_name: str = Field(
        ...,
        description="Common, regional, or scientific name of the feed ingredient (e.g. 'Maize', 'Maize Silage', 'Napier Grass', 'Wheat Bran').",
        min_length=1,
        examples=["Maize", "Maize Silage", "Napier Grass", "Wheat Bran", "Cottonseed Cake"]
    )
    quantity_kg: float = Field(
        default=1.0,
        gt=0.0,
        le=500.0,
        description="Fresh feed quantity in kilograms.",
        examples=[5.0, 10.0, 20.0]
    )


class NutrientProfile(BaseModel):
    """Nutritional delivery profile (grams or MJ energy)."""
    dry_matter_g: Optional[float] = Field(None, description="Dry matter in grams")
    crude_protein_g: Optional[float] = Field(None, description="Crude protein in grams")
    crude_fibre_g: Optional[float] = Field(None, description="Crude fibre in grams")
    ndf_g: Optional[float] = Field(None, description="Neutral detergent fibre in grams")
    adf_g: Optional[float] = Field(None, description="Acid detergent fibre in grams")
    adl_g: Optional[float] = Field(None, description="Acid detergent lignin in grams (or null if unavailable)")
    starch_g: Optional[float] = Field(None, description="Starch in grams (or null if unavailable)")
    ether_extract_g: Optional[float] = Field(None, description="Ether extract / crude fat in grams")
    ash_g: Optional[float] = Field(None, description="Total ash / minerals in grams")
    energy_mj: Optional[float] = Field(None, description="Metabolizable energy in MegaJoules (MJ)")
    calcium_g: Optional[float] = Field(None, description="Calcium in grams")
    phosphorus_g: Optional[float] = Field(None, description="Phosphorus in grams")


class FeedReferenceItem(BaseModel):
    """Catalog item from the authoritative feed composition reference database."""
    feed_name: str = Field(..., description="Canonical feed name")
    category: str = Field(..., description="Agronomic category (e.g. Green Roughage, Silage, Concentrate, Byproduct)")
    dry_matter_pct: float = Field(..., description="Dry matter percentage (% as-fed)")
    crude_protein_pct: float = Field(..., description="Crude protein percentage (% DM)")
    crude_fibre_pct: float = Field(..., description="Crude fibre percentage (% DM)")
    ndf_pct: float = Field(..., description="NDF percentage (% DM)")
    adf_pct: float = Field(..., description="ADF percentage (% DM)")
    adl_pct: Optional[float] = Field(None, description="ADL percentage (% DM)")
    starch_pct: Optional[float] = Field(None, description="Starch percentage (% DM)")
    ether_extract_pct: float = Field(..., description="Ether extract / fat percentage (% DM)")
    ash_pct: float = Field(..., description="Total ash percentage (% DM)")
    energy_mj_kg: float = Field(..., description="Metabolizable Energy (MJ/kg DM)")
    calcium_g_kg: float = Field(..., description="Calcium content (g/kg DM)")
    phosphorus_g_kg: float = Field(..., description="Phosphorus content (g/kg DM)")
    source: str = Field(..., description="Authoritative reference source citation")


class FeedReferenceResponse(BaseModel):
    """Output schema for feed reference nutrition calculation."""
    success: bool = Field(default=True, description="Calculation status")
    feed_name: str = Field(..., description="Original requested feed name")
    matched_feed_name: str = Field(..., description="Matched canonical feed name from reference catalog")
    category: str = Field(..., description="Feed category")
    quantity_kg: float = Field(..., description="Quantity calculated for in kilograms")
    basis: str = Field(default="reference", description="Calculation basis ('reference')")
    per_kg: NutrientProfile = Field(..., description="Nutritional contribution per 1 kg fresh feed")
    total_for_quantity: NutrientProfile = Field(..., description="Total nutritional delivery for quantity_kg (per_kg * quantity_kg)")
    nutrient_percentages_dm: Dict[str, Optional[float]] = Field(
        ...,
        description="Standard proximate composition percentages (% DM basis)"
    )
    source: str = Field(..., description="Authoritative data source citation")
    disclaimer: str = Field(
        default="Reference nutritional values; actual batch composition may vary.",
        description="Reference calculation scope disclaimer"
    )


class FeedCatalogResponse(BaseModel):
    """Output schema for offline catalog export."""
    success: bool = Field(default=True, description="Success status")
    total_feeds: int = Field(..., description="Total verified feed ingredients in database")
    source_database: str = Field(default="ICAR-NIANP Indian Feed Composition Tables (2013/2024)")
    feeds: List[FeedReferenceItem] = Field(..., description="List of all catalogued feed ingredients")
    disclaimer: str = Field(
        default="Reference nutritional values for offline cache; actual field composition may vary depending on crop maturity, season, and processing.",
        description="Disclaimer"
    )
