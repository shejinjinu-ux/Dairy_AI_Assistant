"""
Feed Quality Scoring & Agronomic Interpretation Service
Dynamically evaluates proximal fractions, converts units, and computes quality index.
"""

from typing import Any, Dict, List, Optional, Tuple


def g_per_kg_to_percentage(value_g_per_kg: float) -> float:
    """
    Converts grams per kilogram to percentage.
    Example: 98.17 g/kg -> 9.817%
    """
    return round(value_g_per_kg / 10.0, 3)


def calculate_feed_quality_score(
    feed_category: str,
    dry_matter_g_per_kg: float,
    crude_protein_g_per_kg_dm: float,
    crude_fibre_g_per_kg_dm: float,
    ndf_g_per_kg_dm: float,
    adf_g_per_kg_dm: float,
    adl_g_per_kg_dm: Optional[float] = None,
    starch_g_per_kg_dm: Optional[float] = None
) -> Tuple[float, str, List[str], List[str]]:
    """
    Computes a dynamic quality score (0 - 100), quality tier, explainability facts (why),
    and actionable recommendations based on actual predicted proximal values.
    """
    dm_pct = g_per_kg_to_percentage(dry_matter_g_per_kg)
    cp_pct = g_per_kg_to_percentage(crude_protein_g_per_kg_dm)
    cf_pct = g_per_kg_to_percentage(crude_fibre_g_per_kg_dm)
    ndf_pct = g_per_kg_to_percentage(ndf_g_per_kg_dm)
    adf_pct = g_per_kg_to_percentage(adf_g_per_kg_dm)
    adl_pct = g_per_kg_to_percentage(adl_g_per_kg_dm) if adl_g_per_kg_dm is not None else None
    starch_pct = g_per_kg_to_percentage(starch_g_per_kg_dm) if starch_g_per_kg_dm is not None else None

    category_lower = feed_category.lower()
    score = 70.0
    why: List[str] = []
    actions: List[str] = []

    # 1. Crude Protein Evaluation
    if "concentrate" in category_lower or "cake" in category_lower or "meal" in category_lower:
        if cp_pct >= 35.0:
            score += 15.0
            why.append(f"Crude Protein is high ({cp_pct:.1f}% DM), providing excellent protein supplementation.")
        elif cp_pct >= 20.0:
            score += 8.0
            why.append(f"Crude Protein is balanced ({cp_pct:.1f}% DM) for compound dairy feed.")
        elif cp_pct < 12.0:
            score -= 15.0
            why.append(f"Crude Protein ({cp_pct:.1f}% DM) is low for a concentrate matrix.")
    elif "silage" in category_lower:
        if cp_pct >= 8.5:
            score += 8.0
            why.append(f"Crude Protein ({cp_pct:.1f}% DM) is optimal for preserved whole-plant cereal silage.")
        else:
            score -= 5.0
            why.append(f"Crude Protein ({cp_pct:.1f}% DM) is moderate.")
    else:  # Roughage / Green Forage / Straw
        if cp_pct >= 15.0:
            score += 15.0
            why.append(f"Crude Protein is superior ({cp_pct:.1f}% DM) characteristic of high-quality legume fodder.")
        elif cp_pct >= 8.0:
            score += 5.0
            why.append(f"Crude Protein ({cp_pct:.1f}% DM) meets maintenance and green forage baseline.")
        else:
            score -= 10.0
            why.append(f"Crude Protein ({cp_pct:.1f}% DM) is low; requires protein supplementation.")

    # 2. Dry Matter & Moisture Stability Evaluation
    if "concentrate" in category_lower or "byproduct" in category_lower or "dry" in category_lower:
        if 86.0 <= dm_pct <= 93.0:
            score += 5.0
            why.append(f"Dry Matter ({dm_pct:.1f}%) is in the safe storage band (<14% moisture).")
            actions.append("Store feed on elevated wooden pallets in a dry, well-ventilated storeroom.")
        elif dm_pct < 85.0:
            score -= 18.0
            why.append(f"Moisture is high ({100.0 - dm_pct:.1f}%), presenting increased risk of mould development during storage.")
            actions.append("Aerate and dry feed immediately before storage to prevent mould growth.")
    elif "silage" in category_lower:
        if 30.0 <= dm_pct <= 38.0:
            score += 10.0
            why.append(f"Silage Dry Matter ({dm_pct:.1f}%) is in the ideal ensiling window (62-70% moisture).")
        elif dm_pct < 28.0:
            score -= 12.0
            why.append(f"Silage is overly wet ({dm_pct:.1f}% DM), risking nutrient seepage and clostridial fermentation.")
            actions.append("Feed with dry roughage (e.g. straw) to balance rumen moisture load.")
        else:
            score -= 8.0
            why.append(f"Silage is dry ({dm_pct:.1f}% DM), requiring rigorous silo compaction to prevent aerobic spoilage.")
    else:  # Fresh Green Forage
        if 16.0 <= dm_pct <= 25.0:
            score += 5.0
            why.append(f"Green forage moisture content ({100.0 - dm_pct:.1f}%) is optimal for palatability.")
        elif dm_pct < 12.0:
            score -= 6.0
            why.append(f"Forage is highly watery ({100.0 - dm_pct:.1f}% moisture); may restrict dry matter intake.")

    # 3. Fibre Fractions & Digestibility (NDF, ADF, ADL)
    if adf_pct > 42.0:
        score -= 12.0
        why.append(f"Acid Detergent Fibre (ADF: {adf_pct:.1f}% DM) is high, indicating reduced energy digestibility.")
    elif adf_pct < 30.0:
        score += 6.0
        why.append(f"ADF ({adf_pct:.1f}% DM) is low, indicating high ruminal digestibility.")

    if adl_pct is not None:
        if adl_pct > 6.0:
            score -= 10.0
            why.append(f"Acid Detergent Lignin (ADL: {adl_pct:.1f}% DM) is elevated, creating resistant fibre barriers.")
        elif adl_pct < 3.5:
            score += 5.0
            why.append(f"Lignification is low (ADL: {adl_pct:.1f}% DM), enabling high cellulose digestibility.")

    # 4. Energy / Starch Fraction
    if starch_pct is not None:
        if starch_pct >= 50.0 and ("grain" in category_lower or "concentrate" in category_lower):
            score += 8.0
            why.append(f"Starch content ({starch_pct:.1f}% DM) provides readily fermentable glucogenic energy.")
        elif starch_pct >= 22.0 and "silage" in category_lower:
            score += 6.0
            why.append(f"Silage grain starch ({starch_pct:.1f}% DM) supports robust microbial protein synthesis.")

    # Clamp Score (10.0 - 100.0)
    final_score = round(max(10.0, min(100.0, score)), 1)

    # Determine Quality Tier
    if final_score >= 85.0:
        status = "EXCELLENT"
    elif final_score >= 70.0:
        status = "GOOD"
    elif final_score >= 50.0:
        status = "FAIR"
    else:
        status = "POOR"

    # Default action if none
    if not actions:
        actions.append("Maintain routine feeding protocols and inspect for foreign matter before feeding.")
    actions.append("Ensure clean, potable drinking water is available ad libitum.")

    return final_score, status, why, actions
