"""
Silage Quality Screening & Fermentation Interpretation Service
Combines FQI, FAO Quality Classification, pH, moisture, and acid profiles into a dynamic screening result.
"""

from typing import Any, Dict, List, Optional, Tuple


def evaluate_silage_screening(
    predicted_fqi: float,
    predicted_class: str,
    class_confidence: float,
    ph: float,
    dm_s: float,
    cp_s: float,
    lactic_acid_pct: float,
    acetic_acid_pct: float,
    butyric_acid_pct: float,
    ammonia_n_pct: float
) -> Dict[str, Any]:
    """
    Evaluates silage fermentation quality and produces a comprehensive dynamic screening result:
    Status: GOOD | CAUTION | UNSAFE
    Dynamic Score: 0 - 100
    """
    why: List[str] = []
    actions: List[str] = []
    is_unsafe = False
    is_caution = False

    # Base score derived from FQI regression (0 - 100)
    score = predicted_fqi

    # 1. pH Evaluation
    if 3.6 <= ph <= 4.2:
        score += 5.0
        why.append(f"Silage pH ({ph:.2f}) is optimal (3.6 - 4.2), indicating rapid acidification and anaerobic stability.")
    elif 4.2 < ph <= 4.6:
        score -= 5.0
        is_caution = True
        why.append(f"Silage pH ({ph:.2f}) is marginally elevated (4.2 - 4.6); monitor aerobic stability at silo face.")
    elif ph > 4.6:
        score -= 20.0
        if ph > 5.0:
            is_unsafe = True
            why.append(f"Silage pH ({ph:.2f}) is critically high (>5.0), indicating severe failure of lactic preservation and high clostridial risk.")
        else:
            is_caution = True
            why.append(f"Silage pH ({ph:.2f}) indicates poor lactic acid preservation (>4.6).")
    else:  # ph < 3.6 (over-acidified)
        score -= 5.0
        why.append(f"Silage pH ({ph:.2f}) is very low (<3.6); check for high residual unbuffered acids.")

    # 2. Butyric Acid Evaluation (Clostridial Fermentation Indicator)
    if butyric_acid_pct < 0.1:
        score += 5.0
        why.append(f"Butyric acid ({butyric_acid_pct:.2f}% DM) is well below safety threshold (<0.1%), indicating no clostridial spoilage.")
    elif 0.1 <= butyric_acid_pct <= 0.4:
        score -= 15.0
        is_caution = True
        why.append(f"Butyric acid ({butyric_acid_pct:.2f}% DM) is elevated (0.1 - 0.4%), indicating secondary fermentation.")
        actions.append("Limit feeding amount and balance with dry fibrous roughage.")
    else:  # > 0.4% DM
        score -= 35.0
        is_unsafe = True
        why.append(f"Butyric acid ({butyric_acid_pct:.2f}% DM) is dangerously high (>0.4%), reflecting extensive clostridial degradation.")
        actions.append("DO NOT feed this silage to pregnant cows or calves due to risk of ketosis and digestive disorders.")

    # 3. Ammonia-N Evaluation (Proteolysis Indicator)
    if ammonia_n_pct < 8.0:
        score += 5.0
        why.append(f"Ammonia-N ({ammonia_n_pct:.1f}% total N) is low (<8%), confirming minimal protein breakdown during ensiling.")
    elif 8.0 <= ammonia_n_pct <= 12.0:
        why.append(f"Ammonia-N ({ammonia_n_pct:.1f}% total N) is acceptable (8 - 12%).")
    else:  # > 12.0%
        score -= 18.0
        is_caution = True
        if ammonia_n_pct > 16.0:
            is_unsafe = True
            why.append(f"Ammonia-N ({ammonia_n_pct:.1f}% total N) is excessive (>16%), indicating severe proteolysis and amine accumulation.")
        else:
            why.append(f"Ammonia-N ({ammonia_n_pct:.1f}% total N) is elevated (>12%), indicating noticeable protein breakdown.")

    # 4. Dry Matter / Moisture Evaluation
    moisture_pct = 100.0 - dm_s
    if 30.0 <= dm_s <= 38.0:
        score += 5.0
        why.append(f"Silage Dry Matter ({dm_s:.1f}%) and moisture ({moisture_pct:.1f}%) are in the ideal ensiling window.")
    elif dm_s < 28.0:
        score -= 10.0
        is_caution = True
        why.append(f"Silage is overly wet ({dm_s:.1f}% DM, {moisture_pct:.1f}% moisture), predisposing to nutrient run-off and clostridia.")
    elif dm_s > 42.0:
        score -= 10.0
        is_caution = True
        why.append(f"Silage is dry ({dm_s:.1f}% DM); difficult to compact, predisposing to aerobic mould development.")

    # 5. FAO Fermentation Quality Class
    if predicted_class == "ea":
        score += 5.0
        why.append(f"FAO Fermentation classification: Early Acidity ('ea') with {class_confidence * 100.0:.1f}% confidence (optimal preservation).")
    else:
        score -= 10.0
        is_caution = True
        why.append(f"FAO Fermentation classification: Late Acidity ('la') with {class_confidence * 100.0:.1f}% confidence (slower acid development).")

    # Final Dynamic Score (0 - 100)
    composite_score = round(max(0.0, min(100.0, score)), 1)

    # Screening Status Classification
    if is_unsafe or composite_score < 45.0:
        screening_status = "UNSAFE"
    elif is_caution or composite_score < 75.0:
        screening_status = "CAUTION"
    else:
        screening_status = "GOOD"

    # Default actions
    actions.append("Advance silage face by at least 15-20 cm daily to prevent aerobic heating.")
    actions.append("Discard any visibly moulded or rotten silage from top layer before feeding.")
    if screening_status == "GOOD":
        actions.append("Silage is well-preserved and suitable for standard daily feeding rations.")

    return {
        "screening_status": screening_status,
        "composite_quality_score": composite_score,
        "fermentation_tier": "Optimal / High Quality" if screening_status == "GOOD" else ("Moderate / Caution" if screening_status == "CAUTION" else "Poor / High Risk"),
        "why": why,
        "recommended_action": actions,
        "screening_type": "silage_quality_screening",
        "disclaimer": "Silage quality screening result based on proximal fermentation indicators. Laboratory confirmation required for comprehensive microbiological analysis."
    }
