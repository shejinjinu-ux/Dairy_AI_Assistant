"""
Rule-Based Visual Mould & Spoilage Screening Service (Method 1)
Lightweight heuristic computer vision analyzer for visual mould and spoilage risk screening.
Note: This is a rule-based computer vision analyzer based on chromatic anomaly, spatial neighborhood
clustering, and texture gradient heuristics. It is NOT a trained deep learning neural network model
(no trained weights or labeled image datasets exist in the repository).
"""

import io
import logging
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

from backend.app.schemas.visual_screening import (
    FeedVisualScreeningResponse,
    SilageVisualScreeningResponse,
    VisualIndicators
)
from backend.app.core.exceptions import ImageProcessingError

logger = logging.getLogger("dairy_ai.visual_mould")


FEED_CLASSES = ["GOOD", "MOULD_RISK", "SPOILED"]
SILAGE_CLASSES = ["GOOD", "MOULD_RISK", "SPOILED", "POOR_FERMENTATION"]


class VisualMouldScreeningService:
    """
    Service handling rule-based visual mould, fungal risk, and aerobic spoilage screening.
    Operates via spatial neighborhood density filtering, chromatic segmentation, and texture heuristics.
    Requires multi-evidence agreement before assigning mould or spoilage risk to prevent false positives
    from specular reflections, kernel boundaries, or natural shadows.
    """

    def validate_and_load_image(self, file_bytes: bytes, max_size_mb: float = 15.0) -> Image.Image:
        """Validates image payload bytes and converts to RGB PIL Image."""
        if not file_bytes or len(file_bytes) == 0:
            raise ImageProcessingError("Uploaded image file is empty (0 bytes).")

        if len(file_bytes) > max_size_mb * 1024 * 1024:
            raise ImageProcessingError(f"Image file size exceeds maximum limit of {max_size_mb} MB.")

        try:
            image = Image.open(io.BytesIO(file_bytes))
            image.verify()  # Verify image integrity
            # Reopen for actual processing after verify()
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            return image
        except Exception as e:
            raise ImageProcessingError(f"Failed to decode valid image: {str(e)}")

    def extract_visual_features(
        self,
        image: Image.Image,
        is_silage: bool = False
    ) -> Tuple[VisualIndicators, float, float, Dict[str, float]]:
        """
        Extracts spatial density-filtered chromatic and textural indicators from the image:
        - Surface discolouration index
        - Texture roughness
        - Clustered white mycelium hyphae percentage
        - Clustered green-grey mould percentage
        - Clustered dark rot decomposition percentage
        """
        img_resized = image.resize((224, 224))
        arr = np.array(img_resized, dtype=np.float32) / 255.0  # (224, 224, 3)

        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]

        # 1. Grayscale luminance
        gray = 0.299 * r + 0.587 * g + 0.114 * b

        # 2. Saturation
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        sat = (max_c - min_c) / (max_c + 1e-6)

        # 3. Candidate White / Grey Mycelium (high brightness, low saturation)
        raw_white_mask = (gray > 0.82) & (sat < 0.12)

        # 4. Candidate Green-Grey Mould Spores
        is_yellow_or_brown = (float(np.mean(r)) > 0.38) and (float(np.mean(r)) >= float(np.mean(g)))
        if is_yellow_or_brown:
            raw_green_mask = (g > r + 0.10) & (g > b + 0.08) & (gray < 0.55) & (sat > 0.18)
        else:
            raw_green_mask = (g > r + 0.18) & (b > 0.35) & (sat > 0.25)

        # 5. Candidate Dark Rotten Decomposition (pitch black sludge)
        raw_dark_rot = (gray < 0.12)

        # 6. Candidate Aerobic Heating on Silage (caramelized dark dry patches)
        if is_silage:
            raw_aerobic_heat = (gray < 0.28) & (gray >= 0.12) & (r >= g)
        else:
            raw_aerobic_heat = np.zeros_like(gray, dtype=bool)

        # 7. Spatial Morphological Neighborhood Density (11x11 window)
        # Discards isolated 1-pixel specular reflections or thin interstitial kernel shadows.
        # Retains genuine contiguous fungal colonies and rotting regions.
        white_density = uniform_filter(raw_white_mask.astype(np.float32), size=11, mode='reflect')
        green_density = uniform_filter(raw_green_mask.astype(np.float32), size=11, mode='reflect')
        dark_rot_density = uniform_filter(raw_dark_rot.astype(np.float32), size=11, mode='reflect')
        aerobic_density = uniform_filter(raw_aerobic_heat.astype(np.float32), size=11, mode='reflect') if is_silage else np.zeros_like(gray)

        # Coherent cluster requires local neighborhood density > 0.35
        clustered_white_mask = raw_white_mask & (white_density > 0.35)
        clustered_green_mask = raw_green_mask & (green_density > 0.35)
        clustered_dark_rot = raw_dark_rot & (dark_rot_density > 0.35)
        clustered_aerobic = raw_aerobic_heat & (aerobic_density > 0.35) if is_silage else np.zeros_like(gray, dtype=bool)

        clustered_white_pct = float(np.mean(clustered_white_mask))
        clustered_green_pct = float(np.mean(clustered_green_mask))
        clustered_dark_rot_pct = float(np.mean(clustered_dark_rot))
        clustered_aerobic_pct = float(np.mean(clustered_aerobic))

        total_clustered_anomaly = (
            clustered_white_pct * 2.0 +
            clustered_green_pct * 2.0 +
            clustered_dark_rot_pct +
            (clustered_aerobic_pct if is_silage else 0.0)
        )
        discolouration_index = float(np.clip(total_clustered_anomaly * 4.0, 0.0, 1.0))

        # 8. High-frequency texture roughness (gradient magnitude)
        gx = np.diff(gray, axis=1)
        gy = np.diff(gray, axis=0)
        grad_mag = np.sqrt(gx[:-1, :] ** 2 + gy[:, :-1] ** 2)
        roughness_score = float(np.clip(float(np.mean(grad_mag)) * 2.5, 0.0, 1.0))

        # Multi-evidence flags requiring substantial clustered coverage
        has_mould_clusters = bool(
            clustered_white_pct > 0.035 or
            clustered_green_pct > 0.030 or
            (clustered_white_pct + clustered_green_pct) > 0.040
        )
        has_hyphae = bool(clustered_white_pct > 0.025 or clustered_green_pct > 0.025)

        indicators = VisualIndicators(
            surface_discolouration_index=round(discolouration_index, 3),
            dark_or_mould_cluster_spots=has_mould_clusters,
            texture_roughness_score=round(roughness_score, 3),
            white_grey_hyphae_indicators=has_hyphae
        )

        metrics = {
            "clustered_white_pct": clustered_white_pct,
            "clustered_green_pct": clustered_green_pct,
            "clustered_dark_rot_pct": clustered_dark_rot_pct,
            "clustered_aerobic_pct": clustered_aerobic_pct
        }

        return indicators, discolouration_index, roughness_score, metrics

    def predict_feed_visual(self, file_bytes: bytes) -> FeedVisualScreeningResponse:
        """
        Screens feed sample image for visual mould and spoilage risk using calibrated rule-based computer vision.
        Probabilities are strictly normalized in [0.0, 1.0] and sum to 1.0.
        """
        image = self.validate_and_load_image(file_bytes)
        indicators, discolouration, roughness, metrics = self.extract_visual_features(image, is_silage=False)

        c_white = metrics["clustered_white_pct"]
        c_green = metrics["clustered_green_pct"]
        c_dark = metrics["clustered_dark_rot_pct"]

        has_severe_spoilage = bool(
            c_dark > 0.30 or
            (c_dark > 0.15 and (c_white + c_green) > 0.04)
        )
        has_mould_clusters = indicators.dark_or_mould_cluster_spots

        # Multi-evidence decision logits
        if has_severe_spoilage:
            logit_good = -1.5
            logit_mould = 0.5
            logit_spoiled = 3.5
        elif has_mould_clusters:
            logit_good = -1.0
            logit_mould = 3.5
            logit_spoiled = -0.5
        else:
            logit_good = 3.0 - 2.5 * discolouration
            logit_mould = -1.5 + 4.0 * (c_white + c_green)
            logit_spoiled = -2.5 + 3.0 * c_dark

        logits = np.array([logit_good, logit_mould, logit_spoiled], dtype=np.float32)
        exp_logits = np.exp(logits - np.max(logits))
        raw_probs = exp_logits / np.sum(exp_logits)

        p_good = max(0.0, min(1.0, round(float(raw_probs[0]), 4)))
        p_mould = max(0.0, min(1.0, round(float(raw_probs[1]), 4)))
        p_spoiled = max(0.0, min(1.0, round(1.0 - (p_good + p_mould), 4)))

        prob_map = {
            "GOOD": p_good,
            "MOULD_RISK": p_mould,
            "SPOILED": p_spoiled
        }

        # Class determination based on highest rule agreement probability
        pred_class = max(prob_map, key=prob_map.get)
        confidence = prob_map[pred_class]
        confidence_pct = round(confidence * 100.0, 1)

        # Risk Tier
        if pred_class == "SPOILED":
            risk_level = "CRITICAL"
        elif pred_class == "MOULD_RISK":
            risk_level = "HIGH"
        else:
            risk_level = "LOW"

        # Explainability & Recommendations
        why: List[str] = []
        actions: List[str] = []

        if pred_class == "GOOD":
            why.append("Surface coloration is uniform and consistent with clean, dry feed.")
            why.append(f"Low surface discolouration index ({discolouration:.2f}) and no fungal patches detected.")
            actions.append("Store feed in a clean, elevated, well-ventilated dry area.")
            actions.append("Inspect regularly for insect pests or water seepage.")
        elif pred_class == "MOULD_RISK":
            why.append("Surface discolouration and localized fungal cluster spots detected on sample.")
            why.append(f"Elevated texture anomaly index ({roughness:.2f}) indicates possible mycelial colonization.")
            actions.append("Isolate the affected batch immediately to prevent fungal proliferation.")
            actions.append("DO NOT feed visibly mouldy portions to dairy cows, pregnant cattle, or calves.")
            actions.append("Submit a sample for laboratory ELISA/HPLC analysis if mycotoxin contamination is suspected.")
        else:  # SPOILED
            why.append("Severe surface discolouration, extensive dark/grey/slimy patches, and texture breakdown detected.")
            why.append("High risk of advanced biological decomposition and secondary spoilage.")
            actions.append("Discard this feed batch; unsuitable for animal consumption.")
            actions.append("Clean and disinfect storage bins before introducing new feed stock.")

        return FeedVisualScreeningResponse(
            success=True,
            predicted_class=pred_class,
            confidence=confidence,
            confidence_percentage=confidence_pct,
            risk_level=risk_level,
            screening_type="rule_based_visual_mould_screening",
            probabilities=prob_map,
            visual_indicators=indicators,
            why=why,
            recommended_action=actions,
            disclaimer="Rule-based visual screening only (not an ML model). Laboratory assay required for chemical toxins and mycotoxins."
        )

    def predict_silage_visual(self, file_bytes: bytes) -> SilageVisualScreeningResponse:
        """
        Screens silage sample image for visual mould, aerobic heating, and spoilage risk using calibrated rule-based computer vision.
        Probabilities are strictly normalized in [0.0, 1.0] and sum to 1.0.
        """
        image = self.validate_and_load_image(file_bytes)
        indicators, discolouration, roughness, metrics = self.extract_visual_features(image, is_silage=True)

        c_white = metrics["clustered_white_pct"]
        c_green = metrics["clustered_green_pct"]
        c_dark = metrics["clustered_dark_rot_pct"]
        c_aerobic = metrics["clustered_aerobic_pct"]

        has_severe_spoilage = bool(
            c_dark > 0.30 or
            (c_dark > 0.15 and (c_white + c_green) > 0.04)
        )
        has_mould_clusters = indicators.dark_or_mould_cluster_spots
        has_aerobic_spoilage = bool(c_aerobic > 0.25)

        # Silage multi-class logits: GOOD, MOULD_RISK, SPOILED, POOR_FERMENTATION
        if has_severe_spoilage:
            logit_good, logit_mould, logit_spoiled, logit_poor = -1.5, 0.0, 3.5, 0.0
        elif has_mould_clusters:
            logit_good, logit_mould, logit_spoiled, logit_poor = -1.0, 3.5, -0.5, 0.0
        elif has_aerobic_spoilage:
            logit_good, logit_mould, logit_spoiled, logit_poor = -0.5, 0.0, -0.5, 3.0
        else:
            logit_good = 3.0 - 2.5 * discolouration
            logit_mould = -1.5 + 4.0 * (c_white + c_green)
            logit_spoiled = -2.5 + 3.0 * c_dark
            logit_poor = -1.5 + 3.0 * c_aerobic

        logits = np.array([logit_good, logit_mould, logit_spoiled, logit_poor], dtype=np.float32)
        exp_logits = np.exp(logits - np.max(logits))
        raw_probs = exp_logits / np.sum(exp_logits)

        p_good = max(0.0, min(1.0, round(float(raw_probs[0]), 4)))
        p_mould = max(0.0, min(1.0, round(float(raw_probs[1]), 4)))
        p_spoiled = max(0.0, min(1.0, round(float(raw_probs[2]), 4)))
        p_poor = max(0.0, min(1.0, round(1.0 - (p_good + p_mould + p_spoiled), 4)))

        prob_map = {
            "GOOD": p_good,
            "MOULD_RISK": p_mould,
            "SPOILED": p_spoiled,
            "POOR_FERMENTATION": p_poor
        }

        pred_class = max(prob_map, key=prob_map.get)
        confidence = prob_map[pred_class]
        confidence_pct = round(confidence * 100.0, 1)

        if pred_class == "SPOILED":
            risk_level = "CRITICAL"
        elif pred_class == "MOULD_RISK":
            risk_level = "HIGH"
        elif pred_class == "POOR_FERMENTATION":
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        why: List[str] = []
        actions: List[str] = []

        if pred_class == "GOOD":
            why.append("Silage displays characteristic olive-green/golden coloration with no visible surface mould.")
            actions.append("Advance silo face at least 15-20 cm daily to maintain aerobic stability.")
        elif pred_class == "MOULD_RISK":
            why.append("Visual evidence of white/grey/green surface mould clusters on silage face.")
            actions.append("Manually discard the top 5-10 cm layer of mouldy silage before daily feeding.")
            actions.append("Check silage compaction density and plastic sheeting integrity.")
        elif pred_class == "POOR_FERMENTATION":
            why.append("Surface texture and dark discolouration suggest slow acid production or aerobic heating.")
            actions.append("Check silage pH and compaction; pack tightly and minimize air infiltration.")
        else:  # SPOILED
            why.append("Extensive black/slimy decomposition and structural breakdown observed.")
            actions.append("Discard spoiled material immediately. Never feed rotten or slimy silage to cattle.")

        return SilageVisualScreeningResponse(
            success=True,
            predicted_class=pred_class,
            confidence=confidence,
            confidence_percentage=confidence_pct,
            risk_level=risk_level,
            screening_type="rule_based_silage_spoilage_screening",
            probabilities=prob_map,
            visual_indicators=indicators,
            why=why,
            recommended_action=actions,
            disclaimer="Rule-based visual screening only (not an ML model). Laboratory assay required for chemical toxins and mycotoxins."
        )


visual_mould_service = VisualMouldScreeningService()
