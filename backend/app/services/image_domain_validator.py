"""
Image Domain Validation Service for Cattle Feed & Silage Quality Screening.

Enforces strict domain boundaries before visual feature extraction:
1. Rejects human portraits, selfies, faces, and full body photos as INVALID_IMAGE.
2. ACCEPTS close-up feed/silage samples held or touched by a farmer's hand.
3. Rejects animals/cattle-only photos (without feed) as INVALID_IMAGE.
4. Rejects non-feed objects (vehicles, electronics, furniture) as INVALID_IMAGE.
5. Rejects distant landscape and blank screen photos.
6. Passes genuine cattle feed and silage samples to quality screening.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
from PIL import Image

try:
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger("dairy_ai.image_domain_validator")

# ImageNet semantic label groupings for non-feed domain rejection
HUMAN_PERSON_KEYWORDS = [
    "person", "human", "man", "woman", "boy", "girl", "face",
    "wig", "jersey", "trench coat", "suit", "cloak", "gown", "kimono",
    "brassiere", "bikini", "pajama", "sombrero", "cowboy hat", "bonnet",
    "sweatshirt", "cardigan", "jean", "sock", "shoe", "sandal", "sunglass",
    "face powder", "lipstick", "lotion", "perfume", "diaper", "bib"
]

ANIMAL_KEYWORDS = [
    "dog", "cat", "cow", "ox", "bull", "calf", "bison", "water buffalo",
    "horse", "sheep", "goat", "ram", "pig", "boar", "donkey", "mule",
    "bird", "chicken", "rooster", "hen", "turkey", "duck", "goose",
    "gazelle", "zebra", "elephant", "lion", "tiger", "bear", "monkey"
]

NON_FEED_OBJECT_KEYWORDS = [
    "car", "truck", "automobile", "bus", "train", "motorcycle", "bicycle",
    "laptop", "notebook", "desktop", "computer", "screen", "monitor",
    "television", "tv", "cellphone", "cellular telephone", "ipod", "mouse",
    "keyboard", "printer", "camera", "clock", "watch", "telephone",
    "chair", "table", "desk", "sofa", "couch", "bed", "wardrobe", "cabinet",
    "bottle", "can", "cup", "mug", "plate", "fork", "knife", "spoon",
    "book", "binder", "packet", "carton", "box", "crate", "shoe", "boot"
]

LANDSCAPE_KEYWORDS = [
    "seashore", "coast", "beach", "mountain", "valley", "cliff",
    "lakeside", "lake", "river", "ocean", "sea", "sky", "cloud",
    "alp", "volcano", "promontory", "sandbar", "coral reef"
]

FEED_SILAGE_KEYWORDS = [
    "hay", "straw", "forage", "silage", "grass", "grain", "corn", "maize",
    "wheat", "barley", "oat", "sorghum", "fodder", "chaff", "ear",
    "comb", "honeycomb", "envelope", "sponge", "broom", "wool", "nematode"
]


@dataclass
class DomainValidationResult:
    is_valid: boolean if False else bool
    domain: str  # 'FEED_SAMPLE', 'SILAGE_SAMPLE', 'HUMAN', 'ANIMAL', 'OBJECT', 'LANDSCAPE', 'BLANK'
    error_type: Optional[str] = None  # 'INVALID_IMAGE' or None
    classification: Optional[str] = None  # 'NOT_FEED_OR_SILAGE', 'FEED_SAMPLE', 'SILAGE_SAMPLE'
    message: Optional[str] = None
    confidence: float = 0.0
    detected_labels: Optional[List[str]] = None


class ImageDomainValidator:
    """
    Two-stage domain validator:
    Stage 1: Calibrated Chromatic & Geometric Heuristics (Blank, Landscape, Human-Dominant vs Hand-Held Feed)
    Stage 2: Pretrained Semantic Classification via MobileNetV3 (Animal, Object, Scenery)
    """

    def __init__(self):
        self._model = None
        self._preprocess = None
        self._categories = None
        self._initialized = False

    def _lazy_init_model(self):
        """Initialize MobileNetV3 weights on first inference."""
        if self._initialized or not TORCH_AVAILABLE:
            return

        try:
            weights = MobileNet_V3_Small_Weights.DEFAULT
            self._model = mobilenet_v3_small(weights=weights)
            self._model.eval()
            self._preprocess = weights.transforms()
            self._categories = weights.meta["categories"]
            self._initialized = True
            logger.info("ImageDomainValidator: Pretrained MobileNetV3 model loaded successfully.")
        except Exception as e:
            logger.warning(f"ImageDomainValidator: Failed to load MobileNetV3 ({e}). Heuristics only mode.")
            self._initialized = False

    def validate_sample(self, image: Image.Image, target_domain: str = "feed") -> DomainValidationResult:
        """
        Validates whether the image is a valid cattle feed/silage sample (including hand-held samples).
        """
        # Step 1: Chromatic & Spatial Heuristics
        heuristic_res = self._check_heuristics(image, target_domain)
        if not heuristic_res.is_valid:
            return heuristic_res

        # Step 2: Semantic MobileNetV3 Classification
        if TORCH_AVAILABLE:
            self._lazy_init_model()
            if self._initialized:
                semantic_res = self._check_semantics(image, target_domain, heuristic_res.domain == "HAND_HELD")
                if not semantic_res.is_valid:
                    return semantic_res

        return DomainValidationResult(
            is_valid=True,
            domain=target_domain.upper(),
            classification="FEED_SAMPLE" if target_domain == "feed" else "SILAGE_SAMPLE",
            message="Valid sample image."
        )

    def _check_heuristics(self, image: Image.Image, target_domain: str) -> DomainValidationResult:
        """
        Pixel-level heuristic checks:
        1. Flat / blank / uniform screens
        2. Distant landscape / sky-heavy photos
        3. Human-dominant vs Hand-held feed evaluation
        """
        img_rgb = image.convert("RGB")
        img_small = img_rgb.resize((64, 64))
        arr = np.array(img_small, dtype=np.float32)  # (64, 64, 3)

        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]

        # Check A: Blank / Solid Color Screen (Low Variance)
        r_std = float(np.std(r))
        g_std = float(np.std(g))
        b_std = float(np.std(b))
        avg_std = (r_std + g_std + b_std) / 3.0

        if avg_std < 2.0:
            return DomainValidationResult(
                is_valid=False,
                domain="BLANK",
                error_type="INVALID_IMAGE",
                classification="NOT_FEED_OR_SILAGE",
                message=f"Please upload a clear close-up photo of the {target_domain} sample. The image appears completely blank or unreadable."
            )

        # Check B: Sky and Distant Landscape
        top_sky = (b[:22, :] > r[:22, :] + 15.0) & (b[:22, :] > 110.0)
        is_overcast = (r[:22, :] > 215.0) & (g[:22, :] > 215.0) & (b[:22, :] > 220.0)
        top_sky_ratio = float(np.mean(top_sky | is_overcast))

        bottom_green = (g[32:, :] > r[32:, :] + 12.0) & (g[32:, :] > b[32:, :] + 8.0)
        bottom_dark = ((r[32:, :] + g[32:, :] + b[32:, :]) / 3.0) < 70.0
        bottom_ratio = float(np.mean(bottom_green | bottom_dark))

        if (top_sky_ratio > 0.40 and bottom_ratio > 0.35) or (top_sky_ratio > 0.50):
            return DomainValidationResult(
                is_valid=False,
                domain="LANDSCAPE",
                error_type="INVALID_IMAGE",
                classification="NOT_FEED_OR_SILAGE",
                message=f"Please upload a clear close-up photo of the {target_domain} sample. Avoid distant landscape or sky photos."
            )

        # Check C: Calibrated Human Skin Tone vs Agricultural Feed/Silage Context
        hsv = img_small.convert("HSV")
        hsv_arr = np.array(hsv, dtype=np.float32)
        h = hsv_arr[:, :, 0] * 360.0 / 255.0  # 0-360
        s = hsv_arr[:, :, 1] / 255.0          # 0-1
        v = hsv_arr[:, :, 2] / 255.0          # 0-1

        # Skin Tone Mask (Red/Orange hue: 0-25 deg or 335-360 deg, moderate saturation & brightness)
        is_skin_hue = (h <= 25.0) | (h >= 335.0)
        is_skin_sat = (s >= 0.15) & (s <= 0.70)
        is_skin_val = (v >= 0.30) & (v <= 0.95)
        is_skin_rg = (r > g) & (g > b) & ((r - g) > 0.5 * (g - b + 1e-5))
        skin_mask = is_skin_hue & is_skin_sat & is_skin_val & is_skin_rg
        skin_pct = float(np.mean(skin_mask))

        # Agricultural Feed / Silage Color & Texture Mask (Non-skin regions)
        # Straw / Maize / Golden feed: H in 25-65 deg, S >= 0.18, V >= 0.25
        is_straw_maize = (h >= 25.0) & (h <= 65.0) & (s >= 0.18) & (v >= 0.25)
        # Silage / Green forage: H in 65-150 deg, S >= 0.15, V >= 0.20
        is_green_silage = (h >= 65.0) & (h <= 150.0) & (s >= 0.15) & (v >= 0.20)
        # Pellets / Brown meal: H in 15-45 deg, S >= 0.18, V in 0.15-0.70
        is_brown_feed = (h >= 15.0) & (h <= 45.0) & (s >= 0.18) & (v >= 0.15) & (v <= 0.70) & (g > b)

        feed_color_mask = (is_straw_maize | is_green_silage | is_brown_feed) & (~skin_mask)
        feed_color_pct = float(np.mean(feed_color_mask))

        # 1. Person-dominant selfie / portrait / headshot (skin occupies large portion of image)
        if skin_pct > 0.40:
            return DomainValidationResult(
                is_valid=False,
                domain="HUMAN",
                error_type="INVALID_IMAGE",
                classification="NOT_FEED_OR_SILAGE",
                message=f"The uploaded photo appears to be a person or human image. Please upload a clear close-up photo of cattle {target_domain} for quality testing."
            )

        # 2. Human / Person photo where feed is absent (skin present, but no agricultural feed context)
        if skin_pct > 0.06 and feed_color_pct < 0.20:
            return DomainValidationResult(
                is_valid=False,
                domain="HUMAN",
                error_type="INVALID_IMAGE",
                classification="NOT_FEED_OR_SILAGE",
                message=f"The uploaded photo appears to be a person or human image. Please upload a clear close-up photo of cattle {target_domain} for quality testing."
            )

        # 3. Hand-held feed / silage sample (skin present in minority, strong feed context in majority)
        if skin_pct > 0.06 and feed_color_pct >= 0.20:
            return DomainValidationResult(
                is_valid=True,
                domain="HAND_HELD",
                classification="FEED_SAMPLE" if target_domain == "feed" else "SILAGE_SAMPLE",
                message=f"Valid hand-held {target_domain} sample."
            )

        return DomainValidationResult(is_valid=True, domain=target_domain.upper())

    def _check_semantics(self, image: Image.Image, target_domain: str, is_hand_held: bool = False) -> DomainValidationResult:
        """
        Deep learning semantic classification check using MobileNetV3.
        """
        img_rgb = image.convert("RGB")
        tensor = self._preprocess(img_rgb).unsqueeze(0)

        with torch.inference_mode():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1)
            top5_probs, top5_indices = torch.topk(probs, 5)

        predictions: List[Tuple[str, float]] = []
        for p, idx in zip(top5_probs[0], top5_indices[0]):
            label = self._categories[idx.item()]
            prob_val = float(p.item())
            predictions.append((label, prob_val))

        top_label, top_prob = predictions[0]
        all_top_labels = [label for label, _ in predictions]

        # Compute cumulative probabilities for top categories
        cum_human_prob = sum(p for label, p in predictions[:3] if any(kw in label.lower() for kw in HUMAN_PERSON_KEYWORDS))
        cum_animal_prob = sum(p for label, p in predictions[:3] if any(kw in label.lower() for kw in ANIMAL_KEYWORDS))
        cum_object_prob = sum(p for label, p in predictions[:3] if any(kw in label.lower() for kw in NON_FEED_OBJECT_KEYWORDS))
        cum_landscape_prob = sum(p for label, p in predictions[:3] if any(kw in label.lower() for kw in LANDSCAPE_KEYWORDS))
        has_feed_co_occurrence = any(any(kw in label.lower() for kw in FEED_SILAGE_KEYWORDS) for label in all_top_labels[:3])

        # If image is already verified as hand-held feed by heuristics, allow incidental human clothing/apparel
        if not is_hand_held:
            # 1. Human / Person Detection (Confidently matched clothing/person or cumulative > 0.15)
            if (cum_human_prob >= 0.15 or (top_prob >= 0.20 and any(kw in top_label.lower() for kw in HUMAN_PERSON_KEYWORDS))) and not has_feed_co_occurrence:
                return DomainValidationResult(
                    is_valid=False,
                    domain="HUMAN",
                    error_type="INVALID_IMAGE",
                    classification="NOT_FEED_OR_SILAGE",
                    message=f"The uploaded photo appears to be a person or human image. Please upload a clear close-up photo of cattle {target_domain} for quality testing.",
                    confidence=top_prob,
                    detected_labels=all_top_labels
                )

        # 2. Animal / Cattle-Only Photo Detection (Confidently matched animal >= 0.25 without feed)
        if (cum_animal_prob >= 0.25 or (top_prob >= 0.20 and any(kw in top_label.lower() for kw in ANIMAL_KEYWORDS))) and not has_feed_co_occurrence:
            return DomainValidationResult(
                is_valid=False,
                domain="ANIMAL",
                error_type="INVALID_IMAGE",
                classification="NOT_FEED_OR_SILAGE",
                message=f"The photo shows an animal/cattle rather than a close-up feed or silage sample. Please upload a clear photo of the feed or silage itself.",
                confidence=top_prob,
                detected_labels=all_top_labels
            )

        # 3. Vehicles, Tech Gadgets & Indoor Objects (Confidently matched object >= 0.25 without feed)
        if (cum_object_prob >= 0.25 or (top_prob >= 0.20 and any(kw in top_label.lower() for kw in NON_FEED_OBJECT_KEYWORDS))) and not has_feed_co_occurrence:
            return DomainValidationResult(
                is_valid=False,
                domain="OBJECT",
                error_type="INVALID_IMAGE",
                classification="NOT_FEED_OR_SILAGE",
                message=f"The uploaded photo appears to be a non-agricultural object or vehicle ({top_label}). Please upload a clear photo of cattle {target_domain}.",
                confidence=top_prob,
                detected_labels=all_top_labels
            )

        # 4. Landscape / Nature Scene (Confidently matched landscape >= 0.25 without feed)
        if (cum_landscape_prob >= 0.25 or (top_prob >= 0.20 and any(kw in top_label.lower() for kw in LANDSCAPE_KEYWORDS))) and not has_feed_co_occurrence:
            return DomainValidationResult(
                is_valid=False,
                domain="LANDSCAPE",
                error_type="INVALID_IMAGE",
                classification="NOT_FEED_OR_SILAGE",
                message=f"The photo appears to be a landscape scene. Please upload a clear close-up sample of cattle {target_domain}.",
                confidence=top_prob,
                detected_labels=all_top_labels
            )

        return DomainValidationResult(
            is_valid=True,
            domain=target_domain.upper(),
            classification="FEED_SAMPLE" if target_domain == "feed" else "SILAGE_SAMPLE",
            confidence=top_prob,
            detected_labels=all_top_labels
        )


image_domain_validator = ImageDomainValidator()
