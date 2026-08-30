"""
Comprehensive Unit & Integration Test Suite for
Smart AI-Enabled Rapid Feed and Silage Quality Testing System
Covers all 16 verification requirements from Step 14.
"""

import io
import pytest
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

from backend.main import app
from backend.app.schemas.feed_reference import FeedReferenceRequest
from backend.app.schemas.feed_nutrition import FeedNutritionInput
from backend.app.schemas.silage import SilageInput
from backend.app.services.feed_reference_service import feed_reference_service
from backend.app.services.feed_scoring import (
    g_per_kg_to_percentage,
    calculate_feed_quality_score
)
from backend.app.services.silage_scoring import evaluate_silage_screening
from backend.app.services.visual_mould_service import visual_mould_service

client = TestClient(app)


def generate_test_image(size=(224, 224), color=(140, 160, 120)) -> bytes:
    """Generate in-memory RGB JPEG image bytes with texture variance."""
    import numpy as np
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[:, :, 0] = np.clip(color[0] + np.random.randint(-15, 15, size), 0, 255)
    arr[:, :, 1] = np.clip(color[1] + np.random.randint(-15, 15, size), 0, 255)
    arr[:, :, 2] = np.clip(color[2] + np.random.randint(-15, 15, size), 0, 255)
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    for _ in range(30):
        x1, y1 = np.random.randint(0, size[0] - 10, 2)
        d.line([x1, y1, x1 + 15, y1 + 15], fill=(max(0, color[0]-25), max(0, color[1]-25), max(0, color[2]-25)), width=2)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ==============================================================================
# 1. Feed Reference Lookup
# ==============================================================================

def test_01_feed_reference_lookup():
    """Verify POST /api/v1/feed/reference returns valid ICAR-NIANP reference values."""
    payload = {"feed_name": "Maize", "quantity_kg": 1.0}
    res = client.post("/api/v1/feed/reference", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["basis"] == "reference"
    assert "Maize" in data["matched_feed_name"]
    assert data["per_kg"]["dry_matter_g"] == 880.0
    assert data["per_kg"]["crude_protein_g"] == 83.6
    assert data["per_kg"]["energy_mj"] is not None
    assert "ICAR-NIANP" in data["source"]
    assert "disclaimer" in data


# ==============================================================================
# 2. Quantity Calculation
# ==============================================================================

def test_02_quantity_calculation():
    """Verify total nutrient calculation: total = per_kg * quantity_kg."""
    qty = 5.0
    payload = {"feed_name": "Maize", "quantity_kg": qty}
    res = client.post("/api/v1/feed/reference", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["quantity_kg"] == qty

    per_kg = data["per_kg"]
    total = data["total_for_quantity"]

    assert pytest.approx(total["dry_matter_g"], rel=1e-2) == per_kg["dry_matter_g"] * qty
    assert pytest.approx(total["crude_protein_g"], rel=1e-2) == per_kg["crude_protein_g"] * qty
    assert pytest.approx(total["crude_fibre_g"], rel=1e-2) == per_kg["crude_fibre_g"] * qty
    assert pytest.approx(total["ndf_g"], rel=1e-2) == per_kg["ndf_g"] * qty
    assert pytest.approx(total["adf_g"], rel=1e-2) == per_kg["adf_g"] * qty
    assert pytest.approx(total["energy_mj"], rel=1e-2) == per_kg["energy_mj"] * qty


# ==============================================================================
# 3. Missing / Unknown Feed Handling
# ==============================================================================

def test_03_missing_feed_handling():
    """Verify 404 response with helpful suggestions when an uncatalogued feed is requested."""
    payload = {"feed_name": "NonExistentUnknownFeed_12345", "quantity_kg": 5.0}
    res = client.post("/api/v1/feed/reference", json=payload)
    assert res.status_code == 404
    assert "not found in authoritative reference database" in res.json()["detail"]


# ==============================================================================
# 4. Feed Nutrition ML Model
# ==============================================================================

def test_04_feed_nutrition_ml_model():
    """Verify existing POST /api/v1/predict/feed-nutrition continues working with 7 targets."""
    payload = {
        "Feed-category": "Forages",
        "Detailed-feed-category-INRA2018": "Maize silage",
        "Dry-matter-(g/kg)": 350.0,
        "Organic-matter-(g/kg-DM)-": 920.0,
        "Ash-(g/kg-DM)": 80.0,
        "Crude-fibre-(g/kg-DM)": 220.0,
        "NDF-(g/kg-DM)": 450.0,
        "ADF-(g/kg-DM)": 260.0,
        "Starch-(g/kg-DM)": 280.0
    }
    res = client.post("/api/v1/predict/feed-nutrition", json=payload)
    assert res.status_code == 200
    data = res.json()
    preds = data["predictions"]
    assert len(preds) == 7
    for key in ["crude_protein", "dry_matter", "crude_fibre", "ndf", "adf", "adl", "starch"]:
        assert key in preds
        assert preds[key]["predicted_value"] >= 0.0


# ==============================================================================
# 5. Unit Conversion
# ==============================================================================

def test_05_unit_conversion():
    """Verify correct unit conversion: percentage = g/kg / 10."""
    assert g_per_kg_to_percentage(98.17) == 9.817
    assert g_per_kg_to_percentage(350.0) == 35.0
    assert g_per_kg_to_percentage(0.0) == 0.0

    payload = {
        "Feed-category": "Forages",
        "Detailed-feed-category-INRA2018": "Maize silage",
        "Dry-matter-(g/kg)": 350.0,
        "Organic-matter-(g/kg-DM)-": 920.0,
        "Ash-(g/kg-DM)": 80.0,
        "Crude-fibre-(g/kg-DM)": 220.0,
        "NDF-(g/kg-DM)": 450.0,
        "ADF-(g/kg-DM)": 260.0,
        "Starch-(g/kg-DM)": 280.0
    }
    res = client.post("/api/v1/predict/feed-nutrition", json=payload)
    assert res.status_code == 200
    data = res.json()
    cp_pred = data["predictions"]["crude_protein"]
    assert cp_pred["percentage_value"] is not None
    assert pytest.approx(cp_pred["percentage_value"], rel=1e-2) == cp_pred["predicted_value"] / 10.0


# ==============================================================================
# 6. Dynamic Feed Quality Score
# ==============================================================================

def test_06_dynamic_feed_quality_score():
    """Verify quality score dynamically changes with different proximal inputs (no static 58)."""
    # High protein concentrate
    score_high, tier_high, why_high, _ = calculate_feed_quality_score(
        feed_category="Concentrates",
        dry_matter_g_per_kg=900.0,
        crude_protein_g_per_kg_dm=440.0,  # 44% CP
        crude_fibre_g_per_kg_dm=60.0,
        ndf_g_per_kg_dm=200.0,
        adf_g_per_kg_dm=110.0,
        adl_g_per_kg_dm=28.0,
        starch_g_per_kg_dm=120.0
    )
    # Poor low-protein, high-lignin straw
    score_low, tier_low, why_low, _ = calculate_feed_quality_score(
        feed_category="Roughage",
        dry_matter_g_per_kg=900.0,
        crude_protein_g_per_kg_dm=35.0,  # 3.5% CP
        crude_fibre_g_per_kg_dm=380.0,
        ndf_g_per_kg_dm=740.0,
        adf_g_per_kg_dm=500.0,
        adl_g_per_kg_dm=75.0,
        starch_g_per_kg_dm=10.0
    )
    assert score_high != score_low
    assert score_high > score_low
    assert score_high >= 80.0
    assert score_low <= 60.0
    assert tier_high in ["EXCELLENT", "GOOD"]
    assert tier_low in ["FAIR", "POOR"]


# ==============================================================================
# 7. Silage Quality Model
# ==============================================================================

def test_07_silage_quality_classifier():
    """Verify POST /api/v1/predict/silage/quality outputs FAO class ('ea'/'la') with confidence."""
    payload = {
        "dm.f": 32.5, "ash.f": 6.8, "cp.f": 14.2, "ee.f": 2.5, "ndf.f": 48.0,
        "adf.f": 28.5, "lignin.f": 4.2, "wsc.f": 12.0, "starch.f": 22.0,
        "dm.s": 31.8, "ash.s": 7.1, "cp.s": 14.0, "ee.s": 2.8, "ndf.s": 46.5,
        "adf.s": 27.9, "lignin.s": 4.4, "starch.s": 21.0, "wsc.s": 3.5,
        "pH": 3.85, "ammonia.s": 6.5, "glucose.s": 1.2, "fructose.s": 0.8,
        "mannithol.s": 0.5, "ethanol.s": 1.1, "lactic.ac.s": 6.2, "acetic.ac.s": 1.8,
        "propionic.ac.s": 0.2, "butyric.ac.s": 0.05, "dm.loss": 4.5, "dm.ret": 95.5,
        "porosity": 0.42, "density.1": 220.0
    }
    res = client.post("/api/v1/predict/silage/quality", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["predicted_class"] in ["ea", "la"]
    assert 0.0 <= data["confidence"] <= 1.0


# ==============================================================================
# 8. Silage FQI & Dynamic Interpretation
# ==============================================================================

def test_08_silage_fqi_and_screening():
    """Verify POST /api/v1/predict/silage/comprehensive produces dynamic screening result."""
    payload = {
        "dm.f": 32.5, "ash.f": 6.8, "cp.f": 14.2, "ee.f": 2.5, "ndf.f": 48.0,
        "adf.f": 28.5, "lignin.f": 4.2, "wsc.f": 12.0, "starch.f": 22.0,
        "dm.s": 31.8, "ash.s": 7.1, "cp.s": 14.0, "ee.s": 2.8, "ndf.s": 46.5,
        "adf.s": 27.9, "lignin.s": 4.4, "starch.s": 21.0, "wsc.s": 3.5,
        "pH": 3.85, "ammonia.s": 6.5, "glucose.s": 1.2, "fructose.s": 0.8,
        "mannithol.s": 0.5, "ethanol.s": 1.1, "lactic.ac.s": 6.2, "acetic.ac.s": 1.8,
        "propionic.ac.s": 0.2, "butyric.ac.s": 0.05, "dm.loss": 4.5, "dm.ret": 95.5,
        "porosity": 0.42, "density.1": 220.0
    }
    res = client.post("/api/v1/predict/silage/comprehensive", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "screening_result" in data
    screening = data["screening_result"]
    assert screening["screening_status"] in ["GOOD", "CAUTION", "UNSAFE"]
    assert 0.0 <= screening["composite_quality_score"] <= 100.0
    assert len(screening["why"]) > 0
    assert len(screening["recommended_action"]) > 0
    assert "disclaimer" in screening


# ==============================================================================
# 9. Image Validation
# ==============================================================================

def test_09_image_validation():
    """Verify PIL decode and feature extraction on valid JPEG image."""
    img_bytes = generate_test_image(size=(224, 224))
    img = visual_mould_service.validate_and_load_image(img_bytes)
    assert img.size == (224, 224)
    indicators, discolouration, roughness, metrics = visual_mould_service.extract_visual_features(img)
    assert 0.0 <= discolouration <= 1.0
    assert 0.0 <= roughness <= 1.0
    assert isinstance(indicators.dark_or_mould_cluster_spots, bool)
    assert "clustered_white_pct" in metrics


# ==============================================================================
# 10. Mould Visual Screening (Method 1 - Rule-Based Image Analysis)
# ==============================================================================

def create_synthetic_maize_image() -> bytes:
    """Realistic photograph of maize grains with specular reflections, white germ tips, and shadows."""
    img = Image.new("RGB", (300, 300), color=(225, 170, 35))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    import numpy as np
    np.random.seed(123)
    for x in range(10, 285, 28):
        for y in range(10, 285, 28):
            draw.ellipse([x, y, x + 24, y + 24], fill=(235, 180, 40), outline=(135, 85, 15), width=2)
            # Specular light reflection on kernel curve
            draw.ellipse([x + 4, y + 4, x + 10, y + 10], fill=(255, 255, 250))
            # Pale tip at base of kernel
            draw.ellipse([x + 14, y + 14, x + 20, y + 20], fill=(245, 240, 225))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def create_synthetic_mouldy_image() -> bytes:
    """Feed sample with contiguous white-grey fungal hyphae mycelium and green mold patches."""
    img = Image.new("RGB", (300, 300), color=(220, 165, 40))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    # Dense white cottony mycelium blob covering multiple kernels
    draw.ellipse([70, 70, 180, 180], fill=(240, 245, 240), outline=(210, 220, 210), width=3)
    draw.ellipse([85, 85, 165, 165], fill=(250, 252, 250))
    # Green spore colony
    draw.ellipse([160, 150, 240, 230], fill=(45, 95, 65), outline=(35, 75, 50), width=2)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def create_synthetic_spoiled_image() -> bytes:
    """Severely decomposed feed with extensive black slimy rotten patches on grain background."""
    import numpy as np
    arr = np.zeros((300, 300, 3), dtype=np.uint8)
    arr[:, :, 0] = np.random.randint(160, 200, (300, 300))
    arr[:, :, 1] = np.random.randint(130, 170, (300, 300))
    arr[:, :, 2] = np.random.randint(50, 80, (300, 300))
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    # Massive contiguous dark rot & black slimy decomposition patches (>35% coverage)
    draw.rectangle([20, 20, 280, 200], fill=(15, 12, 10))
    draw.ellipse([50, 180, 260, 290], fill=(18, 14, 12))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_10_mould_visual_screening_normal_maize():
    """Verify normal Maize Grain image produces 'GOOD' with valid non-negative probabilities."""
    maize_bytes = create_synthetic_maize_image()
    res = client.post(
        "/api/v1/predict/feed-visual",
        files={"file": ("maize_grain.jpg", maize_bytes, "image/jpeg")}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["predicted_class"] == "GOOD"
    assert data["risk_level"] == "LOW"
    probs = data["probabilities"]
    # Verify strict non-negativity and valid distribution
    for cls_name, prob in probs.items():
        assert 0.0 <= prob <= 1.0, f"Probability for {cls_name} ({prob}) is out of [0, 1]!"
    assert pytest.approx(sum(probs.values()), rel=1e-3) == 1.0
    assert probs["GOOD"] > 0.90
    assert data["visual_indicators"]["dark_or_mould_cluster_spots"] is False
    assert data["visual_indicators"]["white_grey_hyphae_indicators"] is False
    assert data["visual_indicators"]["surface_discolouration_index"] < 0.10
    assert "disclaimer" in data


def test_10b_mould_visual_screening_mouldy_feed():
    """Verify mouldy feed image correctly triggers 'MOULD_RISK' with valid probabilities."""
    mouldy_bytes = create_synthetic_mouldy_image()
    res = client.post(
        "/api/v1/predict/feed-visual",
        files={"file": ("mouldy_grain.jpg", mouldy_bytes, "image/jpeg")}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["predicted_class"] == "MOULD_RISK"
    assert data["risk_level"] == "HIGH"
    probs = data["probabilities"]
    for cls_name, prob in probs.items():
        assert 0.0 <= prob <= 1.0
    assert pytest.approx(sum(probs.values()), rel=1e-3) == 1.0
    assert probs["MOULD_RISK"] > 0.85
    assert data["visual_indicators"]["dark_or_mould_cluster_spots"] is True


def test_10c_mould_visual_screening_spoiled_feed():
    """Verify decomposed rotten feed correctly triggers 'SPOILED' with valid probabilities."""
    spoiled_bytes = create_synthetic_spoiled_image()
    res = client.post(
        "/api/v1/predict/feed-visual",
        files={"file": ("spoiled_feed.jpg", spoiled_bytes, "image/jpeg")}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["predicted_class"] == "SPOILED"
    assert data["risk_level"] == "CRITICAL"
    probs = data["probabilities"]
    for cls_name, prob in probs.items():
        assert 0.0 <= prob <= 1.0
    assert pytest.approx(sum(probs.values()), rel=1e-3) == 1.0
    assert probs["SPOILED"] > 0.85


def test_10d_silage_visual_screening_valid_distribution():
    """Verify silage visual screening returns 4 valid non-negative probabilities summing to 1."""
    silage_bytes = generate_test_image(size=(224, 224), color=(105, 130, 55))
    res = client.post(
        "/api/v1/predict/silage-visual",
        files={"file": ("silage_face.jpg", silage_bytes, "image/jpeg")}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["predicted_class"] in ["GOOD", "MOULD_RISK", "SPOILED", "POOR_FERMENTATION"]
    probs = data["probabilities"]
    assert len(probs) == 4
    for cls_name, prob in probs.items():
        assert 0.0 <= prob <= 1.0, f"Probability for {cls_name} ({prob}) is out of [0, 1]!"
    assert pytest.approx(sum(probs.values()), rel=1e-3) == 1.0


# ==============================================================================
# 11. Invalid Image Rejection
# ==============================================================================

def test_11_invalid_image_rejection():
    """Verify 400 Bad Request on non-image file upload."""
    res = client.post(
        "/api/v1/predict/feed-visual",
        files={"file": ("document.txt", b"plain text content not an image", "text/plain")}
    )
    assert res.status_code == 400


# ==============================================================================
# 12. Empty Image Rejection
# ==============================================================================

def test_12_empty_image_rejection():
    """Verify 400 Bad Request on 0-byte image file."""
    res = client.post(
        "/api/v1/predict/feed-visual",
        files={"file": ("empty.jpg", b"", "image/jpeg")}
    )
    assert res.status_code == 400


# ==============================================================================
# 13. Combined Feed Analysis (Exact Match with Standalone Visual Screening)
# ==============================================================================

def test_13_combined_feed_analysis_normal_maize():
    """Verify POST /api/v1/analyze/feed accurately evaluates normal maize grain without false penalties."""
    maize_bytes = create_synthetic_maize_image()
    res = client.post(
        "/api/v1/analyze/feed",
        data={"feed_name": "Maize Grain", "quantity_kg": 5.0},
        files={"image": ("maize_sample.jpg", maize_bytes, "image/jpeg")}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["feed_name"] == "Maize Grain"
    assert data["status"] in ["EXCELLENT", "GOOD"]
    assert data["quality_score"] >= 75.0, f"Quality score ({data['quality_score']}) was penalized incorrectly!"

    # Verify visual screening matches standalone endpoint
    v_res = data["visual_screening"]
    assert v_res is not None
    assert v_res["predicted_class"] == "GOOD"
    assert v_res["risk_level"] == "LOW"
    assert v_res["probabilities"]["GOOD"] > 0.90
    assert v_res["visual_indicators"]["dark_or_mould_cluster_spots"] is False
    assert v_res["visual_indicators"]["white_grey_hyphae_indicators"] is False
    assert v_res["visual_indicators"]["surface_discolouration_index"] < 0.10

    # Risk analysis matches
    risk = data["risk_analysis"]
    assert risk["mould_risk"]["level"] == "LOW"
    assert risk["mould_risk"]["basis"] == "RULE_BASED_IMAGE_ANALYSIS"


def test_13b_combined_feed_analysis_mouldy_feed():
    """Verify POST /api/v1/analyze/feed correctly detects mould and penalizes quality score."""
    mouldy_bytes = create_synthetic_mouldy_image()
    res = client.post(
        "/api/v1/analyze/feed",
        data={"feed_name": "Maize Grain", "quantity_kg": 5.0},
        files={"image": ("mouldy_maize.jpg", mouldy_bytes, "image/jpeg")}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["visual_screening"]["predicted_class"] == "MOULD_RISK"
    assert data["visual_screening"]["risk_level"] == "HIGH"
    assert data["risk_analysis"]["mould_risk"]["level"] == "HIGH"
    assert data["quality_score"] <= 60.0  # Properly penalized
    assert data["status"] in ["EXCELLENT", "GOOD", "FAIR", "POOR"]
    assert data["nutrition_reference"] is not None
    assert data["visual_screening"] is not None
    assert data["risk_analysis"] is not None
    assert len(data["why"]) > 0
    assert len(data["recommended_action"]) > 0


# ==============================================================================
# 14. Combined Silage Analysis
# ==============================================================================

def test_14_combined_silage_analysis():
    """Verify POST /api/v1/analyze/silage consolidates ML, acids, vision, and risk."""
    img_bytes = generate_test_image(size=(224, 224))
    form_data = {
        "pH": "3.85",
        "dm_s": "32.0",
        "cp_s": "14.2",
        "lactic_ac_s": "6.5",
        "acetic_ac_s": "1.8",
        "butyric_ac_s": "0.04",
        "ammonia_s": "6.0"
    }
    res = client.post(
        "/api/v1/analyze/silage",
        data=form_data,
        files={"image": ("silage.jpg", img_bytes, "image/jpeg")}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["status"] in ["GOOD", "CAUTION", "UNSAFE"]
    assert data["fermentation_ml"] is not None
    assert data["visual_screening"] is not None
    assert data["risk_analysis"] is not None
    assert data["fermentation_metrics"]["pH"] == 3.85


# ==============================================================================
# 15. Chat Integration
# ==============================================================================

def test_15_chat_feed_and_silage_context():
    """Verify chat endpoint accurately responds to feed nutrition queries."""
    res = client.post(
        "/api/v1/chat",
        json={"message": "What is the nutrition value and protein in 5 kg maize grain?", "language": "en"}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert len(data["reply"]) > 0
    assert "Dry Matter" in data["reply"] or "Crude Protein" in data["reply"] or "feed" in data["intent"]


# ==============================================================================
# 16. Error Handling & Edge Cases
# ==============================================================================

def test_16_error_handling():
    """Verify structured error handling on invalid requests."""
    # Negative quantity in reference lookup
    res_neg = client.post("/api/v1/feed/reference", json={"feed_name": "Maize", "quantity_kg": -2.0})
    assert res_neg.status_code == 422

    # Excessive quantity
    res_large = client.post("/api/v1/feed/reference", json={"feed_name": "Maize", "quantity_kg": 99999.0})
    assert res_large.status_code == 422

    # Offline catalog list
    res_all = client.get("/api/v1/feed/reference/all")
    assert res_all.status_code == 200
    assert res_all.json()["total_feeds"] >= 20

    # Rules endpoint
    res_rules = client.get("/api/v1/feed/reference/rules")
    assert res_rules.status_code == 200
    assert "scoring_tiers" in res_rules.json()


# ==============================================================================
# 17-21. Domain Validation & Non-Feed Image Rejection Tests (Cases A, B, C, D, E)
# ==============================================================================

def create_test_human_image() -> bytes:
    img = Image.new("RGB", (224, 224), color=(240, 235, 230))
    d = ImageDraw.Draw(img)
    d.ellipse([60, 30, 164, 130], fill=(225, 175, 140)) # face
    d.ellipse([70, 20, 154, 65], fill=(30, 20, 15))     # hair
    d.rectangle([40, 130, 184, 224], fill=(35, 75, 150)) # shirt
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def create_test_laptop_image() -> bytes:
    img = Image.new("RGB", (224, 224), color=(180, 180, 180))
    d = ImageDraw.Draw(img)
    d.rectangle([30, 30, 194, 150], fill=(30, 30, 30))
    d.rectangle([40, 40, 184, 140], fill=(70, 130, 180))
    d.rectangle([100, 150, 124, 180], fill=(80, 80, 80))
    d.rectangle([70, 180, 154, 195], fill=(50, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def create_test_cattle_image() -> bytes:
    try:
        real_img = Image.open("datasets/extracted/cattle_breeds/cattle/Amritmahal/Amritmahal_1.JPG")
        buf = io.BytesIO()
        real_img.save(buf, format="JPEG")
        return buf.getvalue()
    except Exception:
        img = Image.new("RGB", (224, 224), color=(100, 130, 70))
        d = ImageDraw.Draw(img)
        d.ellipse([50, 80, 180, 160], fill=(40, 40, 40))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

def create_test_feed_grain_image() -> bytes:
    import numpy as np
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    np.random.seed(42)
    arr[:, :, 0] = np.random.randint(180, 225, (224, 224))
    arr[:, :, 1] = np.random.randint(150, 195, (224, 224))
    arr[:, :, 2] = np.random.randint(50, 95, (224, 224))
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    for _ in range(40):
        x1, y1 = np.random.randint(0, 220, 2)
        d.line([x1, y1, x1 + np.random.randint(-10, 10), y1 + np.random.randint(10, 30)], fill=(160, 130, 40), width=2)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def create_test_silage_forage_image() -> bytes:
    import numpy as np
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    np.random.seed(123)
    arr[:, :, 0] = np.random.randint(110, 150, (224, 224))
    arr[:, :, 1] = np.random.randint(130, 170, (224, 224))
    arr[:, :, 2] = np.random.randint(45, 80, (224, 224))
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    for _ in range(50):
        x1, y1 = np.random.randint(0, 220, 2)
        d.line([x1, y1, x1 + np.random.randint(-10, 20), y1 + np.random.randint(5, 25)], fill=(90, 110, 35), width=2)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_17_human_image_rejection_case_a():
    """Case A: Human image must return NOT_FEED_OR_SILAGE, INVALID_IMAGE and never classify as GOOD/MOULD_RISK/SPOILED."""
    human_bytes = create_test_human_image()
    # Feed endpoint test
    res = client.post(
        "/api/v1/predict/feed-visual",
        files={"file": ("human_portrait.jpg", human_bytes, "image/jpeg")}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is False
    assert data["error_type"] == "INVALID_IMAGE"
    assert data["classification"] == "NOT_FEED_OR_SILAGE"
    assert data["predicted_class"] is None
    assert data["confidence"] == 0.0
    assert "person" in data["message"].lower() or "human" in data["message"].lower() or "feed" in data["message"].lower()

    # Silage endpoint test
    res_silage = client.post(
        "/api/v1/predict/silage-visual",
        files={"file": ("human_portrait.jpg", human_bytes, "image/jpeg")}
    )
    assert res_silage.status_code == 200
    sdata = res_silage.json()
    assert sdata["success"] is False
    assert sdata["error_type"] == "INVALID_IMAGE"
    assert sdata["classification"] == "NOT_FEED_OR_SILAGE"
    assert sdata["predicted_class"] is None


def test_18_random_non_feed_image_rejection_case_b():
    """Case B: Random non-feed object (e.g. laptop/screen) must return NOT_FEED_OR_SILAGE."""
    laptop_bytes = create_test_laptop_image()
    res = client.post(
        "/api/v1/predict/feed-visual",
        files={"file": ("laptop.jpg", laptop_bytes, "image/jpeg")}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is False
    assert data["error_type"] == "INVALID_IMAGE"
    assert data["classification"] == "NOT_FEED_OR_SILAGE"
    assert data["predicted_class"] is None


def test_19_cattle_only_photo_rejection_case_c():
    """Case C: Cattle-only photo without feed must return NOT_FEED_OR_SILAGE."""
    cattle_bytes = create_test_cattle_image()
    res = client.post(
        "/api/v1/predict/feed-visual",
        files={"file": ("cattle.jpg", cattle_bytes, "image/jpeg")}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is False
    assert data["error_type"] == "INVALID_IMAGE"
    assert data["classification"] == "NOT_FEED_OR_SILAGE"
    assert data["predicted_class"] is None


def test_20_valid_feed_image_passes_case_d():
    """Case D: Valid feed image continues to quality analysis with valid predicted_class."""
    feed_bytes = create_test_feed_grain_image()
    res = client.post(
        "/api/v1/predict/feed-visual",
        files={"file": ("feed_grain.jpg", feed_bytes, "image/jpeg")}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["classification"] == "FEED_SAMPLE"
    assert data["predicted_class"] in ["GOOD", "MOULD_RISK", "SPOILED"]
    assert data["confidence"] > 0.0


def test_21_valid_silage_image_passes_case_e():
    """Case E: Valid silage image continues to silage quality analysis with valid predicted_class."""
    silage_bytes = create_test_silage_forage_image()
    res = client.post(
        "/api/v1/predict/silage-visual",
        files={"file": ("silage_forage.jpg", silage_bytes, "image/jpeg")}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["classification"] == "SILAGE_SAMPLE"
    assert data["predicted_class"] in ["GOOD", "MOULD_RISK", "SPOILED", "POOR_FERMENTATION"]
    assert data["confidence"] > 0.0

def create_feed_held_by_hand_image() -> bytes:
    """Close up hay/straw feed with farmer hand/fingers visible at corner."""
    import numpy as np
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    np.random.seed(42)
    arr[:, :, 0] = np.random.randint(180, 225, (224, 224))
    arr[:, :, 1] = np.random.randint(150, 195, (224, 224))
    arr[:, :, 2] = np.random.randint(50, 95, (224, 224))
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    for i in range(0, 224, 6):
        d.line([(i, 0), (i + 15, 224)], fill=(220, 190, 70), width=2)
        d.line([(0, i), (224, i + 10)], fill=(150, 110, 30), width=1)
    d.polygon([(0, 150), (90, 150), (100, 224), (0, 224)], fill=(215, 165, 130))
    d.ellipse([(20, 140), (60, 200)], fill=(220, 170, 135))
    d.ellipse([(55, 145), (95, 205)], fill=(210, 160, 125))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def create_silage_held_by_hand_image() -> bytes:
    """Close up chopped silage with farmer hand/fingers visible at bottom."""
    import numpy as np
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    np.random.seed(123)
    arr[:, :, 0] = np.random.randint(110, 150, (224, 224))
    arr[:, :, 1] = np.random.randint(130, 170, (224, 224))
    arr[:, :, 2] = np.random.randint(45, 80, (224, 224))
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    for x in range(5, 220, 15):
        for y in range(5, 220, 15):
            d.rectangle([x, y, x + 10, y + 6], fill=(135, 160, 50), outline=(90, 110, 35))
    d.polygon([(140, 160), (224, 150), (224, 224), (130, 224)], fill=(210, 160, 125))
    d.ellipse([(145, 145), (185, 210)], fill=(215, 165, 130))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

# ==============================================================================
# 17-24. Domain Validation Test Suite (Cases A through H)
# ==============================================================================

def test_17_human_image_rejection_case_a():
    """Case A: Human portrait/selfie must return NOT_FEED_OR_SILAGE and INVALID_IMAGE."""
    human_bytes = create_test_human_image()
    res = client.post(
        "/api/v1/predict/feed-visual",
        files={"file": ("human_selfie.jpg", human_bytes, "image/jpeg")}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is False
    assert data["error_type"] == "INVALID_IMAGE"
    assert data["classification"] == "NOT_FEED_OR_SILAGE"
    assert data["predicted_class"] is None
    assert "person or human" in data["message"].lower() or "feed" in data["message"].lower()


def test_18_random_non_feed_image_rejection_case_b():
    """Case B: Random non-feed object (e.g. laptop/screen) must return NOT_FEED_OR_SILAGE."""
    laptop_bytes = create_test_laptop_image()
    res = client.post(
        "/api/v1/predict/feed-visual",
        files={"file": ("laptop.jpg", laptop_bytes, "image/jpeg")}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is False
    assert data["error_type"] == "INVALID_IMAGE"
    assert data["classification"] == "NOT_FEED_OR_SILAGE"
    assert data["predicted_class"] is None


def test_19_cattle_only_photo_rejection_case_c():
    """Case C: Cattle-only portrait photo without feed must return NOT_FEED_OR_SILAGE."""
    cattle_bytes = create_test_cattle_image()
    res = client.post(
        "/api/v1/predict/feed-visual",
        files={"file": ("cattle.jpg", cattle_bytes, "image/jpeg")}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is False
    assert data["error_type"] == "INVALID_IMAGE"
    assert data["classification"] == "NOT_FEED_OR_SILAGE"
    assert data["predicted_class"] is None


def test_20_valid_feed_image_passes_case_d():
    """Case D: Valid feed image without hand continues to quality analysis with valid predicted_class."""
    feed_bytes = create_test_feed_grain_image()
    res = client.post(
        "/api/v1/predict/feed-visual",
        files={"file": ("feed_grain.jpg", feed_bytes, "image/jpeg")}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["classification"] == "FEED_SAMPLE"
    assert data["predicted_class"] in ["GOOD", "MOULD_RISK", "SPOILED"]
    assert 0.0 <= data["confidence"] <= 1.0


def test_21_valid_feed_held_by_hand_passes_case_e():
    """Case E: Valid feed held by human hand MUST be accepted and evaluated for quality."""
    feed_hand_bytes = create_feed_held_by_hand_image()
    res = client.post(
        "/api/v1/predict/feed-visual",
        files={"file": ("feed_in_hand.jpg", feed_hand_bytes, "image/jpeg")}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["classification"] == "FEED_SAMPLE"
    assert data["predicted_class"] in ["GOOD", "MOULD_RISK", "SPOILED"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert "visual_indicators" in data


def test_22_valid_silage_image_passes_case_f():
    """Case F: Valid silage image without hand continues to silage quality analysis."""
    silage_bytes = create_test_silage_forage_image()
    res = client.post(
        "/api/v1/predict/silage-visual",
        files={"file": ("silage_forage.jpg", silage_bytes, "image/jpeg")}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["classification"] == "SILAGE_SAMPLE"
    assert data["predicted_class"] in ["GOOD", "MOULD_RISK", "SPOILED", "POOR_FERMENTATION"]
    assert 0.0 <= data["confidence"] <= 1.0


def test_23_valid_silage_held_by_hand_passes_case_g():
    """Case G: Valid silage held by human hand MUST be accepted and evaluated for quality."""
    silage_hand_bytes = create_silage_held_by_hand_image()
    res = client.post(
        "/api/v1/predict/silage-visual",
        files={"file": ("silage_in_hand.jpg", silage_hand_bytes, "image/jpeg")}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["classification"] == "SILAGE_SAMPLE"
    assert data["predicted_class"] in ["GOOD", "MOULD_RISK", "SPOILED", "POOR_FERMENTATION"]
    assert 0.0 <= data["confidence"] <= 1.0


def test_24_human_portrait_rejected_on_silage_case_h():
    """Case H: Human portrait uploaded to silage screening must be rejected as INVALID_IMAGE."""
    human_bytes = create_test_human_image()
    res = client.post(
        "/api/v1/predict/silage-visual",
        files={"file": ("human_portrait.jpg", human_bytes, "image/jpeg")}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is False
    assert data["error_type"] == "INVALID_IMAGE"
    assert data["classification"] == "NOT_FEED_OR_SILAGE"
    assert data["predicted_class"] is None
