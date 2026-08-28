"""
Comprehensive Smoke & Integration Test Suite for Dairy AI Assistant Backend
"""

import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from backend.main import app
from backend.config import settings
from backend.app.services.model_loader import model_loader


@pytest.fixture(scope="module")
def client():
    """Create TestClient with startup lifespan executed."""
    with TestClient(app) as test_client:
        yield test_client


def create_mock_image_bytes(size=(300, 300), color=(120, 150, 180)) -> bytes:
    """Generate in-memory RGB JPEG image bytes for vision endpoint testing."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ----------------------------------------------------------------------
# 1. Health & Status Tests (Lazy Loading Architecture)
# ----------------------------------------------------------------------

def test_health_endpoints(client: TestClient):
    """Verify root /health and /api/v1/health endpoints."""
    res_root = client.get("/health")
    assert res_root.status_code == 200
    data_root = res_root.json()
    assert data_root["status"] == "healthy"
    assert data_root["production_models_ready"] == 13
    assert isinstance(data_root["models_loaded"], int)
    assert data_root["models_loaded"] >= 0

    res_v1 = client.get("/api/v1/health")
    assert res_v1.status_code == 200
    assert res_v1.json()["status"] == "healthy"

    res_ping = client.get("/api/v1/ping")
    assert res_ping.status_code == 200
    assert res_ping.json()["ping"] == "pong"


def test_lazy_loading_lifecycle(client: TestClient):
    """Verify models are loaded on-demand and cached without duplicate loading."""
    # 1. Start with clean cache
    model_loader.clear_cache()
    assert model_loader.get_loaded_models_count() == 0

    # 2. Health check reports 0 loaded models without triggering model load
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["models_loaded"] == 0

    # 3. Call tabular prediction endpoint -> loads only milk_production pipeline
    payload = {
        "body_weight_kg": 450.0,
        "days_in_milk": 60,
        "lactation_number": 2,
        "temperature_celsius": 26.0,
        "humidity_percent": 65.0,
        "thi": 72.0,
        "species": "cow",
        "breed": "gir"
    }
    res_milk = client.post("/api/v1/predict/milk-production", json=payload)
    assert res_milk.status_code == 200
    assert model_loader.is_cached("milk_production") is True
    assert model_loader.get_loaded_models_count() == 1

    # 4. Repeated call reuses cached model without increasing count
    res_milk2 = client.post("/api/v1/predict/milk-production", json=payload)
    assert res_milk2.status_code == 200
    assert model_loader.get_loaded_models_count() == 1

    # 5. Call disease endpoint -> lazy-loads disease model (count becomes 2)
    img_bytes = create_mock_image_bytes()
    files = {"file": ("disease.jpg", img_bytes, "image/jpeg")}
    res_disease = client.post("/api/v1/predict/disease", files=files)
    assert res_disease.status_code == 200
    assert model_loader.is_cached("cattle_disease") is True
    assert model_loader.get_loaded_models_count() == 2

    # 6. Unload a model explicitly to test memory reclamation
    unloaded = model_loader.unload_model("cattle_disease")
    assert unloaded is True
    assert model_loader.is_cached("cattle_disease") is False
    assert model_loader.get_loaded_models_count() == 1


def test_models_registry_status(client: TestClient):
    """Verify /api/v1/models listing and individual model status."""
    res = client.get("/api/v1/models")
    assert res.status_code == 200
    data = res.json()
    assert data["total_models"] == 15
    assert data["production_count"] == 13
    assert data["experimental_count"] == 2
    assert len(data["models"]) == 15

    # Check individual model
    res_disease = client.get("/api/v1/models/cattle_disease")
    assert res_disease.status_code == 200
    disease_data = res_disease.json()
    assert disease_data["key"] == "cattle_disease"
    assert disease_data["status"] == "production"
    assert disease_data["framework"] == "pytorch"
    assert disease_data["is_enabled"] is True
    assert isinstance(disease_data["is_cached_in_memory"], bool)


def test_model_not_found(client: TestClient):
    """Verify 404 response for unregistered model key."""
    res = client.get("/api/v1/models/non_existent_model")
    assert res.status_code == 404
    assert res.json()["error_type"] == "ModelNotFoundError"


# ----------------------------------------------------------------------
# 2. Vision Models: Disease & Breed
# ----------------------------------------------------------------------

def test_disease_prediction(client: TestClient):
    """Verify POST /api/v1/predict/disease with synthetic image."""
    img_bytes = create_mock_image_bytes(size=(300, 300))
    files = {"file": ("test_cow.jpg", img_bytes, "image/jpeg")}

    res = client.post("/api/v1/predict/disease", files=files)
    assert res.status_code == 200
    data = res.json()
    assert "predicted_class" in data
    assert data["predicted_class"] in ["FMD", "IBK", "LSD", "Normal"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["probabilities"]) == 4
    assert "FMD" in data["probabilities"]
    assert "Normal" in data["probabilities"]
    assert data["model_version"] == "efficientnet_b3"


def test_disease_real_cattle_image(client: TestClient):
    """Verify POST /api/v1/predict/disease with real dataset cattle lesion image."""
    from pathlib import Path
    img_path = Path("datasets/raw/cattle_disease/FMD/FMD1.jpg")
    if img_path.exists():
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        res = client.post("/api/v1/predict/disease", files={"file": ("FMD1.jpg", img_bytes, "image/jpeg")})
        assert res.status_code == 200
        data = res.json()
        assert data["predicted_class"] == "FMD"
        assert data["is_disease_detected"] is True
        assert data["disease_name_full"] == "Foot-and-Mouth Disease (Aphthovirus)"
        assert data["confidence"] > 0.5


def test_disease_invalid_file(client: TestClient):
    """Verify error handling on non-image upload."""
    files = {"file": ("test.txt", b"not an image", "text/plain")}
    res = client.post("/api/v1/predict/disease", files=files)
    assert res.status_code == 400
    assert res.json()["error_type"] == "ImageProcessingError"


def test_disease_empty_image(client: TestClient):
    """Verify error handling on empty image file."""
    files = {"file": ("empty.jpg", b"", "image/jpeg")}
    res = client.post("/api/v1/predict/disease", files=files)
    assert res.status_code == 400
    assert res.json()["error_type"] == "ImageProcessingError"
    assert "empty" in res.json()["message"].lower()


def test_disease_corrupt_image(client: TestClient):
    """Verify error handling on corrupted image payload."""
    files = {"file": ("corrupt.jpg", b"GIF89a_not_a_real_image_data_just_garbage", "image/jpeg")}
    res = client.post("/api/v1/predict/disease", files=files)
    assert res.status_code == 400
    assert res.json()["error_type"] == "ImageProcessingError"


def test_disease_missing_image(client: TestClient):
    """Verify error handling when file field is omitted."""
    res = client.post("/api/v1/predict/disease", data={})
    assert res.status_code == 422
    assert res.json()["error_type"] == "RequestValidationError"


def test_breed_prediction_default(client: TestClient):
    """Verify POST /api/v1/predict/breed with synthetic image and default threshold."""
    img_bytes = create_mock_image_bytes(size=(224, 224))
    files = {"file": ("test_bovine.jpg", img_bytes, "image/jpeg")}

    res = client.post("/api/v1/predict/breed", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["breed_status"] in ["identified", "uncertain"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert 0.0 <= data["confidence_percentage"] <= 100.0
    assert len(data["top_5_predictions"]) == 5
    assert data["total_classes_supported"] == 41
    assert data["model_architecture"] == "convnext_tiny"
    if data["breed_status"] == "identified":
        assert data["predicted_breed"] is not None
        assert data["recommendation"] is None
    else:
        assert data["predicted_breed"] is None
        assert data["recommendation"] == "Upload a clearer side-profile image with the full animal visible."


def test_breed_prediction_uncertain_below_threshold(client: TestClient):
    """Verify POST /api/v1/predict/breed when confidence < threshold returns uncertain payload."""
    img_bytes = create_mock_image_bytes(size=(224, 224))
    files = {"file": ("test_bovine.jpg", img_bytes, "image/jpeg")}

    # Force threshold higher than top prediction (0.999) to verify uncertainty handling
    res = client.post("/api/v1/predict/breed?confidence_threshold=0.999", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["breed_status"] == "uncertain"
    assert data["predicted_breed"] is None
    assert data["recommendation"] == "Upload a clearer side-profile image with the full animal visible."
    assert len(data["top_5_predictions"]) == 5
    assert 0.0 <= data["confidence"] <= 1.0
    assert 0.0 <= data["confidence_percentage"] <= 100.0
    assert data["total_classes_supported"] == 41
    assert data["model_architecture"] == "convnext_tiny"


def test_breed_prediction_confident_above_threshold(client: TestClient):
    """Verify POST /api/v1/predict/breed when confidence >= threshold returns identified breed."""
    img_bytes = create_mock_image_bytes(size=(224, 224))
    files = {"file": ("test_bovine.jpg", img_bytes, "image/jpeg")}

    # Force threshold lower than top prediction (0.0) to verify normal identified return
    res = client.post("/api/v1/predict/breed?confidence_threshold=0.0", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["breed_status"] == "identified"
    assert data["predicted_breed"] is not None
    assert isinstance(data["predicted_breed"], str)
    assert len(data["predicted_breed"]) > 0
    assert data["recommendation"] is None
    assert len(data["top_5_predictions"]) == 5
    assert data["total_classes_supported"] == 41


def test_breed_prediction_invalid_file(client: TestClient):
    """Verify error handling on non-image upload for breed endpoint."""
    files = {"file": ("test.txt", b"not an image", "text/plain")}
    res = client.post("/api/v1/predict/breed", files=files)
    assert res.status_code == 400
    assert res.json()["error_type"] == "ImageProcessingError"


# ----------------------------------------------------------------------
# 3. Tabular Models: Milk Production, Silage, Feed Nutrition
# ----------------------------------------------------------------------

def test_milk_production_prediction(client: TestClient):
    """Verify POST /api/v1/predict/milk-production."""
    payload = {
        "Age_Months": 48.0,
        "Weight_kg": 460.0,
        "Parity": 2,
        "Days_in_Milk": 95.0,
        "Previous_Week_Avg_Yield": 20.5,
        "Body_Condition_Score": 3.2,
        "Milking_Interval_hrs": 12.0,
        "Feed_Quantity_kg": 24.0,
        "Feeding_Frequency": 2.0,
        "Water_Intake_L": 80.0,
        "Grazing_Duration_hrs": 3.5,
        "Walking_Distance_km": 1.2,
        "Rumination_Time_hrs": 8.5,
        "Resting_Hours": 10.0,
        "Ambient_Temperature_C": 23.0,
        "Humidity_percent": 65.0,
        "Housing_Score": 4.0,
        "FMD_Vaccine": 1,
        "Brucellosis_Vaccine": 1,
        "HS_Vaccine": 1,
        "BQ_Vaccine": 1,
        "Anthrax_Vaccine": 0,
        "IBR_Vaccine": 0,
        "BVD_Vaccine": 0,
        "Rabies_Vaccine": 0,
        "Cattle_ID": "COW_TEST",
        "Breed": "Holstein_Friesian",
        "Region": "Northern",
        "Country": "India",
        "Climate_Zone": "Tropical",
        "Management_System": "Intensive",
        "Lactation_Stage": "Mid",
        "Feed_Type": "TMR",
        "Season": "Monsoon",
        "Date": "2026-08-26",
        "Farm_ID": "FARM_TEST"
    }

    res = client.post("/api/v1/predict/milk-production", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "predicted_milk_yield_litres" in data
    assert data["predicted_milk_yield_litres"] > 0.0
    assert data["target_unit"] == "Litres / Day"
    assert data["model_r2_score"] > 0.90


def test_silage_quality_and_fqi(client: TestClient):
    """Verify Silage classification, FQI regression, and comprehensive endpoint."""
    silage_data = {
        "dm.f": 33.0,
        "ash.f": 6.5,
        "cp.f": 14.5,
        "ee.f": 2.6,
        "ndf.f": 47.0,
        "adf.f": 28.0,
        "lignin.f": 4.1,
        "wsc.f": 12.5,
        "starch.f": 23.0,
        "dm.s": 32.2,
        "ash.s": 6.9,
        "cp.s": 14.2,
        "ee.s": 2.9,
        "ndf.s": 45.8,
        "adf.s": 27.5,
        "lignin.s": 4.3,
        "starch.s": 22.0,
        "wsc.s": 3.8,
        "pH": 3.80,
        "ammonia.s": 6.2,
        "glucose.s": 1.3,
        "fructose.s": 0.9,
        "mannithol.s": 0.4,
        "ethanol.s": 1.0,
        "lactic.ac.s": 6.5,
        "acetic.ac.s": 1.7,
        "propionic.ac.s": 0.2,
        "butyric.ac.s": 0.04,
        "dm.loss": 4.2,
        "dm.ret": 95.8,
        "porosity": 0.41,
        "density.1": 225.0
    }

    # 1. Quality Classification
    res_cls = client.post("/api/v1/predict/silage/quality", json=silage_data)
    assert res_cls.status_code == 200
    data_cls = res_cls.json()
    assert data_cls["predicted_class"] in ["ea", "la"]
    assert 0.0 <= data_cls["confidence"] <= 1.0
    assert "ea" in data_cls["probabilities"]

    # 2. FQI Regression
    res_reg = client.post("/api/v1/predict/silage/fqi", json=silage_data)
    assert res_reg.status_code == 200
    data_reg = res_reg.json()
    assert "predicted_fqi" in data_reg
    assert data_reg["predicted_fqi"] > 0

    # 3. Comprehensive
    res_comp = client.post("/api/v1/predict/silage/comprehensive", json=silage_data)
    assert res_comp.status_code == 200
    data_comp = res_comp.json()
    assert "quality_classification" in data_comp
    assert "fermentation_quality_index" in data_comp


def test_feed_nutrition_predictions(client: TestClient):
    """Verify multi-target feed nutrition analysis."""
    payload = {
        "Feed-category": "Forages",
        "Detailed-feed-category-INRA2018": "Maize silage",
        "Dry-matter-(g/kg)": 340.0,
        "Organic-matter-(g/kg-DM)-": 930.0,
        "Ash-(g/kg-DM)": 70.0,
        "Crude-fibre-(g/kg-DM)": 210.0,
        "NDF-(g/kg-DM)": 440.0,
        "ADF-(g/kg-DM)": 250.0,
        "Starch-(g/kg-DM)": 290.0
    }

    # All 7 targets
    res = client.post("/api/v1/predict/feed-nutrition", json=payload)
    assert res.status_code == 200
    data = res.json()
    preds = data["predictions"]
    assert len(preds) == 7
    for target_key in ["crude_protein", "dry_matter", "crude_fibre", "ndf", "adf", "adl", "starch"]:
        assert target_key in preds
        assert preds[target_key]["predicted_value"] >= 0.0
        assert len(preds[target_key]["unit"]) > 0

    # Single target
    res_single = client.post("/api/v1/predict/feed-nutrition/crude_protein", json=payload)
    assert res_single.status_code == 200
    assert res_single.json()["target_name"] == "Crude Protein"


# ----------------------------------------------------------------------
# 4. NIR Spectroscopy: Milk Quality & Validation
# ----------------------------------------------------------------------

def test_milk_quality_fat_nir(client: TestClient):
    """Verify NIR milk fat prediction with 1024 spectral channels."""
    # Synthetic spectral signature (exactly 1024 Trans_* channels)
    mock_spectra = [float(0.5 + 0.1 * (i % 10)) for i in range(1024)]
    payload = {
        "sample_id": "MOCK_NIR_001",
        "spectra": mock_spectra,
        "temperature_c": 20.0
    }

    res = client.post("/api/v1/predict/milk-quality/fat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "predicted_fat_percentage" in data
    assert data["predicted_fat_percentage"] >= 0.0
    assert data["spectral_channels_used"] == 1024
    assert data["pca_components"] == 95
    assert data["model_r2_score"] > 0.85


def test_milk_quality_invalid_spectrum(client: TestClient):
    """Verify validation failure if spectrum length != 1024 (e.g. 3 or legacy 1032)."""
    # 3 features
    payload_short = {
        "sample_id": "INVALID_SHORT",
        "spectra": [0.5, 0.6, 0.7]
    }
    res_short = client.post("/api/v1/predict/milk-quality/fat", json=payload_short)
    assert res_short.status_code == 422

    # 1032 features (legacy feature count with metadata leakage should now be rejected)
    payload_legacy = {
        "sample_id": "INVALID_LEGACY_1032",
        "spectra": [0.5] * 1032
    }
    res_legacy = client.post("/api/v1/predict/milk-quality/fat", json=payload_legacy)
    assert res_legacy.status_code == 422


# ----------------------------------------------------------------------
# 5. Sensor & Lab Endpoints & Experimental Model Gates
# ----------------------------------------------------------------------

def test_contamination_screening(client: TestClient):
    """Verify sensor contamination screening without data fabrication."""
    # Normal telemetry
    payload_normal = {
        "electrical_conductivity_ms_cm": 5.2,
        "freezing_point_c": -0.535,
        "milk_ph": 6.65,
        "somatic_cell_count_raw": 150.0
    }
    res_normal = client.post("/api/v1/sensor-lab/contamination-screen", json=payload_normal)
    assert res_normal.status_code == 200
    data_normal = res_normal.json()
    assert data_normal["is_sensor_data_valid"] is True
    assert data_normal["water_adulteration_suspected"] is False
    assert data_normal["subclinical_mastitis_risk"] == "Low / Normal"
    assert data_normal["acidity_anomaly"] is False
    assert data_normal["lab_verification_required"] is False

    # Adulterated & Anomaly telemetry
    payload_adulterated = {
        "freezing_point_c": -0.420,  # Extraneous water indicator
        "electrical_conductivity_ms_cm": 7.2,  # Elevated EC -> Mastitis
        "milk_ph": 7.1  # Alkaline -> Abnormal
    }
    res_adulterated = client.post("/api/v1/sensor-lab/contamination-screen", json=payload_adulterated)
    assert res_adulterated.status_code == 200
    data_adulterated = res_adulterated.json()
    assert data_adulterated["water_adulteration_suspected"] is True
    assert data_adulterated["subclinical_mastitis_risk"] == "High"
    assert data_adulterated["acidity_anomaly"] is True
    assert data_adulterated["lab_verification_required"] is True


def test_mycotoxin_experimental_disabled_by_default(client: TestClient):
    """Verify experimental mycotoxin endpoint returns 403 when disabled."""
    payload = {
        "protein_percent": 9.5,
        "fat_percent": 3.8,
        "moisture_percent": 14.0,
        "fiber_percent": 2.2,
        "starch_percent": 72.0,
        "ash_ai_percent": 1.5,
        "harvest_year": 2024,
        "sample_type": "Corn Grain",
        "sample_location": "Silo 1"
    }
    res = client.post("/api/v1/sensor-lab/mycotoxin-don", json=payload)
    if not settings.ENABLE_EXPERIMENTAL_MODELS:
        assert res.status_code == 403
        assert res.json()["error_type"] == "ModelDisabledError"
    else:
        assert res.status_code == 200


def test_urea_silica_screen(client: TestClient):
    """Verify urea/silica screening ingestion."""
    payload = {
        "sample_matrix": "Feed",
        "wet_chemistry_value": 18.5,
        "spectral_absorption_peaks": {"1450nm": 0.45, "1940nm": 0.82}
    }
    res = client.post("/api/v1/sensor-lab/urea-silica-screen", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["lab_data_provided"] is True
    assert "Feed" in data["sample_matrix"]


# ----------------------------------------------------------------------
# 6. Chat Endpoint Smoke Tests
# ----------------------------------------------------------------------

def test_chat_endpoint_smoke_english(client: TestClient):
    """Verify POST /api/v1/chat returns 200 with AI response."""
    payload = {
        "message": "What is the recommended feed ration for high milk yield cows?",
        "language": "en",
        "session_id": "test_smoke_chat_sess_001"
    }
    res = client.post("/api/v1/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert len(data["reply"]) > 0
    assert data["session_id"] == "test_smoke_chat_sess_001"


def test_chat_endpoint_smoke_multiturn(client: TestClient):
    """Verify multi-turn follow up retains conversation session."""
    payload1 = {
        "message": "My cow produces 18 litres of milk daily.",
        "session_id": "test_smoke_chat_multiturn_002"
    }
    res1 = client.post("/api/v1/chat", json=payload1)
    assert res1.status_code == 200

    payload2 = {
        "message": "What should the green and dry fodder ratio be?",
        "session_id": "test_smoke_chat_multiturn_002"
    }
    res2 = client.post("/api/v1/chat", json=payload2)
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["success"] is True
    assert len(data2["reply"]) > 0
    assert data2["session_id"] == "test_smoke_chat_multiturn_002"

