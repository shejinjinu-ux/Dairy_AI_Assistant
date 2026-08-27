"""
Standalone Smoke Test Runner for Dairy AI Assistant FastAPI Backend
"""

import io
import sys
import json
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient

# Ensure project root is in python path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.main import app
from backend.config import settings


def generate_mock_image(size=(300, 300), color=(140, 160, 200)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def run_smoke_tests():
    print("=" * 80)
    print("DAIRY AI ASSISTANT - FASTAPI BACKEND SMOKE TESTS")
    print("=" * 80)
    print(f"Project Root: {settings.PROJECT_ROOT}")
    print(f"Device:       {settings.torch_device}")
    print(f"Experimental: {settings.ENABLE_EXPERIMENTAL_MODELS}")
    print("-" * 80)

    client = TestClient(app)
    passed = 0
    failed = 0

    def check(name: str, condition: bool, details: str = ""):
        nonlocal passed, failed
        if condition:
            print(f"  [PASS] {name} {details}")
            passed += 1
        else:
            print(f"  [FAIL] {name} {details}")
            failed += 1

    # 1. Health Checks
    print("\n1. Testing Health & Status Endpoints...")
    res = client.get("/health")
    check("GET /health", res.status_code == 200, f"(Status: {res.status_code}, Ready Models: {res.json().get('production_models_ready')})")

    res = client.get("/api/v1/health")
    check("GET /api/v1/health", res.status_code == 200, f"(Status: {res.status_code})")

    res = client.get("/api/v1/ping")
    check("GET /api/v1/ping", res.status_code == 200 and res.json().get("ping") == "pong")

    # 2. Model Registry Status
    print("\n2. Testing Model Registry Status...")
    res = client.get("/api/v1/models")
    data = res.json()
    check(
        "GET /api/v1/models",
        res.status_code == 200 and data.get("total_models") == 15,
        f"(Total: {data.get('total_models')}, Prod: {data.get('production_count')}, Exp: {data.get('experimental_count')})"
    )

    res = client.get("/api/v1/models/cattle_disease")
    check("GET /api/v1/models/cattle_disease", res.status_code == 200 and res.json().get("framework") == "pytorch")

    res = client.get("/api/v1/models/milk_production")
    check("GET /api/v1/models/milk_production", res.status_code == 200 and res.json().get("framework") == "xgboost")

    # 3. Disease Vision Diagnosis
    print("\n3. Testing Disease Vision Diagnosis...")
    img300 = generate_mock_image(size=(300, 300))
    res = client.post("/api/v1/predict/disease", files={"file": ("cow.jpg", img300, "image/jpeg")})
    data = res.json()
    check(
        "POST /api/v1/predict/disease",
        res.status_code == 200 and "predicted_class" in data,
        f"(Prediction: {data.get('predicted_class')}, Confidence: {data.get('confidence_percentage')}%)"
    )

    # 4. Bovine Breed Classification & Confidence Thresholding
    print("\n4. Testing Indian Bovine Breed Classification...")
    img224 = generate_mock_image(size=(224, 224))
    res = client.post("/api/v1/predict/breed", files={"file": ("bovine.jpg", img224, "image/jpeg")})
    data = res.json()
    top5 = [p["breed"] for p in data.get("top_5_predictions", [])]
    check(
        "POST /api/v1/predict/breed (Default)",
        res.status_code == 200 and len(top5) == 5 and data.get("breed_status") in ["identified", "uncertain"],
        f"(Status: {data.get('breed_status')}, Breed: {data.get('predicted_breed')}, Conf: {data.get('confidence_percentage')}%)"
    )

    # 4b. Bovine Breed Threshold Verification (Forced Uncertain < 0.70 via threshold=0.999)
    res_uncertain = client.post(
        "/api/v1/predict/breed?confidence_threshold=0.999",
        files={"file": ("bovine.jpg", img224, "image/jpeg")}
    )
    data_unc = res_uncertain.json()
    check(
        "POST /api/v1/predict/breed (Uncertain < Threshold)",
        res_uncertain.status_code == 200
        and data_unc.get("breed_status") == "uncertain"
        and data_unc.get("predicted_breed") is None
        and data_unc.get("recommendation") == "Upload a clearer side-profile image with the full animal visible."
        and len(data_unc.get("top_5_predictions", [])) == 5,
        f"(Status: {data_unc.get('breed_status')}, Recommendation: '{data_unc.get('recommendation')[:30]}...')"
    )

    # 4c. Bovine Breed Threshold Verification (Forced Confident >= Threshold via threshold=0.0)
    res_confident = client.post(
        "/api/v1/predict/breed?confidence_threshold=0.0",
        files={"file": ("bovine.jpg", img224, "image/jpeg")}
    )
    data_conf = res_confident.json()
    check(
        "POST /api/v1/predict/breed (Confident >= Threshold)",
        res_confident.status_code == 200
        and data_conf.get("breed_status") == "identified"
        and data_conf.get("predicted_breed") is not None
        and data_conf.get("recommendation") is None,
        f"(Status: {data_conf.get('breed_status')}, Predicted Breed: {data_conf.get('predicted_breed')})"
    )

    # 5. Milk Production Estimation
    print("\n5. Testing Milk Production Yield Estimation...")
    milk_payload = {
        "Age_Months": 48.0, "Weight_kg": 450.0, "Parity": 2, "Days_in_Milk": 90.0,
        "Previous_Week_Avg_Yield": 18.5, "Body_Condition_Score": 3.0, "Milking_Interval_hrs": 12.0,
        "Feed_Quantity_kg": 22.0, "Feeding_Frequency": 2.0, "Water_Intake_L": 75.0,
        "Grazing_Duration_hrs": 4.0, "Walking_Distance_km": 1.5, "Rumination_Time_hrs": 8.0,
        "Resting_Hours": 10.0, "Ambient_Temperature_C": 24.0, "Humidity_percent": 60.0,
        "Housing_Score": 4.0, "FMD_Vaccine": 1, "Brucellosis_Vaccine": 1, "HS_Vaccine": 1,
        "BQ_Vaccine": 1, "Anthrax_Vaccine": 0, "IBR_Vaccine": 0, "BVD_Vaccine": 0, "Rabies_Vaccine": 0,
        "Cattle_ID": "COW_01", "Breed": "Holstein_Friesian", "Region": "Northern", "Country": "India",
        "Climate_Zone": "Tropical", "Management_System": "Intensive", "Lactation_Stage": "Mid",
        "Feed_Type": "TMR", "Season": "Monsoon", "Date": "2026-08-26", "Farm_ID": "FARM_01"
    }
    res = client.post("/api/v1/predict/milk-production", json=milk_payload)
    data = res.json()
    check(
        "POST /api/v1/predict/milk-production",
        res.status_code == 200 and data.get("predicted_milk_yield_litres") is not None,
        f"(Predicted Yield: {data.get('predicted_milk_yield_litres')} Litres/day)"
    )

    # 6. Silage Quality & FQI
    print("\n6. Testing Silage Quality & FQI Endpoints...")
    silage_payload = {
        "dm.f": 32.5, "ash.f": 6.8, "cp.f": 14.2, "ee.f": 2.5, "ndf.f": 48.0,
        "adf.f": 28.5, "lignin.f": 4.2, "wsc.f": 12.0, "starch.f": 22.0,
        "dm.s": 31.8, "ash.s": 7.1, "cp.s": 14.0, "ee.s": 2.8, "ndf.s": 46.5,
        "adf.s": 27.9, "lignin.s": 4.4, "starch.s": 21.0, "wsc.s": 3.5,
        "pH": 3.85, "ammonia.s": 6.5, "glucose.s": 1.2, "fructose.s": 0.8,
        "mannithol.s": 0.5, "ethanol.s": 1.1, "lactic.ac.s": 6.2, "acetic.ac.s": 1.8,
        "propionic.ac.s": 0.2, "butyric.ac.s": 0.05, "dm.loss": 4.5, "dm.ret": 95.5,
        "porosity": 0.42, "density.1": 220.0
    }
    res_cls = client.post("/api/v1/predict/silage/quality", json=silage_payload)
    data_cls = res_cls.json()
    check(
        "POST /api/v1/predict/silage/quality",
        res_cls.status_code == 200 and data_cls.get("predicted_class") in ["ea", "la"],
        f"(FAO Class: {data_cls.get('predicted_class')}, Conf: {data_cls.get('confidence')})"
    )

    res_fqi = client.post("/api/v1/predict/silage/fqi", json=silage_payload)
    data_fqi = res_fqi.json()
    check(
        "POST /api/v1/predict/silage/fqi",
        res_fqi.status_code == 200 and data_fqi.get("predicted_fqi") is not None,
        f"(FQI Score: {data_fqi.get('predicted_fqi')})"
    )

    res_comp = client.post("/api/v1/predict/silage/comprehensive", json=silage_payload)
    check("POST /api/v1/predict/silage/comprehensive", res_comp.status_code == 200)

    # 7. Feed Nutrition Multi-Target
    print("\n7. Testing Feed Nutrition Multi-Target Regression...")
    feed_payload = {
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
    res_feed = client.post("/api/v1/predict/feed-nutrition", json=feed_payload)
    data_feed = res_feed.json()
    preds = data_feed.get("predictions", {})
    check(
        "POST /api/v1/predict/feed-nutrition (All 7)",
        res_feed.status_code == 200 and len(preds) == 7,
        f"(Predicted: {list(preds.keys())})"
    )

    # 8. Milk Quality NIR Spectroscopy
    print("\n8. Testing Milk Quality NIR Spectroscopy (1024 Channels)...")
    mock_spectra = [float(0.45 + 0.05 * (i % 12)) for i in range(1024)]
    nir_payload = {"sample_id": "NIR_SAMPLE_TEST", "spectra": mock_spectra}
    res_nir = client.post("/api/v1/predict/milk-quality/fat", json=nir_payload)
    data_nir = res_nir.json()
    check(
        "POST /api/v1/predict/milk-quality/fat (1024 Channels)",
        res_nir.status_code == 200
        and data_nir.get("predicted_fat_percentage") is not None
        and data_nir.get("spectral_channels_used") == 1024,
        f"(Fat: {data_nir.get('predicted_fat_percentage')}%, Channels: {data_nir.get('spectral_channels_used')}, Tier: {data_nir.get('interpretation')})"
    )

    # 8b. Reject Invalid Spectral Dimensions (e.g. legacy 1032)
    res_nir_inv = client.post("/api/v1/predict/milk-quality/fat", json={"sample_id": "INV", "spectra": [0.5] * 1032})
    check(
        "POST /api/v1/predict/milk-quality/fat (Reject 1032 Legacy)",
        res_nir_inv.status_code == 422,
        f"(Expected 422 for non-1024 length -> Received HTTP {res_nir_inv.status_code})"
    )

    # 9. Sensor Contamination Screening
    print("\n9. Testing Sensor Contamination Screening (No synthetic data fabrication)...")
    contam_payload = {
        "electrical_conductivity_ms_cm": 5.1,
        "freezing_point_c": -0.535,
        "milk_ph": 6.65,
        "somatic_cell_count_raw": 120.0
    }
    res_contam = client.post("/api/v1/sensor-lab/contamination-screen", json=contam_payload)
    data_contam = res_contam.json()
    check(
        "POST /api/v1/sensor-lab/contamination-screen",
        res_contam.status_code == 200 and data_contam.get("water_adulteration_suspected") is False,
        f"(Water Adulteration: {data_contam.get('water_adulteration_suspected')}, Mastitis: {data_contam.get('subclinical_mastitis_risk')})"
    )

    # 10. Experimental Gating Verification
    print("\n10. Testing Experimental Model Access Gates...")
    myco_payload = {
        "protein_percent": 9.0, "fat_percent": 3.5, "moisture_percent": 14.0,
        "fiber_percent": 2.0, "starch_percent": 70.0, "ash_ai_percent": 1.2
    }
    res_myco = client.post("/api/v1/sensor-lab/mycotoxin-don", json=myco_payload)
    check(
        "POST /api/v1/sensor-lab/mycotoxin-don (Disabled check)",
        res_myco.status_code == 403,
        f"(Expected 403 when disabled -> Received HTTP {res_myco.status_code})"
    )

    print("\n" + "=" * 80)
    print(f"SMOKE TEST SUMMARY: {passed} PASSED, {failed} FAILED")
    print("=" * 80)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_smoke_tests()
