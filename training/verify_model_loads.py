import json
from pathlib import Path

import joblib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

with open(ROOT / "models" / "model_registry.json", encoding="utf-8-sig") as f:
    registry = json.load(f)

print("=" * 70)
print("MODEL LOAD & ARTIFACT INTEGRITY VERIFICATION")
print("=" * 70)

passed = 0
failed = 0

for name, info in registry["models"].items():
    path = ROOT / info["path"]
    framework = info["framework"]

    print(f"\nModel: {name}")
    print(f"Path:  {info['path']}")
    print(f"Framework: {framework}")

    try:
        if not path.exists():
            raise FileNotFoundError(f"Model file does not exist: {path}")

        if framework == "xgboost":
            model = joblib.load(path)
            print("  [PASS] XGBoost/joblib pipeline load OK")

            # Specific check for corrected Milk NIR models: verify 1024 features
            if "milk_quality" in name:
                dummy_1024 = np.random.rand(1, 1024)
                pred = model.predict(dummy_1024)
                print(f"  [PASS] 1024-channel inference test OK (dummy prediction: {pred[0]:.4f})")

        elif framework == "pytorch":
            checkpoint = torch.load(path, map_location="cpu")
            print("  [PASS] PyTorch checkpoint load OK")

        else:
            print("  [WARN] Unknown framework")
            failed += 1
            continue

        passed += 1

    except Exception as e:
        print("  [FAIL] LOAD/VERIFICATION FAILED:", e)
        failed += 1

print("\n" + "=" * 70)
print(f"VERIFICATION SUMMARY: {passed} PASSED, {failed} FAILED")
print("=" * 70)

if failed:
    raise SystemExit(1)

print("ALL REGISTERED AND CORRECTED ARTIFACTS VERIFIED SUCCESSFULLY.")
