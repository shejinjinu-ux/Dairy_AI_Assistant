import json
from pathlib import Path

with open("models/model_registry.json", encoding="utf-8-sig") as f:
    registry = json.load(f)

ok = 0
fail = 0

print("MODEL FILE CHECK")
print("=" * 60)

for name, info in registry["models"].items():
    path = Path(info["path"])

    if path.exists():
        print(f"✅ {name} -> {path}")
        ok += 1
    else:
        print(f"❌ {name} -> MISSING: {path}")
        fail += 1

print("=" * 60)
print(f"EXISTS : {ok}")
print(f"MISSING: {fail}")
