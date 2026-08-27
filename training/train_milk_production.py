from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]

DATASET = (
    ROOT
    / "datasets"
    / "extracted"
    / "cattle_health_feeding"
    / "global_cattle_milk_yield_prediction_dataset.csv"
)

OUT = ROOT / "models" / "milk_production"
RESULTS = ROOT / "training" / "results"

OUT.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

print("=" * 70)
print("MILK PRODUCTION MODEL TRAINING")
print("=" * 70)

print("Loading:", DATASET)

df = pd.read_csv(DATASET)

print("\nShape:", df.shape)
print("\nColumns:")
for c in df.columns:
    print("-", c)

print("\nMissing values:")
print(df.isna().sum().sort_values(ascending=False).head(20))

print("\nData types:")
print(df.dtypes)

# ------------------------------------------------------------
# FIND TARGET COLUMN
# ------------------------------------------------------------

target_candidates = [
    "Milk_Yield_L",
    "Milk_Yield",
    "milk_yield",
    "milk_yield_l",
    "milk_yield_litres",
    "MilkYield",
]

target = None

for candidate in target_candidates:
    if candidate in df.columns:
        target = candidate
        break

if target is None:
    raise ValueError(
        "Could not identify the milk-yield target. "
        "Check the printed column list and update target_candidates."
    )

print("\nTarget:", target)

# ------------------------------------------------------------
# BASIC CLEANING
# ------------------------------------------------------------

df = df.drop_duplicates().copy()

# Convert target to numeric where possible
df[target] = pd.to_numeric(df[target], errors="coerce")

# Remove rows where target is missing
df = df.dropna(subset=[target]).copy()

# Remove obviously invalid negative milk yield
df = df[df[target] >= 0].copy()

print("\nCleaned shape:", df.shape)

# ------------------------------------------------------------
# REMOVE LEAKAGE / IDENTIFIER COLUMNS
# ------------------------------------------------------------

X = df.drop(columns=[target]).copy()
y = df[target].astype(float)

drop_keywords = [
    "id",
    "record",
    "timestamp",
    "date_created",
    "created_at",
]

drop_columns = []

for col in X.columns:
    low = col.lower()

    # Drop obvious identifiers only.
    if low in {"id", "animal_id", "record_id"}:
        drop_columns.append(col)

for col in drop_columns:
    X = X.drop(columns=[col], errors="ignore")

print("\nDropped identifier columns:")
print(drop_columns)

print("\nRemaining features:", len(X.columns))

# ------------------------------------------------------------
# DETERMINE NUMERIC/CATEGORICAL FEATURES
# ------------------------------------------------------------

numeric_features = X.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)

# ------------------------------------------------------------
# TRAIN / TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
)

print("\nTrain rows:", len(X_train))
print("Test rows:", len(X_test))

# ------------------------------------------------------------
# PREPROCESSOR
# ------------------------------------------------------------

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "onehot",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True
        )
    )
])

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_pipeline,
            numeric_features
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    ],
    remainder="drop"
)

# ------------------------------------------------------------
# XGBOOST
# ------------------------------------------------------------

model = XGBRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

print("\nTraining XGBoost...")

pipeline.fit(X_train, y_train)

# ------------------------------------------------------------
# EVALUATION
# ------------------------------------------------------------

pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, pred)

rmse = np.sqrt(
    mean_squared_error(y_test, pred)
)

r2 = r2_score(y_test, pred)

print("\n" + "=" * 70)
print("FINAL MILK PRODUCTION RESULTS")
print("=" * 70)

print(f"MAE  : {mae:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"R2   : {r2:.6f}")

# ------------------------------------------------------------
# SAVE MODEL
# ------------------------------------------------------------

model_path = OUT / "milk_production_xgboost.joblib"

joblib.dump(
    pipeline,
    model_path
)

print("\nSaved model:")
print(model_path)

# ------------------------------------------------------------
# SAVE FEATURE INFO
# ------------------------------------------------------------

feature_info = {
    "dataset": str(DATASET.relative_to(ROOT)),
    "target": target,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "random_state": RANDOM_STATE,
    "model": "XGBRegressor",
}

with open(
    OUT / "features.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        feature_info,
        f,
        indent=2
    )

# ------------------------------------------------------------
# SAVE REPORT
# ------------------------------------------------------------

report = f"""# Milk Production Model Report

## Dataset

`{DATASET.relative_to(ROOT)}`

## Target

`{target}`

## Data

- Original rows: {len(df)}
- Train rows: {len(X_train)}
- Test rows: {len(X_test)}

## Model

XGBoost Regressor

## Metrics

- MAE: {mae:.6f}
- RMSE: {rmse:.6f}
- R²: {r2:.6f}

## Saved Model

`{model_path.relative_to(ROOT)}`
"""

with open(
    RESULTS / "milk_production_report.md",
    "w",
    encoding="utf-8"
) as f:
    f.write(report)

# ------------------------------------------------------------
# RELOAD CHECK
# ------------------------------------------------------------

print("\nReloading saved model...")

loaded_model = joblib.load(model_path)

sample_prediction = loaded_model.predict(
    X_test.iloc[:5]
)

print("\nSample predictions:")
print(sample_prediction)

print("\nMilk production model completed successfully.")