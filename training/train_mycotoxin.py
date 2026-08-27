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
    / "mycotoxin"
    / "corn_mycotoxin_feed.xlsx"
)

OUT = ROOT / "models" / "mycotoxin"
RESULTS = ROOT / "training" / "results"

OUT.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

print("=" * 70)
print("MYCOTOXIN REGRESSION MODEL TRAINING")
print("=" * 70)

df = pd.read_excel(DATASET)

print("Dataset shape:", df.shape)

# ------------------------------------------------------------
# INPUT FEATURES
# ------------------------------------------------------------

numeric_features = [
    "Protein",
    "Fat",
    "Moisture",
    "Fiber",
    "Starch",
    "AshAI",
    "L*(D65) SCI",
    "a*(D65) SCI",
    "b*(D65) SCI",
    "harvest year",
]

categorical_features = [
    "sample type",
    "Sample Location",
]

targets = {
    "AFB1": "AFB1 (ppm)",
    "FUM": "FUM  (ppm) LCMS",
    "ZEA": "ZEA (ppm)",
    "DON": "DON (ppm) LCMS",
}

# Keep only existing columns
numeric_features = [
    c for c in numeric_features if c in df.columns
]

categorical_features = [
    c for c in categorical_features if c in df.columns
]

feature_columns = (
    numeric_features +
    categorical_features
)

print("Features:", feature_columns)

# ------------------------------------------------------------
# CLEAN
# ------------------------------------------------------------

df = df.drop_duplicates().copy()

for col in numeric_features + list(targets.values()):
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

results = {}

for model_name, target_column in targets.items():

    print("\n" + "=" * 70)
    print("TARGET:", model_name)
    print("=" * 70)

    work = df[
        feature_columns + [target_column]
    ].copy()

    work = work.dropna(
        subset=[target_column]
    )

    print("Valid rows:", len(work))

    X = work[feature_columns].copy()

    y_original = work[target_column].astype(float)

    # All concentrations must be non-negative.
    y_original = y_original.clip(lower=0)

    # Log transform for skewed toxin concentrations.
    y = np.log1p(y_original)

    X_train, X_test, y_train, y_test, y_original_train, y_original_test = (
        train_test_split(
            X,
            y,
            y_original,
            test_size=0.20,
            random_state=RANDOM_STATE,
        )
    )

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

    preprocessor = ColumnTransformer([
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
    ])

    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    print("Training XGBoost...")

    pipeline.fit(
        X_train,
        y_train
    )

    pred_log = pipeline.predict(
        X_test
    )

    # Return to original ppm scale
    predictions = np.expm1(pred_log)

    predictions = np.clip(
        predictions,
        0,
        None
    )

    mae = mean_absolute_error(
        y_original_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_original_test,
            predictions
        )
    )

    r2 = r2_score(
        y_original_test,
        predictions
    )

    print("\nMAE :", mae)
    print("RMSE:", rmse)
    print("R2  :", r2)

    model_path = (
        OUT / f"mycotoxin_{model_name.lower()}_xgboost.joblib"
    )

    joblib.dump(
        pipeline,
        model_path
    )

    # Reload check
    loaded = joblib.load(
        model_path
    )

    reload_log = loaded.predict(
        X_test.iloc[:5]
    )

    reload_predictions = np.expm1(
        reload_log
    )

    print("Reload sample predictions:")
    print(reload_predictions)

    results[model_name] = {
        "target": target_column,
        "valid_rows": int(len(work)),
        "mae_ppm": float(mae),
        "rmse_ppm": float(rmse),
        "r2": float(r2),
        "model_path": str(
            model_path.relative_to(ROOT)
        ),
    }

# ------------------------------------------------------------
# SAVE METADATA
# ------------------------------------------------------------

metadata = {
    "dataset": str(
        DATASET.relative_to(ROOT)
    ),
    "samples": int(len(df)),
    "features": feature_columns,
    "targets": targets,
    "results": results,
    "target_transform": "log1p",
    "random_state": RANDOM_STATE,
    "limitations": [
        "Dataset contains corn/maize feed samples.",
        "AFB1 and ZEA category labels are incomplete.",
        "Regression models use measured toxin concentrations.",
        "This is a screening prototype, not laboratory replacement."
    ]
}

with open(
    OUT / "metadata.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        metadata,
        f,
        indent=2
    )

print("\n" + "=" * 70)
print("MYCOTOXIN TRAINING COMPLETE")
print("=" * 70)

print(json.dumps(
    results,
    indent=2
))