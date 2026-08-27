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
    / "feed_nutrition"
    / "CowNflow_3_Feeds.tab"
)

OUT = ROOT / "models" / "feed_nutrition"
RESULTS = ROOT / "training" / "results"

OUT.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

print("=" * 70)
print("FEED NUTRITION MODEL TRAINING")
print("=" * 70)

df = pd.read_csv(DATASET, sep="\t")

print("Original shape:", df.shape)

# ---------------------------------------------------------
# FEATURE/TARGET DEFINITIONS
# ---------------------------------------------------------

categorical_features = [
    "Feed-category",
    "Detailed-feed-category-INRA2018",
]

numeric_input_features = [
    "Dry-matter-(g/kg)",
    "Organic-matter-(g/kg-DM)-",
    "Ash-(g/kg-DM)",
    "Crude-fibre-(g/kg-DM)",
    "NDF-(g/kg-DM)",
    "ADF-(g/kg-DM)",
    "Starch-(g/kg-DM)",
]

targets = {
    "crude_protein": "Crude-protein-(g/kg-DM)",
    "dry_matter": "Dry-matter-(g/kg)",
    "crude_fibre": "Crude-fibre-(g/kg-DM)",
    "ndf": "NDF-(g/kg-DM)",
    "adf": "ADF-(g/kg-DM)",
    "adl": "ADL-(g/kg-DM)",
    "starch": "Starch-(g/kg-DM)",
}

# ---------------------------------------------------------
# CLEAN DATA
# ---------------------------------------------------------

df = df.drop_duplicates().copy()

print("After duplicates:", df.shape)

# Numeric conversion
for col in numeric_input_features + list(targets.values()):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ---------------------------------------------------------
# TRAIN A SEPARATE MODEL FOR EACH TARGET
# ---------------------------------------------------------

results = {}

for model_name, target_column in targets.items():

    print("\n" + "=" * 70)
    print(f"TARGET: {model_name}")
    print("=" * 70)

    if target_column not in df.columns:
        print("Target not available:", target_column)
        continue

    # Avoid using the target itself as an input feature.
    numeric_features = [
        c for c in numeric_input_features
        if c != target_column
    ]

    feature_columns = (
        numeric_features +
        categorical_features
    )

    work = df[
        feature_columns + [target_column]
    ].copy()

    # Target must exist
    work = work.dropna(
        subset=[target_column]
    )

    if len(work) < 30:
        print(
            f"Skipping {model_name}: only {len(work)} valid rows."
        )
        continue

    X = work[feature_columns]
    y = work[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    numeric_available = [
        c for c in numeric_features
        if c in X.columns
    ]

    categorical_available = [
        c for c in categorical_features
        if c in X.columns
    ]

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
                numeric_available
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_available
            )
        ]
    )

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

    print("Valid rows:", len(work))
    print("Training...")

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print(f"MAE : {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R2  : {r2:.6f}")

    model_path = (
        OUT / f"feed_{model_name}_xgboost.joblib"
    )

    joblib.dump(
        pipeline,
        model_path
    )

    # Reload validation
    loaded = joblib.load(
        model_path
    )

    sample_predictions = loaded.predict(
        X_test.iloc[:5]
    )

    print("Reload sample predictions:")
    print(sample_predictions)

    results[model_name] = {
        "target_column": target_column,
        "valid_rows": int(len(work)),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "model_path": str(
            model_path.relative_to(ROOT)
        ),
    }

# ---------------------------------------------------------
# METADATA
# ---------------------------------------------------------

metadata = {
    "dataset": str(
        DATASET.relative_to(ROOT)
    ),
    "dataset_rows": int(len(df)),
    "targets": list(results.keys()),
    "results": results,
    "limitations": [
        "Dataset does not contain direct energy variables.",
        "Dataset does not contain direct mineral variables.",
        "Predictions apply to nutritional variables represented in the dataset."
    ]
}

with open(
    OUT / "feed_model_metadata.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        metadata,
        f,
        indent=2
    )

# ---------------------------------------------------------
# REPORT
# ---------------------------------------------------------

report = """
# Feed Nutrition Model Report

Dataset:
`datasets/extracted/feed_nutrition/CowNflow_3_Feeds.tab`

The dataset contains 241 feed observations.

Models were trained only for variables actually represented in
the dataset.

"""

for name, info in results.items():

    report += f"""
## {name}

- Target: `{info['target_column']}`
- Valid rows: {info['valid_rows']}
- MAE: {info['mae']:.6f}
- RMSE: {info['rmse']:.6f}
- R²: {info['r2']:.6f}
- Model: `{info['model_path']}`

"""

report += """
## Dataset limitations

The current dataset does not contain direct measurements for:

- Metabolizable energy
- Calcium
- Phosphorus
- Magnesium
- Sodium
- Potassium
- Iron
- Zinc
- Manganese

These should not be fabricated or inferred without a compatible
validated dataset.
"""

with open(
    RESULTS / "feed_nutrition_report.md",
    "w",
    encoding="utf-8"
) as f:
    f.write(report)

print("\n" + "=" * 70)
print("FEED NUTRITION TRAINING COMPLETE")
print("=" * 70)

print(json.dumps(
    results,
    indent=2
))