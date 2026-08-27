from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]

DATASET = (
    ROOT
    / "datasets"
    / "extracted"
    / "milk_quality"
    / "milk_nir_composition.csv"
)

OUT = ROOT / "models" / "milk_quality_nir_corrected"
RESULTS = ROOT / "training" / "results"

OUT.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

TARGETS = ["Fat", "Prot", "Lact", "SCC", "Urea"]

RANDOM_STATE = 42

print("=" * 70)
print("MILK QUALITY NIR MODEL TRAINING (1024 SPECTRAL CHANNELS)")
print("=" * 70)

df = pd.read_csv(DATASET)

print("Dataset shape:", df.shape)

# ------------------------------------------------------------
# Identify strictly NIR spectral columns (Trans_*)
# ------------------------------------------------------------

spectral_prefixes = ("Trans_Dark_", "Trans_Sample_", "Trans_Tot_", "Trans_White_")

spectral_columns = [
    c for c in df.columns
    if c.startswith(spectral_prefixes) and pd.api.types.is_numeric_dtype(df[c])
]

# Explicit verification: must be exactly 1024 spectral features
print("Identified NIR spectral features:", len(spectral_columns))
if len(spectral_columns) != 1024:
    raise ValueError(f"Expected exactly 1024 spectral features, found {len(spectral_columns)}")

# Strict check: none of the 9 numeric metadata columns may be included
forbidden_metadata = [
    "Cow_ID", "Milk_yield", "Milk_Interv", "SET", "Time_Dark",
    "Time_Milk", "Time_PrevMilk", "Time_Sample", "Time_White"
]
for meta_col in forbidden_metadata:
    if meta_col in spectral_columns:
        raise ValueError(f"CRITICAL ERROR: Metadata column '{meta_col}' found in spectral feature set!")

print("Metadata exclusion check passed. Excluded all non-spectral columns.")

X = df[spectral_columns].copy()

# Convert any accidental invalid values
X = X.apply(pd.to_numeric, errors="coerce")

# Fill spectral missing values just in case
X = X.fillna(X.median())

results = {}

for target in TARGETS:

    print("\n" + "=" * 70)
    print("TARGET:", target)
    print("=" * 70)

    y = pd.to_numeric(
        df[target],
        errors="coerce"
    )

    valid = y.notna()

    X_target = X.loc[valid].copy()
    y_target = y.loc[valid].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_target,
        y_target,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    n_features = X_train.shape[1]
    n_components = min(
        95,
        n_features,
        max(2, X_train.shape[0] - 1)
    )

    # --------------------------------------------------------
    # PCA + XGBoost
    # --------------------------------------------------------

    xgb_pipeline = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "pca",
            PCA(
                n_components=n_components,
                random_state=RANDOM_STATE
            )
        ),
        (
            "model",
            XGBRegressor(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="reg:squarederror",
                eval_metric="rmse",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        )
    ])

    print("Training XGBoost...")

    xgb_pipeline.fit(
        X_train,
        y_train
    )

    xgb_pred = xgb_pipeline.predict(
        X_test
    )

    xgb_mae = mean_absolute_error(
        y_test,
        xgb_pred
    )

    xgb_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            xgb_pred
        )
    )

    xgb_r2 = r2_score(
        y_test,
        xgb_pred
    )

    print("\nXGBoost")
    print("MAE :", xgb_mae)
    print("RMSE:", xgb_rmse)
    print("R2  :", xgb_r2)

    # --------------------------------------------------------
    # PCA + Random Forest
    # --------------------------------------------------------

    rf_pipeline = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "pca",
            PCA(
                n_components=n_components,
                random_state=RANDOM_STATE
            )
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        )
    ])

    print("\nTraining Random Forest...")

    rf_pipeline.fit(
        X_train,
        y_train
    )

    rf_pred = rf_pipeline.predict(
        X_test
    )

    rf_mae = mean_absolute_error(
        y_test,
        rf_pred
    )

    rf_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            rf_pred
        )
    )

    rf_r2 = r2_score(
        y_test,
        rf_pred
    )

    print("\nRandom Forest")
    print("MAE :", rf_mae)
    print("RMSE:", rf_rmse)
    print("R2  :", rf_r2)

    # --------------------------------------------------------
    # Select best model using R2
    # --------------------------------------------------------

    candidates = [
        (
            "xgboost",
            xgb_pipeline,
            xgb_mae,
            xgb_rmse,
            xgb_r2,
        ),
        (
            "random_forest",
            rf_pipeline,
            rf_mae,
            rf_rmse,
            rf_r2,
        ),
    ]

    best = max(
        candidates,
        key=lambda item: item[4]
    )

    best_name, best_pipeline, best_mae, best_rmse, best_r2 = best

    model_path = (
        OUT / f"milk_quality_{target.lower()}_{best_name}.joblib"
    )

    joblib.dump(
        best_pipeline,
        model_path
    )

    # Reload test
    loaded = joblib.load(model_path)

    reload_prediction = loaded.predict(
        X_test.iloc[:5]
    )

    print("\nSELECTED MODEL:", best_name)
    print("Best R2:", best_r2)

    print("Reload sample predictions:")
    print(reload_prediction)

    results[target] = {
        "selected_model": best_name,
        "mae": float(best_mae),
        "rmse": float(best_rmse),
        "r2": float(best_r2),
        "test_rows": int(len(X_test)),
        "spectral_features": int(n_features),
        "pca_components": int(n_components),
        "model_path": str(
            model_path.relative_to(ROOT)
        ),
    }

# ------------------------------------------------------------
# Save metadata
# ------------------------------------------------------------

metadata = {
    "dataset": str(
        DATASET.relative_to(ROOT)
    ),
    "samples": int(len(df)),
    "spectral_features": int(len(spectral_columns)),
    "targets": TARGETS,
    "results": results,
    "random_state": RANDOM_STATE,
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

# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

report = "# Corrected Milk Quality NIR Model Report (1024 Spectral Channels)\n\n"

report += f"Dataset: `{DATASET.relative_to(ROOT)}`\n\n"
report += f"Samples: {len(df)}\n\n"
report += f"Spectral features: {len(spectral_columns)} (Exclusively Trans_* channels)\n\n"
report += "Non-spectral metadata columns excluded: `Cow_ID`, `Milk_yield`, `Milk_Interv`, `SET`, `Time_Dark`, `Time_Milk`, `Time_PrevMilk`, `Time_Sample`, `Time_White`\n\n"

for target, info in results.items():

    report += f"""
## {target}

- Selected model: {info['selected_model']}
- Test rows: {info['test_rows']}
- MAE: {info['mae']:.6f}
- RMSE: {info['rmse']:.6f}
- R²: {info['r2']:.6f}
- PCA components: {info['pca_components']}
- Saved model: `{info['model_path']}`

"""

report += """
## Interpretation

These models are trained strictly on the 1024 NIR-spectroscopy channels without metadata leakage.
They should be used with compatible NIR spectral measurements in future sensor integrations.
"""

with open(
    RESULTS / "milk_quality_nir_corrected_report.md",
    "w",
    encoding="utf-8"
) as f:
    f.write(report)

print("\n" + "=" * 70)
print("MILK QUALITY NIR TRAINING COMPLETE")
print("=" * 70)

print(json.dumps(
    results,
    indent=2
))