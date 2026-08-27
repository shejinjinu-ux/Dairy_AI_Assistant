from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]

DATASET = (
    ROOT
    / "datasets"
    / "extracted"
    / "silage"
    / "datasilage.xlsx"
)

OUT = ROOT / "models" / "silage"
RESULTS = ROOT / "training" / "results"

OUT.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


print("=" * 70)
print("SILAGE QUALITY MODEL TRAINING")
print("=" * 70)

print("Loading:", DATASET)

df = pd.read_excel(
    DATASET,
    sheet_name="elab.all"
)

print("\nOriginal shape:", df.shape)

# ------------------------------------------------------------
# REMOVE COMPLETELY EMPTY / UNNAMED COLUMNS
# ------------------------------------------------------------

empty_columns = []

for col in df.columns:
    if str(col).startswith("Unnamed"):
        empty_columns.append(col)

df = df.drop(columns=empty_columns, errors="ignore")

print("\nDropped unnamed columns:")
print(empty_columns)

# ------------------------------------------------------------
# DISPLAY TARGET INFORMATION
# ------------------------------------------------------------

print("\nFAO class distribution:")
print(df["fao.class"].value_counts(dropna=False))

print("\nFQI statistics:")
print(df["fqi"].describe())

print("\npH statistics:")
print(df["pH"].describe())

# ------------------------------------------------------------
# FEATURES
# ------------------------------------------------------------

FEATURES = [
    "dm.f",
    "ash.f",
    "cp.f",
    "ee.f",
    "ndf.f",
    "adf.f",
    "lignin.f",
    "wsc.f",
    "starch.f",

    "dm.s",
    "ash.s",
    "cp.s",
    "ee.s",
    "ndf.s",
    "adf.s",
    "lignin.s",
    "starch.s",
    "wsc.s",

    "pH",
    "ammonia.s",
    "glucose.s",
    "fructose.s",
    "mannithol.s",
    "ethanol.s",
    "lactic.ac.s",
    "acetic.ac.s",
    "propionic.ac.s",
    "butyric.ac.s",

    "dm.loss",
    "dm.ret",
    "porosity",
    "density.1",
]

available_features = [
    col for col in FEATURES
    if col in df.columns
]

missing_features = [
    col for col in FEATURES
    if col not in df.columns
]

print("\nAvailable features:", len(available_features))

if missing_features:
    print("\nMissing features:")
    for col in missing_features:
        print("-", col)

# ------------------------------------------------------------
# MODEL A — FAO CLASS CLASSIFICATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL A — SILAGE CLASSIFICATION")
print("=" * 70)

classification_df = df[
    ["fao.class"] + available_features
].copy()

classification_df = classification_df.dropna(
    subset=["fao.class"]
)

classification_df["fao.class"] = (
    classification_df["fao.class"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# Keep only documented classes
classification_df = classification_df[
    classification_df["fao.class"].isin(["la", "ea"])
].copy()

print("\nClassification rows:", len(classification_df))

class_mapping = {
    "ea": 0,
    "la": 1,
}

y_cls = classification_df["fao.class"].map(class_mapping)

X_cls = classification_df[available_features].copy()

# ------------------------------------------------------------
# CLASSIFICATION SPLIT
# ------------------------------------------------------------

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_cls,
    y_cls,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_cls,
)

numeric_features = X_cls.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

categorical_features = X_cls.select_dtypes(
    include=["object", "category"]
).columns.tolist()

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

preprocessor_cls = ColumnTransformer([
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

classifier = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

classification_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor_cls
    ),
    (
        "model",
        classifier
    )
])

print("\nTraining classification model...")

classification_pipeline.fit(
    X_train_cls,
    y_train_cls
)

cls_pred = classification_pipeline.predict(
    X_test_cls
)

cls_accuracy = accuracy_score(
    y_test_cls,
    cls_pred
)

cls_precision = precision_score(
    y_test_cls,
    cls_pred,
    average="macro",
    zero_division=0
)

cls_recall = recall_score(
    y_test_cls,
    cls_pred,
    average="macro",
    zero_division=0
)

cls_f1 = f1_score(
    y_test_cls,
    cls_pred,
    average="macro",
    zero_division=0
)

cls_cm = confusion_matrix(
    y_test_cls,
    cls_pred
)

print("\nClassification metrics:")
print("Accuracy :", cls_accuracy)
print("Precision:", cls_precision)
print("Recall   :", cls_recall)
print("F1       :", cls_f1)

print("\nClassification report:")
print(
    classification_report(
        y_test_cls,
        cls_pred,
        target_names=["ea", "la"],
        zero_division=0
    )
)

print("\nConfusion matrix:")
print(cls_cm)

classification_model_path = (
    OUT / "silage_quality_classifier.joblib"
)

joblib.dump(
    classification_pipeline,
    classification_model_path
)

# ------------------------------------------------------------
# MODEL B — FQI REGRESSION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL B — FQI REGRESSION")
print("=" * 70)

regression_df = df[
    ["fqi"] + available_features
].copy()

regression_df = regression_df.dropna(
    subset=["fqi"]
)

print("\nRegression rows:", len(regression_df))

X_reg = regression_df[available_features].copy()

y_reg = pd.to_numeric(
    regression_df["fqi"],
    errors="coerce"
)

valid_mask = y_reg.notna()

X_reg = X_reg.loc[valid_mask].copy()
y_reg = y_reg.loc[valid_mask].copy()

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=RANDOM_STATE,
)

numeric_features_reg = X_reg.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

categorical_features_reg = X_reg.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numeric_pipeline_reg = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])

categorical_pipeline_reg = Pipeline([
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

preprocessor_reg = ColumnTransformer([
    (
        "num",
        numeric_pipeline_reg,
        numeric_features_reg
    ),
    (
        "cat",
        categorical_pipeline_reg,
        categorical_features_reg
    )
])

regressor = XGBRegressor(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

regression_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor_reg
    ),
    (
        "model",
        regressor
    )
])

print("\nTraining FQI regression model...")

regression_pipeline.fit(
    X_train_reg,
    y_train_reg
)

reg_pred = regression_pipeline.predict(
    X_test_reg
)

reg_mae = mean_absolute_error(
    y_test_reg,
    reg_pred
)

reg_rmse = np.sqrt(
    mean_squared_error(
        y_test_reg,
        reg_pred
    )
)

reg_r2 = r2_score(
    y_test_reg,
    reg_pred
)

print("\nRegression metrics:")
print("MAE :", reg_mae)
print("RMSE:", reg_rmse)
print("R2  :", reg_r2)

regression_model_path = (
    OUT / "silage_fqi_regressor.joblib"
)

joblib.dump(
    regression_pipeline,
    regression_model_path
)

# ------------------------------------------------------------
# SAVE METADATA
# ------------------------------------------------------------

metadata = {
    "dataset": str(
        DATASET.relative_to(ROOT)
    ),
    "sheet": "elab.all",
    "features": available_features,
    "classification_target": "fao.class",
    "classification_classes": ["ea", "la"],
    "classification_rows": int(len(classification_df)),
    "regression_target": "fqi",
    "regression_rows": int(len(regression_df)),
    "classification_metrics": {
        "accuracy": float(cls_accuracy),
        "precision_macro": float(cls_precision),
        "recall_macro": float(cls_recall),
        "f1_macro": float(cls_f1)
    },
    "regression_metrics": {
        "mae": float(reg_mae),
        "rmse": float(reg_rmse),
        "r2": float(reg_r2)
    },
    "random_state": RANDOM_STATE,
}

with open(
    OUT / "features.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        metadata,
        f,
        indent=2
    )

# ------------------------------------------------------------
# SAVE REPORT
# ------------------------------------------------------------

report = f"""
# Silage Quality Model Report

## Dataset

`{DATASET.relative_to(ROOT)}`

Sheet: `elab.all`

## Classification

Target: `fao.class`

Classes:
- ea
- la

Rows: {len(classification_df)}

### Metrics

- Accuracy: {cls_accuracy:.6f}
- Precision macro: {cls_precision:.6f}
- Recall macro: {cls_recall:.6f}
- F1 macro: {cls_f1:.6f}

## FQI Regression

Target: `fqi`

Rows: {len(regression_df)}

### Metrics

- MAE: {reg_mae:.6f}
- RMSE: {reg_rmse:.6f}
- R²: {reg_r2:.6f}

## Saved Models

Classification:
`{classification_model_path.relative_to(ROOT)}`

Regression:
`{regression_model_path.relative_to(ROOT)}`
"""

with open(
    RESULTS / "silage_report.md",
    "w",
    encoding="utf-8"
) as f:
    f.write(report)

# ------------------------------------------------------------
# RELOAD VALIDATION
# ------------------------------------------------------------

print("\nReloading models...")

loaded_classifier = joblib.load(
    classification_model_path
)

loaded_regressor = joblib.load(
    regression_model_path
)

classification_sample = loaded_classifier.predict(
    X_test_cls.iloc[:5]
)

regression_sample = loaded_regressor.predict(
    X_test_reg.iloc[:5]
)

print("\nClassification sample predictions:")
print(classification_sample)

print("\nFQI sample predictions:")
print(regression_sample)

print("\n" + "=" * 70)
print("SILAGE MODEL TRAINING COMPLETE")
print("=" * 70)