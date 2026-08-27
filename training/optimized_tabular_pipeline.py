from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / 'models' / 'model_registry.json'


def load_frame(path: Path, target: str, max_rows: int, seed: int) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if len(frame) > max_rows:
        frame, _ = train_test_split(frame, train_size=max_rows, random_state=seed, stratify=None)
    return frame.dropna(subset=[target]).reset_index(drop=True)


def build_preprocessor(frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    X = frame.drop(columns=[target])
    y = frame[target]
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    preprocessor = ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median'), numeric),
        ('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
        ]), categorical),
    ], sparse_threshold=0)
    return X, y, preprocessor


def estimator_for(name: str):
    if name == 'random_forest':
        return RandomForestRegressor(n_estimators=40, max_depth=16, max_features='sqrt', n_jobs=1, random_state=42)
    if name == 'xgboost':
        return XGBRegressor(n_estimators=250, max_depth=6, learning_rate=0.06, subsample=0.8, colsample_bytree=0.8, tree_method='hist', n_jobs=1, random_state=42, objective='reg:squarederror')
    raise ValueError(f'Unsupported model: {name}')


def train_regression(task: str, model_name: str, source: Path, target: str, output_dir: Path, max_rows: int):
    frame = load_frame(source, target, max_rows, 42)
    X, y, preprocessor = build_preprocessor(frame, target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_matrix = preprocessor.fit_transform(X_train)
    test_matrix = preprocessor.transform(X_test)
    del frame, X, X_train, X_test
    gc.collect()

    model = estimator_for(model_name)
    model.fit(train_matrix, y_train)
    prediction = model.predict(test_matrix)
    metrics = {
        'mae': float(mean_absolute_error(y_test, prediction)),
        'rmse': float(mean_squared_error(y_test, prediction) ** 0.5),
        'r2': float(r2_score(y_test, prediction)),
        'rows_used': int(len(y_train) + len(y_test)),
        'test_rows': int(len(y_test)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{task}_{model_name}'
    model_path = output_dir / f'{stem}.joblib'
    preprocessor_path = output_dir / f'{stem}_preprocessor.joblib'
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    registry = json.loads(REGISTRY_PATH.read_text(encoding='utf-8')) if REGISTRY_PATH.exists() else {}
    registry[stem] = {
        'model_path': model_path.relative_to(ROOT).as_posix(),
        'model_type': type(model).__name__,
        'input_features': list(y_train.index.names) if False else list(preprocessor.feature_names_in_),
        'target': target,
        'preprocessing_path': preprocessor_path.relative_to(ROOT).as_posix(),
        'evaluation_metrics': metrics,
        'dataset_source': source.relative_to(ROOT).as_posix(),
    }
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'model': stem, 'metrics': metrics, 'model_path': str(model_path.relative_to(ROOT))}, indent=2))


def main():
    parser = argparse.ArgumentParser(description='Train exactly one memory-bounded tabular model per invocation.')
    parser.add_argument('--task', choices=['milk_production', 'silage', 'disease_status'], required=True)
    parser.add_argument('--model', choices=['random_forest', 'xgboost'], required=True)
    parser.add_argument('--max-rows', type=int, default=100000)
    args = parser.parse_args()
    if args.task == 'disease_status':
        raise SystemExit('Disease-status training is intentionally deferred; use the approved 250,000-row plan later.')
    if args.task == 'milk_production':
        train_regression('milk_production', args.model, ROOT / 'datasets/extracted/cattle_health_feeding/global_cattle_milk_yield_prediction_dataset.csv', 'Milk_Yield_L', ROOT / 'models/milk_production', args.max_rows)
    else:
        raise SystemExit('Silage training requires a reviewed target and is intentionally deferred until target approval.')


if __name__ == '__main__':
    main()
