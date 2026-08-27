
# Silage Quality Model Report

## Dataset

`datasets\extracted\silage\datasilage.xlsx`

Sheet: `elab.all`

## Classification

Target: `fao.class`

Classes:
- ea
- la

Rows: 1425

### Metrics

- Accuracy: 0.971930
- Precision macro: 0.976618
- Recall macro: 0.958961
- F1 macro: 0.967120

## FQI Regression

Target: `fqi`

Rows: 1486

### Metrics

- MAE: 1.232211
- RMSE: 1.747282
- R²: 0.971934

## Saved Models

Classification:
`models\silage\silage_quality_classifier.joblib`

Regression:
`models\silage\silage_fqi_regressor.joblib`
