# Milk Quality NIR Model Report

Dataset: `datasets\extracted\milk_quality\milk_nir_composition.csv`

Samples: 1224

Spectral features: 1032


## Fat

- Selected model: xgboost
- Test rows: 245
- MAE: 0.176379
- RMSE: 0.290158
- R²: 0.872255
- PCA components: 95
- Saved model: `models\milk_quality_nir\milk_quality_fat_xgboost.joblib`


## Prot

- Selected model: xgboost
- Test rows: 245
- MAE: 0.177646
- RMSE: 0.224689
- R²: 0.556177
- PCA components: 95
- Saved model: `models\milk_quality_nir\milk_quality_prot_xgboost.joblib`


## Lact

- Selected model: random_forest
- Test rows: 245
- MAE: 0.106618
- RMSE: 0.144936
- R²: 0.255786
- PCA components: 95
- Saved model: `models\milk_quality_nir\milk_quality_lact_random_forest.joblib`


## SCC

- Selected model: xgboost
- Test rows: 245
- MAE: 178.404587
- RMSE: 325.553087
- R²: -0.217173
- PCA components: 95
- Saved model: `models\milk_quality_nir\milk_quality_scc_xgboost.joblib`


## Urea

- Selected model: random_forest
- Test rows: 245
- MAE: 3.588218
- RMSE: 4.435769
- R²: 0.240223
- PCA components: 95
- Saved model: `models\milk_quality_nir\milk_quality_urea_random_forest.joblib`


## Interpretation

These models are NIR-spectroscopy models. They should be used
with compatible NIR spectral measurements in the future sensor
integration.

They should not be presented as image-only milk chemistry models.
