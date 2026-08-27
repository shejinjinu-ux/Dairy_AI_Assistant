# Corrected Milk Quality NIR Model Report (1024 Spectral Channels)

Dataset: `datasets\extracted\milk_quality\milk_nir_composition.csv`

Samples: 1224

Spectral features: 1024 (Exclusively Trans_* channels)

Non-spectral metadata columns excluded: `Cow_ID`, `Milk_yield`, `Milk_Interv`, `SET`, `Time_Dark`, `Time_Milk`, `Time_PrevMilk`, `Time_Sample`, `Time_White`


## Fat

- Selected model: xgboost
- Test rows: 245
- MAE: 0.173652
- RMSE: 0.282112
- R²: 0.879242
- PCA components: 95
- Saved model: `models\milk_quality_nir_corrected\milk_quality_fat_xgboost.joblib`


## Prot

- Selected model: xgboost
- Test rows: 245
- MAE: 0.191767
- RMSE: 0.254531
- R²: 0.430456
- PCA components: 95
- Saved model: `models\milk_quality_nir_corrected\milk_quality_prot_xgboost.joblib`


## Lact

- Selected model: xgboost
- Test rows: 245
- MAE: 0.109011
- RMSE: 0.148092
- R²: 0.223025
- PCA components: 95
- Saved model: `models\milk_quality_nir_corrected\milk_quality_lact_xgboost.joblib`


## SCC

- Selected model: xgboost
- Test rows: 245
- MAE: 185.917526
- RMSE: 340.849146
- R²: -0.334237
- PCA components: 95
- Saved model: `models\milk_quality_nir_corrected\milk_quality_scc_xgboost.joblib`


## Urea

- Selected model: random_forest
- Test rows: 245
- MAE: 3.545714
- RMSE: 4.399990
- R²: 0.252430
- PCA components: 95
- Saved model: `models\milk_quality_nir_corrected\milk_quality_urea_random_forest.joblib`


## Interpretation

These models are trained strictly on the 1024 NIR-spectroscopy channels without metadata leakage.
They should be used with compatible NIR spectral measurements in future sensor integrations.
