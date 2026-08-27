
# Feed Nutrition Model Report

Dataset:
`datasets/extracted/feed_nutrition/CowNflow_3_Feeds.tab`

The dataset contains 241 feed observations.

Models were trained only for variables actually represented in
the dataset.


## crude_protein

- Target: `Crude-protein-(g/kg-DM)`
- Valid rows: 241
- MAE: 64.785781
- RMSE: 172.788632
- R²: 0.931749
- Model: `models\feed_nutrition\feed_crude_protein_xgboost.joblib`


## dry_matter

- Target: `Dry-matter-(g/kg)`
- Valid rows: 200
- MAE: 28.267100
- RMSE: 52.275108
- R²: 0.973496
- Model: `models\feed_nutrition\feed_dry_matter_xgboost.joblib`


## crude_fibre

- Target: `Crude-fibre-(g/kg-DM)`
- Valid rows: 75
- MAE: 17.952860
- RMSE: 25.798265
- R²: 0.820593
- Model: `models\feed_nutrition\feed_crude_fibre_xgboost.joblib`


## ndf

- Target: `NDF-(g/kg-DM)`
- Valid rows: 187
- MAE: 31.761598
- RMSE: 57.379151
- R²: 0.855574
- Model: `models\feed_nutrition\feed_ndf_xgboost.joblib`


## adf

- Target: `ADF-(g/kg-DM)`
- Valid rows: 184
- MAE: 19.980364
- RMSE: 34.743678
- R²: 0.851280
- Model: `models\feed_nutrition\feed_adf_xgboost.joblib`


## adl

- Target: `ADL-(g/kg-DM)`
- Valid rows: 172
- MAE: 4.949024
- RMSE: 7.932071
- R²: 0.764882
- Model: `models\feed_nutrition\feed_adl_xgboost.joblib`


## starch

- Target: `Starch-(g/kg-DM)`
- Valid rows: 60
- MAE: 20.483360
- RMSE: 30.679030
- R²: 0.958814
- Model: `models\feed_nutrition\feed_starch_xgboost.joblib`


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
