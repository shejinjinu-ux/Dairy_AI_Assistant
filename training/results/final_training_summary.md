# Final Training Summary

## Status

Training was attempted only for the ready tabular features:

- Cattle disease tabular classification
- Milk production prediction
- Silage quality regression

The disease tabular Random Forest completed before the process was stopped. The remaining model fits did not complete within the available execution window. No fabricated metrics or model files were retained.

## Dataset Collection

Downloaded and inspected datasets are documented in `training/results/dataset_inventory.md` and `training/dataset_paths.json`.

Pending datasets and reasons are documented in:

- `training/results/feed_nutrition_pending.md`
- `training/results/feed_adulteration_pending.md`
- `training/results/feed_mold_pending.md`
- `training/results/milk_quality_pending.md`

The Indian bovine breed archive remained incomplete after repeated network interruption; only its metadata CSV is complete.

## Models

The prior disease tabular Random Forest was rejected for 0.0182981701829817 accuracy and removed from the application registry and filesystem. No trained model is currently registered.

## Validation

The rejected model artifacts were removed and the registry was verified empty. Raw datasets and inspection reports were not modified.
