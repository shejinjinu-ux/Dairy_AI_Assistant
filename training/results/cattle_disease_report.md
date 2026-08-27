# Cattle Disease Image Model Report

- Model: `models/cattle_disease/cattle_disease_efficientnet_b3.pth`
- GPU: `NVIDIA GeForce RTX 3050 6GB Laptop GPU`
- Split: 70% train (13947), 15% validation (2989), 15% test (2989)
- Skipped unreadable images: 0

## Metrics

- accuracy: 0.989294
- precision_macro: 0.988431
- recall_macro: 0.989916
- f1_macro: 0.989152
- f1_weighted: 0.989292

## Classification Report

```text
              precision    recall  f1-score   support

         FMD       0.99      0.98      0.99       423
         IBK       0.99      1.00      0.99       469
         LSD       0.98      0.99      0.99       638
      Normal       0.99      0.99      0.99      1459

    accuracy                           0.99      2989
   macro avg       0.99      0.99      0.99      2989
weighted avg       0.99      0.99      0.99      2989

```
## Confusion Matrix

| Actual / Predicted | FMD | IBK | LSD | Normal |
|---|---|---|---|---|
| FMD | 415 | 1 | 0 | 7 |
| IBK | 0 | 468 | 0 | 1 |
| LSD | 0 | 1 | 634 | 3 |
| Normal | 4 | 3 | 12 | 1440 |

## Prediction Smoke Test

- `Normal_0_4177 (2).jpg` -> **Normal** (0.997559)
- `Normal_0_947.jpg` -> **Normal** (0.999512)
- `KERATO-CONJUNCTIVITIS_0_3300.jpg` -> **IBK** (1.000000)
- `Normal_0_6534.jpg` -> **Normal** (0.997559)
- `Lumpy_Skin_30_png.rf.306378faff772c1b3d186c1df0a2f87c.jpg` -> **LSD** (0.862793)
- `FMD_0_4541.jpg` -> **FMD** (1.000000)
- `Normal_0_7031 (2).jpg` -> **Normal** (0.999512)
- `lumpy-797-_png.rf.4f42be57e010c66c742a933709013936.jpg` -> **LSD** (1.000000)
- `FMD_0_9491.jpg` -> **FMD** (1.000000)
- `lumpy-310-_png.rf.0429cb88930f13348394cb1e07a1dbcc.jpg` -> **LSD** (1.000000)
