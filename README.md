**3D Printing Defect Classification:**

Classifies defects in 3D-printed parts from a single photo, predicting one of 13 defect types (or "good").

**Dataset**:

Three public sources merged into** 13 classes** (~12,500 training images after balancing):
- **Roboflow** – 12 defect classes (backbone)
- **Bast et al.** – the `extrusion_error` class
- **Kaggle** – good / stringing / under-extrusion / spaghetti

Split 70 / 15 / 15, with near-duplicates grouped so no images leak between splits.

## Models & results

Same data and training recipe for all three (PyTorch + timm, weighted cross-entropy, AdamW). Main metric: macro-F1.

| Model | Pretrained | Test macro-F1 |
|---|---|---|
| Simple CNN (from scratch) | No | 0.84 |
| ResNet-50 | Yes | 0.93 |
| ViT-Small | Yes | 0.94 |

ResNet-50 and ViT-Small are effectively tied; **ResNet-50 is the final model (0.93)**. Transfer learning beat the baseline by ~9 points.

## Limitations

`extrusion_error` comes from one source (a possible shortcut), good vs. mild defects can look alike, and training images show prints in progress rather than finished parts.

## Run

1. `defect_dataset_builder_v6.ipynb` – builds the merged dataset.
2. `defect_training_v2.ipynb` – trains and evaluates the models.
