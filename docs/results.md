# FactorySense-R Results

## Evaluation Setup

The current evaluation uses a synthetic MVTec-style demo dataset.

Category: bottle

Test split:

- Good images: 6
- Defective images: 12
- Total test images: 18

Defect types:

- broken_large
- contamination
- scratch

The goal is to compare three anomaly detection models:

1. Simple pixel-difference baseline
2. PatchCore-style feature baseline
3. Rotation-augmented PatchCore-style feature baseline

---

## Clean Test Comparison

On the clean test split, all three models correctly classify all images.

| Model                     | Accuracy |  TP |  TN |  FP |  FN |
| ------------------------- | -------: | --: | --: | --: | --: |
| Simple baseline           |     100% |  12 |   6 |   0 |   0 |
| PatchCore-style           |     100% |  12 |   6 |   0 |   0 |
| PatchCore-style augmented |     100% |  12 |   6 |   0 |   0 |

Clean test result:

- 12 defective images were rejected.
- 6 good images were passed.
- No false positives.
- No false negatives.

This means the clean dataset alone is not enough to show the difference between the models.

---

## Robustness Comparison

The robustness test applies real-world visual shifts:

- Brightness changes
- Contrast changes
- Rotation changes

### Simple Baseline

The simple pixel-difference baseline fails under brightness, contrast, and rotation shifts.

Main issue:

- It compares raw pixels against an average normal reference image.
- Global lighting or pose changes create large pixel differences.
- Normal images are incorrectly rejected.

Result:

| Shift Type | Behavior |
| ---------- | -------- |
| Brightness | Fails    |
| Contrast   | Fails    |
| Rotation   | Fails    |

### PatchCore-style Feature Baseline

The PatchCore-style model uses ResNet18 patch features and a normal feature memory bank.

Result:

| Shift Type | Behavior |
| ---------- | -------- |
| Brightness | Passes   |
| Contrast   | Passes   |
| Rotation   | Fails    |

Finding:

The feature-based model is more robust to brightness and contrast changes, but it is still sensitive to rotation and pose shifts.

### Rotation-Augmented PatchCore-style

The final model augments the normal memory bank with rotated versions of normal training images.

Rotation angles used:

- -10 degrees
- -5 degrees
- 5 degrees
- 10 degrees
- 90 degrees
- 180 degrees
- 270 degrees

Result:

| Shift Type | Behavior |
| ---------- | -------- |
| Brightness | Passes   |
| Contrast   | Passes   |
| Rotation   | Passes   |

Final finding:

The rotation-augmented PatchCore-style model improves rotation robustness without losing defect detection performance on the synthetic demo test split.

---

## Main Technical Lesson

FactorySense-R demonstrates that anomaly detection models must be evaluated under real-world shifts.

A model can look perfect on clean data but fail when lighting, contrast, or pose changes.

The project shows a full improvement path:

1. Build a simple baseline.
2. Measure failure under real-world shifts.
3. Replace pixel comparison with feature-based anomaly detection.
4. Identify remaining rotation weakness.
5. Improve robustness using rotation-augmented normal memory bank.

---

## Current Best Model

The current best model is:

**Rotation-augmented PatchCore-style feature baseline**

Why:

- It keeps 100% accuracy on the clean synthetic test split.
- It stays robust under brightness changes.
- It stays robust under contrast changes.
- It stays robust under rotation changes.
- It reduces false positives on shifted normal images.

---

## Limitations

These results are based on a synthetic demo dataset.

Next steps:

- Evaluate on real MVTec AD classes.
- Compare against a standard PatchCore implementation.
- Add PaDiM as a second feature-based baseline.
- Test on more realistic industrial defects.
- Add more difficult shifts such as blur, noise, partial occlusion, and uneven illumination.
