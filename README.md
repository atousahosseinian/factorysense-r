# FactorySense-R

**Robust Industrial Anomaly Detection Under Real-World Shifts**

FactorySense-R is a computer vision project for industrial anomaly detection and robustness analysis.

The project demonstrates how anomaly detection models can perform well on clean test images but fail under real-world visual shifts such as brightness, contrast, and rotation changes.

---

## Dashboard Preview

### Data Explorer

![Data Explorer](assets/screenshots/data_explorer.png)

### Batch Inspection

![Batch Inspection](assets/screenshots/batch_inspection.png)

### Robustness Model Comparison

![Robustness Model Comparison](assets/screenshots/model_comparison.png)

---

## Main Features

- Synthetic MVTec-style demo dataset generator
- MVTec-style dataset explorer
- Simple pixel-difference anomaly baseline
- PatchCore-style feature baseline using ResNet18 patch features
- Rotation-augmented PatchCore-style model
- Single-image inspection dashboard
- Batch inspection dashboard
- Robustness testing for brightness, contrast, and rotation shifts
- Model comparison dashboard
- CSV result export
- Environment check script

---

## Current Result

On the synthetic bottle test split:

| Model                              | Clean Test | Brightness | Contrast | Rotation |
| ---------------------------------- | ---------: | ---------: | -------: | -------: |
| Simple baseline                    |       100% |      Fails |    Fails |    Fails |
| PatchCore-style                    |       100% |     Passes |   Passes |    Fails |
| Rotation-augmented PatchCore-style |       100% |     Passes |   Passes |   Passes |

The current best model is the **rotation-augmented PatchCore-style feature baseline**.

---

## Result Files

The main result CSV files are available here:

- [Clean test model comparison](assets/results/bottle_test_model_comparison_summary.csv)
- [Robustness model comparison](assets/results/bottle_test_robustness_model_comparison_summary.csv)

Detailed result documentation:

- [FactorySense-R Results](docs/results.md)

---

## Quick Start

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Check the environment:

```bash
python scripts/11_check_environment.py
```

Create the synthetic demo dataset:

```bash
python scripts/00_create_demo_dataset.py
```

Train the simple baseline:

```bash
python scripts/02_train_simple_baseline.py --data-root data/mvtec --category bottle
```

Train the PatchCore-style baseline:

```bash
python scripts/07_train_patchcore_style.py \
  --data-root data/mvtec \
  --category bottle \
  --device cpu \
  --max-memory-patches 3000
```

Train the rotation-augmented PatchCore-style model:

```bash
python scripts/07_train_patchcore_style.py \
  --data-root data/mvtec \
  --category bottle \
  --device cpu \
  --max-memory-patches 8000 \
  --augmentation-rotations=-10,-5,5,10,90,180,270 \
  --output models/patchcore_style_aug_bottle.npz
```

Run the dashboard:

```bash
streamlit run app.py
```

---

## Project Structure

```text
factorysense-r/
├── app.py
├── assets/
│   ├── results/
│   └── screenshots/
├── configs/
├── docs/
├── scripts/
├── src/factorysense/
├── tests/
├── requirements.txt
└── README.md
```

---

## Documentation

- [Results](docs/results.md)
- [Release Checklist](docs/release_checklist.md)

---

## Limitations

This release uses a synthetic demo dataset.

Future work includes:

- Evaluation on real MVTec AD classes
- PaDiM baseline
- Full PatchCore implementation
- More difficult shifts such as blur, noise, occlusion, and uneven illumination

---

## Release

Current release:

- `v0.1.0`

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
