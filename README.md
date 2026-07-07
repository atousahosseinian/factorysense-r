# FactorySense-R

**Robust Industrial Anomaly Detection Under Real-World Shifts**

FactorySense-R is an educational and practical computer vision project for industrial quality inspection.

The goal is to detect visual anomalies in industrial product images, generate anomaly heatmaps, and make Pass/Reject decisions under real-world shifts such as rotation, lighting changes, limited normal samples, and defect scale variation.

---

## Project Goals

- Build an industrial anomaly detection pipeline
- Start with training-light models suitable for a MacBook
- Generate anomaly scores and anomaly heatmaps
- Calibrate thresholds for Pass/Reject decisions
- Analyze robustness under real-world image shifts
- Build an educational Streamlit dashboard
- Keep the project clean and reproducible on GitHub

---

## Why This Project?

Most beginner computer vision projects focus on object detection.

FactorySense-R focuses on industrial anomaly detection, where the system learns what "normal" looks like and detects deviations from normal patterns.

This makes the project more relevant to real-world quality inspection problems.

---

## Planned Pipeline

```text
Image
  ↓
Preprocessing
  ↓
Anomaly Detection Model
  ↓
Anomaly Score + Heatmap
  ↓
Threshold Calibration
  ↓
Pass / Reject / Risk Level
  ↓
Dashboard + CSV Report
```

---

## Main Models

### Phase 1: PatchCore

PatchCore will be the first baseline model because it is suitable for anomaly detection without heavy training.

### Phase 2: PaDiM

PaDiM will be added as a lightweight statistical baseline.

### Optional Future Models

- EfficientAD
- WinCLIP / AnomalyCLIP

---

## Robustness Experiments

FactorySense-R will evaluate model stability under:

- Normal data diversity
- Image rotation
- Lighting changes
- Contrast changes
- Small vs large defects
- Limited normal samples
- Threshold sensitivity

---

## Project Structure

```text
factorysense-r/
├── app.py
├── configs/
├── scripts/
├── src/factorysense/
├── notebooks/
├── tests/
├── data/
├── models/
├── outputs/
├── reports/
└── assets/
```

---

## Current Status

- [x] Project architecture planned
- [x] Repository structure initialized
- [ ] Data explorer
- [ ] PatchCore baseline
- [ ] Threshold calibration
- [ ] Robustness experiments
- [ ] Streamlit dashboard
- [ ] Model comparison
- [ ] Final GitHub presentation

---

## Educational Roadmap

This project is built step by step.

Each phase includes code, explanation, and Git commits so the developer can learn the concepts while building the project.

---

## Limitations

- The first version will use public datasets only.
- The project starts with training-light models due to MacBook hardware constraints.
- Real factory deployment requires additional calibration with real production data.
