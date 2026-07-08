# FactorySense-R v0.1.0 Release Checklist

## Core Project

- [x] Project structure created
- [x] GitHub repository initialized
- [x] Virtual environment configured
- [x] Dependencies saved in requirements.txt
- [x] README created
- [x] Results documented

## Data

- [x] Synthetic MVTec-style demo dataset generator
- [x] MVTec-style dataset explorer
- [x] Train/test split support
- [x] Good/defective image display
- [x] Ground-truth mask display

## Models

- [x] Simple pixel-difference baseline
- [x] PatchCore-style feature baseline
- [x] Rotation-augmented PatchCore-style model
- [ ] PaDIМ baseline
- [ ] Full PatchCore implementation
- [ ] Real MVTec AD evaluation

## Dashboard

- [x] Data Explorer tab
- [x] Single-image inspection tab
- [x] Batch inspection tab
- [x] Robustness tests tab
- [x] Model comparison tab

## Evaluation

- [x] Clean test comparison
- [x] Batch inspection CSV report
- [x] Robustness testing
- [x] Robustness model comparison
- [x] Final result CSV files
- [x] Error analysis preview

## Current Best Model

Rotation-augmented PatchCore-style feature baseline.

## Current Main Result

The simple baseline fails under brightness, contrast, and rotation shifts.

PatchCore-style improves brightness and contrast robustness.

Rotation-augmented PatchCore-style improves rotation robustness while keeping clean test performance.

## Before Public Release

- [ ] Add dashboard screenshots
- [ ] Clean README formatting
- [ ] Add installation troubleshooting notes for macOS
- [ ] Add requirements check
- [ ] Add license
- [ ] Add project tags/topics on GitHub
