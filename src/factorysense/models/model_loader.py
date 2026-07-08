from pathlib import Path

from factorysense.models.simple_baseline import SimpleDifferenceAnomalyDetector
from factorysense.models.patchcore_style import PatchCoreStyleAnomalyDetector


SUPPORTED_MODELS = {
    "simple": "Simple difference baseline",
    "patchcore_style": "PatchCore-style feature baseline",
    "patchcore_style_aug": "Rotation-augmented PatchCore-style feature baseline",
}


def load_detector(model_name: str, model_path: str | Path, device: str = "auto"):
    """
    Load an anomaly detector by name.

    Supported models:
    - simple
    - patchcore_style
    - patchcore_style_aug
    """
    if model_name == "simple":
        return SimpleDifferenceAnomalyDetector.load(model_path)

    if model_name in {"patchcore_style", "patchcore_style_aug"}:
        return PatchCoreStyleAnomalyDetector.load(model_path, device=device)

    supported = ", ".join(SUPPORTED_MODELS.keys())
    raise ValueError(f"Unsupported model_name: {model_name}. Supported: {supported}")
