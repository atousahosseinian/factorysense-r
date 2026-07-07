from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def load_image_array(path: str | Path, image_size: int = 256) -> np.ndarray:
    """
    Load image, convert to RGB, resize, and normalize to [0, 1].
    """
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size))
    array = np.asarray(image).astype("float32") / 255.0
    return array


class SimpleDifferenceAnomalyDetector:
    """
    A lightweight educational anomaly detector.

    It builds a reference image from normal training images and detects anomalies
    by measuring pixel-level differences from that reference.

    This is not a strong industrial model, but it teaches the full anomaly
    detection pipeline:
    reference normal pattern -> anomaly map -> score -> threshold -> decision.
    """

    def __init__(self, image_size: int = 256):
        self.image_size = image_size
        self.reference_image: np.ndarray | None = None
        self.threshold: float | None = None

    def fit(self, normal_image_paths: Iterable[str | Path]) -> None:
        arrays = [
            load_image_array(path, image_size=self.image_size)
            for path in normal_image_paths
        ]

        if not arrays:
            raise ValueError("No normal images were provided for fitting.")

        self.reference_image = np.mean(np.stack(arrays, axis=0), axis=0)

    def anomaly_map(self, image_path: str | Path) -> np.ndarray:
        if self.reference_image is None:
            raise ValueError("Model is not fitted yet.")

        image = load_image_array(image_path, image_size=self.image_size)

        diff = np.abs(image - self.reference_image)

        # Convert RGB difference into a single-channel anomaly map.
        anomaly = np.mean(diff, axis=2)

        return anomaly.astype("float32")

    def score(self, image_path: str | Path) -> float:
        amap = self.anomaly_map(image_path)

        # Use high-percentile difference instead of mean.
        # This makes the score more sensitive to local defects.
        return float(np.quantile(amap, 0.99))

    def calibrate_threshold(self, normal_image_paths: Iterable[str | Path], quantile: float = 0.95) -> float:
        scores = [self.score(path) for path in normal_image_paths]

        if not scores:
            raise ValueError("No normal images were provided for threshold calibration.")

        self.threshold = float(np.quantile(scores, quantile))
        return self.threshold

    def predict(self, image_path: str | Path) -> dict:
        if self.threshold is None:
            raise ValueError("Threshold is not calibrated yet.")

        score = self.score(image_path)
        decision = "Reject" if score >= self.threshold else "Pass"

        return {
            "image_path": str(image_path),
            "anomaly_score": score,
            "threshold": self.threshold,
            "decision": decision,
        }

    def binary_mask(self, image_path: str | Path, threshold: float | None = None) -> np.ndarray:
        amap = self.anomaly_map(image_path)
        used_threshold = self.threshold if threshold is None else threshold

        if used_threshold is None:
            raise ValueError("Threshold is not available.")

        return (amap >= used_threshold).astype("uint8")

    def save(self, path: str | Path) -> None:
        if self.reference_image is None:
            raise ValueError("Model is not fitted yet.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            path,
            image_size=self.image_size,
            reference_image=self.reference_image,
            threshold=-1.0 if self.threshold is None else self.threshold,
        )

    @classmethod
    def load(cls, path: str | Path) -> "SimpleDifferenceAnomalyDetector":
        data = np.load(path)
        image_size = int(data["image_size"])

        model = cls(image_size=image_size)
        model.reference_image = data["reference_image"]

        threshold = float(data["threshold"])
        model.threshold = None if threshold < 0 else threshold

        return model
