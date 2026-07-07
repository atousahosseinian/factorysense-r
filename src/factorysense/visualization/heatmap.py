from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def normalize_map(anomaly_map: np.ndarray) -> np.ndarray:
    """
    Normalize anomaly map to [0, 1].
    """
    amap = anomaly_map.astype("float32")
    min_value = float(amap.min())
    max_value = float(amap.max())

    if max_value - min_value < 1e-8:
        return np.zeros_like(amap)

    return (amap - min_value) / (max_value - min_value)


def save_anomaly_heatmap(anomaly_map: np.ndarray, output_path: str | Path) -> None:
    """
    Save anomaly map as a heatmap image.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    normalized = normalize_map(anomaly_map)

    plt.figure(figsize=(4, 4))
    plt.imshow(normalized, cmap="inferno")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_overlay(image_path: str | Path, anomaly_map: np.ndarray, output_path: str | Path, alpha: float = 0.45) -> None:
    """
    Save original image with anomaly heatmap overlay.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    image = image.resize((anomaly_map.shape[1], anomaly_map.shape[0]))

    normalized = normalize_map(anomaly_map)

    plt.figure(figsize=(4, 4))
    plt.imshow(image)
    plt.imshow(normalized, cmap="inferno", alpha=alpha)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_binary_mask(binary_mask: np.ndarray, output_path: str | Path) -> None:
    """
    Save binary anomaly mask.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mask = (binary_mask.astype("uint8") * 255)
    Image.fromarray(mask).save(output_path)
