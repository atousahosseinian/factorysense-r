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


def figure_to_pil(fig) -> Image.Image:
    """
    Convert a Matplotlib figure into a PIL image.

    Uses buffer_rgba instead of tostring_rgb for compatibility with newer
    Matplotlib versions.
    """
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    image = Image.fromarray(rgba).convert("RGB")
    return image


def anomaly_map_to_pil(anomaly_map: np.ndarray) -> Image.Image:
    """
    Convert anomaly map into a heatmap PIL image.
    """
    normalized = normalize_map(anomaly_map)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(normalized, cmap="inferno")
    ax.axis("off")
    fig.tight_layout(pad=0)

    image = figure_to_pil(fig)
    plt.close(fig)

    return image


def overlay_to_pil(
    image_path: str | Path,
    anomaly_map: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """
    Convert original image + anomaly map overlay into a PIL image.
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize((anomaly_map.shape[1], anomaly_map.shape[0]))

    normalized = normalize_map(anomaly_map)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image)
    ax.imshow(normalized, cmap="inferno", alpha=alpha)
    ax.axis("off")
    fig.tight_layout(pad=0)

    overlay = figure_to_pil(fig)
    plt.close(fig)

    return overlay


def binary_mask_to_pil(binary_mask: np.ndarray) -> Image.Image:
    """
    Convert binary mask into a PIL image.
    """
    mask = binary_mask.astype("uint8") * 255
    return Image.fromarray(mask)


def save_anomaly_heatmap(anomaly_map: np.ndarray, output_path: str | Path) -> None:
    """
    Save anomaly map as a heatmap image.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = anomaly_map_to_pil(anomaly_map)
    image.save(output_path)


def save_overlay(
    image_path: str | Path,
    anomaly_map: np.ndarray,
    output_path: str | Path,
    alpha: float = 0.45,
) -> None:
    """
    Save original image with anomaly heatmap overlay.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = overlay_to_pil(image_path, anomaly_map, alpha=alpha)
    image.save(output_path)


def save_binary_mask(binary_mask: np.ndarray, output_path: str | Path) -> None:
    """
    Save binary anomaly mask.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = binary_mask_to_pil(binary_mask)
    image.save(output_path)
