import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from factorysense.models.simple_baseline import SimpleDifferenceAnomalyDetector
from factorysense.visualization.heatmap import (
    save_anomaly_heatmap,
    save_binary_mask,
    save_overlay,
)


def main():
    parser = argparse.ArgumentParser(description="Run single-image anomaly prediction.")
    parser.add_argument("--model-path", default="models/simple_baseline_bottle.npz")
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--output-dir", default="outputs/simple_baseline")
    args = parser.parse_args()

    model = SimpleDifferenceAnomalyDetector.load(args.model_path)

    image_path = Path(args.image_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = model.predict(image_path)
    anomaly_map = model.anomaly_map(image_path)
    binary_mask = model.binary_mask(image_path)

    heatmap_path = output_dir / f"{image_path.stem}_heatmap.png"
    overlay_path = output_dir / f"{image_path.stem}_overlay.png"
    mask_path = output_dir / f"{image_path.stem}_mask.png"

    save_anomaly_heatmap(anomaly_map, heatmap_path)
    save_overlay(image_path, anomaly_map, overlay_path)
    save_binary_mask(binary_mask, mask_path)

    defect_area = float(binary_mask.mean() * 100)

    print("Prediction result")
    print("=" * 40)
    print(f"Image: {result['image_path']}")
    print(f"Anomaly score: {result['anomaly_score']:.6f}")
    print(f"Threshold: {result['threshold']:.6f}")
    print(f"Decision: {result['decision']}")
    print(f"Defect area: {defect_area:.2f}%")
    print(f"Heatmap: {heatmap_path}")
    print(f"Overlay: {overlay_path}")
    print(f"Binary mask: {mask_path}")


if __name__ == "__main__":
    main()
