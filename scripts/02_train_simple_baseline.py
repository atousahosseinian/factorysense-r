import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from factorysense.data.dataset_utils import list_image_files
from factorysense.models.simple_baseline import SimpleDifferenceAnomalyDetector


def main():
    parser = argparse.ArgumentParser(description="Train a simple difference-based anomaly baseline.")
    parser.add_argument("--data-root", default="data/mvtec")
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--threshold-multiplier", type=float, default=1.2)
    parser.add_argument("--threshold-margin", type=float, default=0.0)
    parser.add_argument("--output", default="models/simple_baseline_bottle.npz")
    args = parser.parse_args()

    train_good_dir = Path(args.data_root) / args.category / "train" / "good"
    normal_paths = list_image_files(train_good_dir)

    if not normal_paths:
        raise FileNotFoundError(f"No normal training images found in {train_good_dir}")

    model = SimpleDifferenceAnomalyDetector(image_size=args.image_size)
    model.fit(normal_paths)

    threshold = model.calibrate_threshold(
        normal_paths,
        quantile=args.threshold_quantile,
        multiplier=args.threshold_multiplier,
        margin=args.threshold_margin,
    )

    model.save(args.output)

    print("Simple baseline trained.")
    print(f"Category: {args.category}")
    print(f"Normal train images: {len(normal_paths)}")
    print(f"Base threshold: {model.base_threshold:.6f}")
    print(f"Threshold multiplier: {model.threshold_multiplier:.3f}")
    print(f"Threshold margin: {model.threshold_margin:.6f}")
    print(f"Final threshold: {threshold:.6f}")
    print(f"Saved model to: {args.output}")


if __name__ == "__main__":
    main()
