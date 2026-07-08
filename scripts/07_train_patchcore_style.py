import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from factorysense.data.dataset_utils import list_image_files
from factorysense.models.patchcore_style import PatchCoreStyleAnomalyDetector


def parse_rotation_list(value: str) -> list[float]:
    if not value:
        return []

    return [float(item.strip()) for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Train a lightweight PatchCore-style anomaly detector."
    )
    parser.add_argument("--data-root", default="data/mvtec")
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--max-memory-patches", type=int, default=5000)
    parser.add_argument("--augmentation-rotations", default="")
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--threshold-multiplier", type=float, default=1.2)
    parser.add_argument("--threshold-margin", type=float, default=0.0)
    parser.add_argument("--output", default="models/patchcore_style_bottle.npz")
    args = parser.parse_args()

    train_good_dir = Path(args.data_root) / args.category / "train" / "good"
    normal_paths = list_image_files(train_good_dir)

    if not normal_paths:
        raise FileNotFoundError(f"No normal training images found in {train_good_dir}")

    augmentation_rotations = parse_rotation_list(args.augmentation_rotations)

    model = PatchCoreStyleAnomalyDetector(
        image_size=args.image_size,
        device=args.device,
        pretrained=not args.no_pretrained,
    )

    print("Training PatchCore-style model...")
    print(f"Device: {model.device}")
    print(f"Normal images: {len(normal_paths)}")
    print(f"Augmentation rotations: {augmentation_rotations}")

    model.fit(
        normal_paths,
        max_memory_patches=args.max_memory_patches,
        augmentation_rotations=augmentation_rotations,
    )

    threshold = model.calibrate_threshold(
        normal_paths,
        quantile=args.threshold_quantile,
        multiplier=args.threshold_multiplier,
        margin=args.threshold_margin,
    )

    model.save(args.output)

    print("\nPatchCore-style model trained.")
    print(f"Category: {args.category}")
    print(f"Memory bank patches: {model.memory_bank.shape[0]}")
    print(f"Base threshold: {model.base_threshold:.6f}")
    print(f"Threshold multiplier: {model.threshold_multiplier:.3f}")
    print(f"Final threshold: {threshold:.6f}")
    print(f"Augmentation rotations: {model.augmentation_rotations}")
    print(f"Saved model to: {args.output}")


if __name__ == "__main__":
    main()
