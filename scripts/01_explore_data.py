import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from factorysense.data.mvtec_loader import MVTecDatasetExplorer


def main():
    parser = argparse.ArgumentParser(description="Explore an MVTec AD-style dataset.")
    parser.add_argument("--data-root", default="data/mvtec", help="Path to MVTec dataset root.")
    parser.add_argument("--category", default=None, help="MVTec category name, e.g. bottle.")
    args = parser.parse_args()

    explorer = MVTecDatasetExplorer(args.data_root)
    categories = explorer.categories()

    print("\nFactorySense-R Data Explorer")
    print("=" * 40)
    print(f"Dataset root: {args.data_root}")

    if not categories:
        print("\nNo MVTec categories found yet.")
        print("\nExpected structure:")
        print("data/mvtec/bottle/train/good/*.png")
        print("data/mvtec/bottle/test/good/*.png")
        print("data/mvtec/bottle/test/<defect_type>/*.png")
        print("data/mvtec/bottle/ground_truth/<defect_type>/*_mask.png")
        return

    print("\nAvailable categories:")
    for category in categories:
        print(f"- {category}")

    category = args.category or categories[0]

    if category not in categories:
        raise ValueError(f"Category '{category}' not found. Available: {categories}")

    print(f"\nSelected category: {category}")
    print("\nSummary:")
    print(explorer.summary(category).to_string(index=False))


if __name__ == "__main__":
    main()
