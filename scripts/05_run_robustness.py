import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from factorysense.data.mvtec_loader import MVTecDatasetExplorer
from factorysense.models.simple_baseline import SimpleDifferenceAnomalyDetector
from factorysense.robustness.robustness_runner import (
    build_robustness_report,
    summarize_robustness_report,
)


def default_corruptions():
    corruptions = []

    corruptions.append(("original", 0))

    for angle in [5, -5, 10, -10, 90, 180, 270]:
        corruptions.append(("rotation", angle))

    for factor in [0.7, 0.85, 1.15, 1.3]:
        corruptions.append(("brightness", factor))

    for factor in [0.7, 0.85, 1.15, 1.3]:
        corruptions.append(("contrast", factor))

    return corruptions


def main():
    parser = argparse.ArgumentParser(
        description="Run robustness tests for FactorySense-R."
    )
    parser.add_argument("--data-root", default="data/mvtec")
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--split", default="test")
    parser.add_argument("--model-path", default="models/simple_baseline_bottle.npz")
    parser.add_argument("--include-defective", action="store_true")
    parser.add_argument("--output", default="reports/simple_baseline_robustness_report.csv")
    parser.add_argument("--summary-output", default="reports/simple_baseline_robustness_summary.csv")
    args = parser.parse_args()

    model_path = Path(args.model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Train it first with scripts/02_train_simple_baseline.py"
        )

    explorer = MVTecDatasetExplorer(args.data_root)
    df = explorer.dataframe(args.category)

    if df.empty:
        raise FileNotFoundError(
            f"No images found for category '{args.category}' in {args.data_root}"
        )

    df = df[df["split"] == args.split].copy()

    if not args.include_defective:
        df = df[df["label"] == 0].copy()

    if df.empty:
        raise FileNotFoundError("No images available for robustness testing.")

    model = SimpleDifferenceAnomalyDetector.load(model_path)

    report_df = build_robustness_report(
        model=model,
        image_records_df=df,
        corruptions=default_corruptions(),
    )

    summary_df = summarize_robustness_report(report_df)

    output_path = Path(args.output)
    summary_output_path = Path(args.summary_output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)

    report_df.to_csv(output_path, index=False)
    summary_df.to_csv(summary_output_path, index=False)

    print("\nRobustness Test Complete")
    print("=" * 40)
    print(f"Category: {args.category}")
    print(f"Split: {args.split}")
    print(f"Images tested: {len(df)}")
    print(f"Include defective: {args.include_defective}")
    print(f"Detailed report: {output_path}")
    print(f"Summary report: {summary_output_path}")

    print("\nSummary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
