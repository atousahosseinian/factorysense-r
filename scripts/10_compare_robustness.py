import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from factorysense.data.mvtec_loader import MVTecDatasetExplorer
from factorysense.models.model_loader import load_detector
from factorysense.robustness.robustness_runner import (
    build_robustness_report,
    summarize_robustness_report,
)


def default_corruptions():
    corruptions = [("original", 0.0)]

    for angle in [5, -5, 10, -10, 90, 180, 270]:
        corruptions.append(("rotation", angle))

    for factor in [0.7, 0.85, 1.15, 1.3]:
        corruptions.append(("brightness", factor))

    for factor in [0.7, 0.85, 1.15, 1.3]:
        corruptions.append(("contrast", factor))

    return corruptions


def main():
    parser = argparse.ArgumentParser(
        description="Compare robustness of FactorySense-R anomaly detection models."
    )
    parser.add_argument("--data-root", default="data/mvtec")
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--normal-only", action="store_true")
    parser.add_argument("--simple-model-path", default="models/simple_baseline_bottle.npz")
    parser.add_argument("--patchcore-model-path", default="models/patchcore_style_bottle.npz")
    parser.add_argument("--output", default="reports/robustness_model_comparison_report.csv")
    parser.add_argument("--summary-output", default="reports/robustness_model_comparison_summary.csv")
    args = parser.parse_args()

    explorer = MVTecDatasetExplorer(args.data_root)
    df = explorer.dataframe(args.category)

    if df.empty:
        raise FileNotFoundError(
            f"No images found for category '{args.category}' in {args.data_root}"
        )

    df = df[df["split"] == args.split].copy()

    if args.normal_only:
        df = df[df["label"] == 0].copy()

    if df.empty:
        raise FileNotFoundError("No images available for robustness comparison.")

    model_specs = [
        ("simple", args.simple_model_path),
        ("patchcore_style", args.patchcore_model_path),
    ]

    all_reports = []
    all_summaries = []

    for model_name, model_path in model_specs:
        model_path = Path(model_path)

        if not model_path.exists():
            print(f"Skipping {model_name}: model file not found at {model_path}")
            continue

        print(f"\nRunning robustness for model: {model_name}")

        model = load_detector(
            model_name=model_name,
            model_path=model_path,
            device=args.device,
        )

        report_df = build_robustness_report(
            model=model,
            image_records_df=df,
            corruptions=default_corruptions(),
        )

        summary_df = summarize_robustness_report(report_df)

        report_df.insert(0, "model_name", model_name)
        summary_df.insert(0, "model_name", model_name)

        all_reports.append(report_df)
        all_summaries.append(summary_df)

    if not all_reports:
        raise FileNotFoundError("No robustness reports were generated.")

    comparison_report = pd.concat(all_reports, ignore_index=True)
    comparison_summary = pd.concat(all_summaries, ignore_index=True)

    output_path = Path(args.output)
    summary_output_path = Path(args.summary_output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)

    comparison_report.to_csv(output_path, index=False)
    comparison_summary.to_csv(summary_output_path, index=False)

    print("\nRobustness Model Comparison Complete")
    print("=" * 50)
    print(f"Category: {args.category}")
    print(f"Split: {args.split}")
    print(f"Normal only: {args.normal_only}")
    print(f"Images per model: {len(df)}")
    print(f"Detailed report: {output_path}")
    print(f"Summary report: {summary_output_path}")

    print("\nSummary:")
    print(comparison_summary.to_string(index=False))


if __name__ == "__main__":
    main()
