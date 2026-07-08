import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from factorysense.data.mvtec_loader import MVTecDatasetExplorer
from factorysense.models.model_loader import load_detector
from factorysense.reporting.csv_report import summarize_inspection_report


def inspect_dataframe(model, df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    rows = []

    for row in df.to_dict("records"):
        image_path = row["path"]

        result = model.predict(image_path)
        binary_mask = model.binary_mask(image_path)

        defect_area_percent = float(binary_mask.mean() * 100)

        predicted_label = 1 if result["decision"] == "Reject" else 0
        true_label = int(row["label"])
        correct = predicted_label == true_label

        rows.append(
            {
                "model_name": model_name,
                "image_path": image_path,
                "category": row["category"],
                "split": row["split"],
                "defect_type": row["defect_type"],
                "true_label": true_label,
                "status": row["status"],
                "anomaly_score": result["anomaly_score"],
                "threshold": result["threshold"],
                "decision": result["decision"],
                "predicted_label": predicted_label,
                "defect_area_percent": defect_area_percent,
                "correct": correct,
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compare FactorySense-R anomaly detection models."
    )
    parser.add_argument("--data-root", default="data/mvtec")
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--simple-model-path", default="models/simple_baseline_bottle.npz")
    parser.add_argument("--patchcore-model-path", default="models/patchcore_style_bottle.npz")
    parser.add_argument("--output", default="reports/model_comparison_report.csv")
    parser.add_argument("--summary-output", default="reports/model_comparison_summary.csv")
    args = parser.parse_args()

    explorer = MVTecDatasetExplorer(args.data_root)
    df = explorer.dataframe(args.category)

    if df.empty:
        raise FileNotFoundError(
            f"No images found for category '{args.category}' in {args.data_root}"
        )

    df = df[df["split"] == args.split].copy()

    if df.empty:
        raise FileNotFoundError(
            f"No images found for split '{args.split}' in category '{args.category}'"
        )

    model_specs = [
        ("simple", args.simple_model_path),
        ("patchcore_style", args.patchcore_model_path),
    ]

    all_reports = []
    summary_rows = []

    for model_name, model_path in model_specs:
        model_path = Path(model_path)

        if not model_path.exists():
            print(f"Skipping {model_name}: model file not found at {model_path}")
            continue

        print(f"\nRunning model: {model_name}")
        model = load_detector(model_name, model_path, device=args.device)

        report_df = inspect_dataframe(
            model=model,
            df=df,
            model_name=model_name,
        )

        summary = summarize_inspection_report(report_df)
        summary["model_name"] = model_name
        summary_rows.append(summary)

        all_reports.append(report_df)

    if not all_reports:
        raise FileNotFoundError("No model reports were generated. Train at least one model first.")

    comparison_df = pd.concat(all_reports, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    # Put model_name first
    summary_columns = ["model_name"] + [
        col for col in summary_df.columns if col != "model_name"
    ]
    summary_df = summary_df[summary_columns]

    output_path = Path(args.output)
    summary_output_path = Path(args.summary_output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)

    comparison_df.to_csv(output_path, index=False)
    summary_df.to_csv(summary_output_path, index=False)

    print("\nModel Comparison Complete")
    print("=" * 40)
    print(f"Category: {args.category}")
    print(f"Split: {args.split}")
    print(f"Images per model: {len(df)}")
    print(f"Detailed report: {output_path}")
    print(f"Summary report: {summary_output_path}")

    print("\nSummary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
