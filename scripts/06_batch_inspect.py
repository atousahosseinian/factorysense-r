import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from factorysense.data.mvtec_loader import MVTecDatasetExplorer
from factorysense.models.simple_baseline import SimpleDifferenceAnomalyDetector
from factorysense.reporting.csv_report import (
    save_inspection_report,
    summarize_inspection_report,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run batch inspection on an MVTec-style category."
    )
    parser.add_argument("--data-root", default="data/mvtec")
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--split", default="test")
    parser.add_argument("--model-path", default="models/simple_baseline_bottle.npz")
    parser.add_argument("--output", default="reports/simple_baseline_batch_report.csv")
    args = parser.parse_args()

    model_path = Path(args.model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            "Train it first with scripts/02_train_simple_baseline.py"
        )

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

    model = SimpleDifferenceAnomalyDetector.load(model_path)

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

    report_df = save_inspection_report(rows, args.output)
    summary = summarize_inspection_report(report_df)

    print("\nBatch Inspection Complete")
    print("=" * 40)
    print(f"Category: {args.category}")
    print(f"Split: {args.split}")
    print(f"Images inspected: {summary['total_images']}")
    print(f"Rejected images: {summary['rejected_images']}")
    print(f"Passed images: {summary['passed_images']}")
    print(f"Defect rate: {summary['defect_rate_percent']:.2f}%")
    print(f"Mean anomaly score: {summary['mean_anomaly_score']:.6f}")
    print(f"Mean defect area: {summary['mean_defect_area_percent']:.2f}%")

    if summary["accuracy"] is not None:
        print(f"Accuracy: {summary['accuracy']:.2%}")

    print(f"\nCSV report saved to: {args.output}")


if __name__ == "__main__":
    main()
