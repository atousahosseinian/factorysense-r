from pathlib import Path

import pandas as pd


def save_inspection_report(rows: list[dict], output_path: str | Path) -> pd.DataFrame:
    """
    Save batch inspection results as a CSV report.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    return df


def summarize_inspection_report(df: pd.DataFrame) -> dict:
    """
    Create a compact summary from a batch inspection report.
    """
    if df.empty:
        return {
            "total_images": 0,
            "rejected_images": 0,
            "passed_images": 0,
            "defect_rate_percent": 0.0,
            "accuracy": None,
            "mean_anomaly_score": 0.0,
            "mean_defect_area_percent": 0.0,
        }

    total_images = len(df)
    rejected_images = int((df["decision"] == "Reject").sum())
    passed_images = int((df["decision"] == "Pass").sum())
    defect_rate_percent = rejected_images / total_images * 100

    accuracy = None
    if "correct" in df.columns:
        accuracy = float(df["correct"].mean())

    return {
        "total_images": total_images,
        "rejected_images": rejected_images,
        "passed_images": passed_images,
        "defect_rate_percent": float(defect_rate_percent),
        "accuracy": accuracy,
        "mean_anomaly_score": float(df["anomaly_score"].mean()),
        "mean_defect_area_percent": float(df["defect_area_percent"].mean()),
    }
