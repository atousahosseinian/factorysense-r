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
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
        }

    total_images = len(df)
    rejected_images = int((df["decision"] == "Reject").sum())
    passed_images = int((df["decision"] == "Pass").sum())
    defect_rate_percent = rejected_images / total_images * 100

    accuracy = None
    tp = tn = fp = fn = 0

    if {"true_label", "predicted_label"}.issubset(df.columns):
        true = df["true_label"].astype(int)
        pred = df["predicted_label"].astype(int)

        tp = int(((true == 1) & (pred == 1)).sum())
        tn = int(((true == 0) & (pred == 0)).sum())
        fp = int(((true == 0) & (pred == 1)).sum())
        fn = int(((true == 1) & (pred == 0)).sum())

        accuracy = float((true == pred).mean())

    return {
        "total_images": total_images,
        "rejected_images": rejected_images,
        "passed_images": passed_images,
        "defect_rate_percent": float(defect_rate_percent),
        "accuracy": accuracy,
        "mean_anomaly_score": float(df["anomaly_score"].mean()),
        "mean_defect_area_percent": float(df["defect_area_percent"].mean()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
