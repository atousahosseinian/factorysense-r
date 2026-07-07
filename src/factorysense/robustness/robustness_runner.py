from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from PIL import Image

from factorysense.robustness.corruptions import apply_corruption


def inspect_corrupted_image(
    model,
    image_path: str | Path,
    corruption_type: str,
    corruption_value,
) -> dict:
    """
    Apply a corruption to an image and run model inspection.

    The corrupted image is saved temporarily because the current educational
    model accepts image paths.
    """
    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")
    corrupted = apply_corruption(image, corruption_type, corruption_value)

    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / f"{image_path.stem}_{corruption_type}_{corruption_value}.png"
        corrupted.save(tmp_path)

        result = model.predict(tmp_path)
        binary_mask = model.binary_mask(tmp_path)
        defect_area_percent = float(binary_mask.mean() * 100)

    predicted_label = 1 if result["decision"] == "Reject" else 0

    return {
        "image_path": str(image_path),
        "corruption_type": corruption_type,
        "corruption_value": corruption_value,
        "anomaly_score": result["anomaly_score"],
        "threshold": result["threshold"],
        "decision": result["decision"],
        "predicted_label": predicted_label,
        "defect_area_percent": defect_area_percent,
    }


def build_robustness_report(
    model,
    image_records_df: pd.DataFrame,
    corruptions: list[tuple[str, object]],
) -> pd.DataFrame:
    """
    Run robustness inspection on a dataframe of image records.
    """
    rows = []

    for record in image_records_df.to_dict("records"):
        for corruption_type, corruption_value in corruptions:
            result = inspect_corrupted_image(
                model=model,
                image_path=record["path"],
                corruption_type=corruption_type,
                corruption_value=corruption_value,
            )

            true_label = int(record["label"])
            correct = result["predicted_label"] == true_label

            rows.append(
                {
                    "original_image_path": record["path"],
                    "category": record["category"],
                    "split": record["split"],
                    "defect_type": record["defect_type"],
                    "true_label": true_label,
                    "status": record["status"],
                    **result,
                    "correct": correct,
                }
            )

    return pd.DataFrame(rows)


def summarize_robustness_report(report_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize robustness results per corruption type/value.
    """
    if report_df.empty:
        return pd.DataFrame()

    summary = (
        report_df.groupby(["corruption_type", "corruption_value"])
        .agg(
            total_images=("image_path", "count"),
            rejected_images=("decision", lambda x: int((x == "Reject").sum())),
            passed_images=("decision", lambda x: int((x == "Pass").sum())),
            accuracy=("correct", "mean"),
            mean_anomaly_score=("anomaly_score", "mean"),
            mean_defect_area_percent=("defect_area_percent", "mean"),
        )
        .reset_index()
    )

    summary["reject_rate_percent"] = (
        summary["rejected_images"] / summary["total_images"] * 100
    )

    return summary
