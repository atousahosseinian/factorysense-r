from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from factorysense.data.dataset_utils import (
    discover_mvtec_categories,
    list_image_files,
)


@dataclass
class ImageRecord:
    path: Path
    category: str
    split: str
    label: int
    defect_type: str
    mask_path: Optional[Path]


class MVTecDatasetExplorer:
    """
    Lightweight explorer for MVTec AD-style datasets.

    Expected structure:
      data/mvtec/
        bottle/
          train/
            good/
          test/
            good/
            broken_large/
            contamination/
          ground_truth/
            broken_large/
              image_mask.png
    """

    def __init__(self, dataset_root: str | Path):
        self.dataset_root = Path(dataset_root)

    def categories(self) -> list[str]:
        return discover_mvtec_categories(self.dataset_root)

    def category_path(self, category: str) -> Path:
        return self.dataset_root / category

    def _find_mask_path(self, image_path: Path, category: str, defect_type: str) -> Optional[Path]:
        if defect_type == "good":
            return None

        mask_name = f"{image_path.stem}_mask.png"
        mask_path = self.dataset_root / category / "ground_truth" / defect_type / mask_name

        return mask_path if mask_path.exists() else None

    def records(self, category: str) -> list[ImageRecord]:
        category_dir = self.category_path(category)

        if not category_dir.exists():
            return []

        records: list[ImageRecord] = []

        for split in ["train", "test"]:
            split_dir = category_dir / split

            if not split_dir.exists():
                continue

            for defect_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
                defect_type = defect_dir.name
                label = 0 if defect_type == "good" else 1

                for image_path in list_image_files(defect_dir):
                    mask_path = self._find_mask_path(
                        image_path=image_path,
                        category=category,
                        defect_type=defect_type,
                    )

                    records.append(
                        ImageRecord(
                            path=image_path,
                            category=category,
                            split=split,
                            label=label,
                            defect_type=defect_type,
                            mask_path=mask_path,
                        )
                    )

        return records

    def dataframe(self, category: str) -> pd.DataFrame:
        rows = []

        for record in self.records(category):
            rows.append(
                {
                    "path": str(record.path),
                    "category": record.category,
                    "split": record.split,
                    "label": record.label,
                    "status": "defective" if record.label == 1 else "good",
                    "defect_type": record.defect_type,
                    "mask_path": str(record.mask_path) if record.mask_path else None,
                }
            )

        return pd.DataFrame(rows)

    def summary(self, category: str) -> pd.DataFrame:
        df = self.dataframe(category)

        if df.empty:
            return pd.DataFrame(
                columns=["split", "defect_type", "status", "count"]
            )

        return (
            df.groupby(["split", "defect_type", "status"])
            .size()
            .reset_index(name="count")
            .sort_values(["split", "status", "defect_type"])
        )
