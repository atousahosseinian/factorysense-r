from pathlib import Path
from typing import List

from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_image_files(directory: str | Path) -> List[Path]:
    """
    Recursively list image files inside a directory.
    """
    directory = Path(directory)

    if not directory.exists():
        return []

    files = [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(files)


def read_rgb_image(path: str | Path) -> Image.Image:
    """
    Read an image as RGB PIL image.
    """
    return Image.open(path).convert("RGB")


def is_mvtec_category_dir(path: str | Path) -> bool:
    """
    Check whether a folder looks like an MVTec AD category folder.
    Expected structure:
      category/
        train/
        test/
    """
    path = Path(path)
    return (path / "train").exists() and (path / "test").exists()


def discover_mvtec_categories(dataset_root: str | Path) -> list[str]:
    """
    Discover available MVTec AD categories in the dataset root.
    """
    dataset_root = Path(dataset_root)

    if not dataset_root.exists():
        return []

    categories = [
        path.name
        for path in dataset_root.iterdir()
        if path.is_dir() and is_mvtec_category_dir(path)
    ]

    return sorted(categories)
