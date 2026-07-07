from PIL import Image, ImageEnhance
import numpy as np


def to_pil_image(image):
    """
    Convert a NumPy image or PIL image into a PIL image.
    """
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, np.ndarray):
        return Image.fromarray(image.astype("uint8")).convert("RGB")

    raise TypeError("image must be a PIL Image or a NumPy array")


def rotate_image(image, angle):
    """
    Rotate an image while keeping the same canvas size.

    The empty background is filled using the image corner color.
    This reduces artificial black-corner artifacts for small rotations.
    """
    pil_image = to_pil_image(image)
    fill_color = pil_image.getpixel((0, 0))

    return pil_image.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=fill_color,
    )


def adjust_brightness(image, factor):
    """
    Adjust image brightness.

    factor < 1.0 makes the image darker.
    factor > 1.0 makes the image brighter.
    """
    pil_image = to_pil_image(image)
    enhancer = ImageEnhance.Brightness(pil_image)
    return enhancer.enhance(factor)


def adjust_contrast(image, factor):
    """
    Adjust image contrast.
    """
    pil_image = to_pil_image(image)
    enhancer = ImageEnhance.Contrast(pil_image)
    return enhancer.enhance(factor)


def apply_corruption(image, corruption_type, value):
    """
    Apply one robustness corruption to an image.
    """
    if corruption_type == "rotation":
        return rotate_image(image, angle=float(value))

    if corruption_type == "brightness":
        return adjust_brightness(image, factor=float(value))

    if corruption_type == "contrast":
        return adjust_contrast(image, factor=float(value))

    if corruption_type == "original":
        return to_pil_image(image)

    raise ValueError(f"Unsupported corruption type: {corruption_type}")
