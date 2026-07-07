from PIL import Image, ImageEnhance
import numpy as np


def to_pil_image(image):
    """
    Convert a NumPy image or PIL image into a PIL image.
    """
    if isinstance(image, Image.Image):
        return image

    if isinstance(image, np.ndarray):
        return Image.fromarray(image.astype("uint8"))

    raise TypeError("image must be a PIL Image or a NumPy array")


def rotate_image(image, angle):
    """
    Rotate an image while keeping the same canvas size.
    """
    pil_image = to_pil_image(image)
    return pil_image.rotate(angle)


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
