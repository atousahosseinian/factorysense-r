import numpy as np
from PIL import Image

from factorysense.robustness.corruptions import (
    rotate_image,
    adjust_brightness,
    adjust_contrast,
)


def test_rotate_image_keeps_size():
    image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    rotated = rotate_image(image, 90)

    assert rotated.size == image.size


def test_adjust_brightness_returns_image():
    image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    result = adjust_brightness(image, 1.2)

    assert isinstance(result, Image.Image)


def test_adjust_contrast_returns_image():
    image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    result = adjust_contrast(image, 1.2)

    assert isinstance(result, Image.Image)
