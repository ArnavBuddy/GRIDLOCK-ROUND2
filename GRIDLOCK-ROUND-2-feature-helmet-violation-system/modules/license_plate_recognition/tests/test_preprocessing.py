"""Tests for preprocessing helpers."""

import numpy as np

from modules.license_plate_recognition.preprocessing import (
    adaptive_threshold,
    grayscale_image,
    resize_image,
)


def test_preprocessing_shapes() -> None:
    image = np.zeros((60, 120, 3), dtype=np.uint8)
    image[:, 40:80] = 180

    gray = grayscale_image(image)
    resized = resize_image(image, size=(28, 28))
    thresholded = adaptive_threshold(image)

    assert gray.shape == (60, 120)
    assert resized.shape == (28, 28, 3)
    assert thresholded.shape == (60, 120)
    assert thresholded.dtype == np.uint8
