"""Tests for contour-based character segmentation."""

import cv2
import numpy as np

from modules.license_plate_recognition.character_segmenter import segment_characters


def test_segment_characters_from_synthetic_plate() -> None:
    plate = np.full((75, 333, 3), 255, dtype=np.uint8)
    cv2.putText(
        plate,
        "AB12",
        (35, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.7,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )

    characters = segment_characters(plate)

    assert len(characters) >= 4
    assert all(character.shape == (28, 28) for character in characters)
