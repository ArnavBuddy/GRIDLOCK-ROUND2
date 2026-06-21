"""Tests for image annotation."""

import numpy as np

from modules.evidence_generation.annotator import annotate_image
from modules.evidence_generation.schemas import Violation


def test_annotate_image_draws_violation_box() -> None:
    image = np.zeros((100, 140, 3), dtype=np.uint8)
    violation = Violation(
        violation_type="red_light_jump",
        bbox=(20, 20, 80, 70),
        confidence=0.92,
    )

    annotated = annotate_image(image, [violation])

    assert annotated.shape == image.shape
    assert not np.array_equal(annotated, image)
    assert annotated[70, 80].tolist() == [0, 0, 255]
