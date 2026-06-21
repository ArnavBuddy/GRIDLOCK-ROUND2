"""Tests for evidence data schemas."""

import pytest

from modules.evidence_generation.schemas import Violation


def test_violation_from_dict_accepts_type_alias() -> None:
    violation = Violation.from_dict(
        {
            "type": "helmet_missing",
            "bbox": [1, 2, 30, 40],
            "confidence": 0.91,
        }
    )

    assert violation.violation_type == "helmet_missing"
    assert violation.bbox == (1, 2, 30, 40)
    assert violation.confidence == 0.91


def test_violation_from_dict_requires_bbox() -> None:
    with pytest.raises(ValueError, match="bbox"):
        Violation.from_dict({"violation_type": "red_light_jump"})
