"""Tests for end-to-end evidence generation."""

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from modules.evidence_generation.config import DEFAULT_CONFIG
from modules.evidence_generation.generator import generate_evidence
from modules.evidence_generation.schemas import Violation


def test_generate_evidence_writes_image_and_metadata(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image = np.zeros((90, 120, 3), dtype=np.uint8)
    cv2.imwrite(str(image_path), image)

    config = replace(
        DEFAULT_CONFIG,
        output_dir=tmp_path / "outputs",
        manifest_filename="manifest.jsonl",
    )
    record = generate_evidence(
        image_path=image_path,
        violations=[
            Violation(
                violation_type="wrong_lane",
                bbox=(10, 10, 60, 60),
                confidence=0.87,
            )
        ],
        output_dir=config.output_dir,
        frame_id="frame-1",
        camera_id="cam-1",
        config=config,
    )

    assert Path(record.annotated_image).exists()
    assert Path(record.metadata_path).exists()
    assert (config.output_dir / config.manifest_filename).exists()
    assert record.camera_id == "cam-1"
    assert record.violations[0].violation_type == "wrong_lane"
