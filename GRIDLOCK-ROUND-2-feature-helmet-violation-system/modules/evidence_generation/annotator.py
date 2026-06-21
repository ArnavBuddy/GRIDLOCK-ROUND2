"""Image annotation utilities for violation evidence."""

from __future__ import annotations

import cv2
import numpy as np

try:
    from .config import DEFAULT_CONFIG, EvidenceConfig
    from .schemas import Violation
    from .utils import clamp_bbox
except ImportError:  # pragma: no cover - supports direct script execution
    from config import DEFAULT_CONFIG, EvidenceConfig
    from schemas import Violation
    from utils import clamp_bbox


def _label_for_violation(violation: Violation) -> str:
    label = violation.violation_type.replace("_", " ").title()
    if violation.confidence is not None:
        label = f"{label} {violation.confidence:.2f}"
    if violation.track_id:
        label = f"{label} #{violation.track_id}"
    return label


def _draw_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    config: EvidenceConfig,
) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        config.label_font_scale,
        config.label_thickness,
    )
    y = max(text_height + baseline + 4, y)
    cv2.rectangle(
        image,
        (x, y - text_height - baseline - 6),
        (x + text_width + 8, y + 2),
        config.label_background_color,
        thickness=-1,
    )
    cv2.putText(
        image,
        text,
        (x + 4, y - baseline - 2),
        font,
        config.label_font_scale,
        config.label_text_color,
        config.label_thickness,
        cv2.LINE_AA,
    )


def annotate_image(
    image: np.ndarray,
    violations: list[Violation],
    config: EvidenceConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """Return a copy of an image annotated with violation boxes and labels."""

    if image is None or image.size == 0:
        raise ValueError("Expected a non-empty image.")
    if not violations:
        raise ValueError("At least one violation is required for annotation.")

    annotated = image.copy()
    for violation in violations:
        x1, y1, x2, y2 = clamp_bbox(violation.bbox, annotated.shape)
        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            config.box_color,
            config.box_thickness,
        )
        _draw_label(
            annotated,
            _label_for_violation(violation),
            (x1, max(0, y1 - 4)),
            config,
        )
    return annotated
