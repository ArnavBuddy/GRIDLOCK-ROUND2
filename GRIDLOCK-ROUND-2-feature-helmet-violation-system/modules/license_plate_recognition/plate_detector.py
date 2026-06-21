"""License plate localization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from .config import DEFAULT_CONFIG, LPRConfig
    from .preprocessing import enhance_image, grayscale_image
    from .utils import clamp_bbox, get_logger
except ImportError:  # pragma: no cover - supports direct script execution
    from config import DEFAULT_CONFIG, LPRConfig
    from preprocessing import enhance_image, grayscale_image
    from utils import clamp_bbox, get_logger


logger = get_logger(__name__)
DetectionResult = dict[str, Any]


def _empty_detection() -> DetectionResult:
    return {"bbox": [], "confidence": 0.0}


def _cascade_file(config: LPRConfig) -> Path | None:
    if config.cascade_path.exists():
        return config.cascade_path

    fallback = Path(cv2.data.haarcascades) / config.fallback_cascade_name
    if fallback.exists():
        return fallback
    return None


def _load_cascade(config: LPRConfig) -> cv2.CascadeClassifier | None:
    cascade_file = _cascade_file(config)
    if cascade_file is None:
        logger.debug("No Haar cascade file found for plate detection.")
        return None

    cascade = cv2.CascadeClassifier(str(cascade_file))
    if cascade.empty():
        logger.warning("Unable to load cascade classifier: %s", cascade_file)
        return None
    return cascade


def _bbox_score(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, ...],
    config: LPRConfig,
) -> float:
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    aspect = width / height
    image_area = max(1, image_shape[0] * image_shape[1])
    area_ratio = (width * height) / image_area

    if not config.min_plate_aspect <= aspect <= config.max_plate_aspect:
        return 0.0
    if not config.min_plate_area_ratio <= area_ratio <= config.max_plate_area_ratio:
        return 0.0

    aspect_score = 1.0 - min(
        abs(aspect - config.expected_plate_aspect) / config.expected_plate_aspect,
        1.0,
    )
    area_score = min(area_ratio / config.preferred_plate_area_ratio, 1.0)
    return float((aspect_score * 0.7) + (area_score * 0.3))


def _cascade_candidates(
    image: np.ndarray,
    config: LPRConfig,
) -> list[DetectionResult]:
    cascade = _load_cascade(config)
    if cascade is None:
        return []

    gray = grayscale_image(image)
    rectangles = cascade.detectMultiScale(
        gray,
        scaleFactor=config.cascade_scale_factor,
        minNeighbors=config.cascade_min_neighbors,
        minSize=config.cascade_min_size,
    )

    candidates: list[DetectionResult] = []
    for x, y, width, height in rectangles:
        bbox = clamp_bbox((x, y, x + width, y + height), image.shape)
        score = _bbox_score(tuple(bbox), image.shape, config)
        if score > 0:
            candidates.append(
                {
                    "bbox": bbox,
                    "confidence": round(score * config.cascade_confidence_scale, 4),
                }
            )
    return candidates


def _contour_candidates(
    image: np.ndarray,
    config: LPRConfig,
) -> list[DetectionResult]:
    gray = grayscale_image(enhance_image(image, config=config))
    filtered = cv2.bilateralFilter(
        gray,
        config.bilateral_filter_diameter,
        config.bilateral_sigma_color,
        config.bilateral_sigma_space,
    )
    edges = cv2.Canny(filtered, config.canny_threshold1, config.canny_threshold2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config.plate_morph_kernel)
    closed = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=config.plate_morph_iterations,
    )

    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[
        : config.max_plate_candidates
    ]

    candidates: list[DetectionResult] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        bbox = clamp_bbox((x, y, x + width, y + height), image.shape)
        score = _bbox_score(tuple(bbox), image.shape, config)
        if score <= 0:
            continue
        candidates.append(
            {
                "bbox": bbox,
                "confidence": round(score * config.contour_confidence_scale, 4),
            }
        )
    return candidates


def detect_license_plate(
    image: np.ndarray,
    config: LPRConfig = DEFAULT_CONFIG,
) -> DetectionResult:
    """Detect the most likely license plate bounding box in an image."""

    if image is None or image.size == 0:
        raise ValueError("Expected a non-empty image for plate detection.")

    enhanced = enhance_image(image, config=config)
    candidates = _cascade_candidates(enhanced, config)
    candidates.extend(_contour_candidates(enhanced, config))
    if not candidates:
        logger.warning("No license plate candidates were detected.")
        return _empty_detection()

    best = max(candidates, key=lambda item: float(item["confidence"]))
    if best["confidence"] < config.confidence_threshold:
        logger.warning("Best plate candidate is below confidence threshold.")
    return best


def crop_plate(
    image: np.ndarray,
    detection: DetectionResult | None = None,
    config: LPRConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """Crop the detected plate region from an image."""

    result = detection or detect_license_plate(image, config=config)
    bbox = result.get("bbox", [])
    if len(bbox) != 4:
        raise ValueError("Cannot crop plate because no valid bbox was detected.")

    x1, y1, x2, y2 = clamp_bbox(bbox, image.shape)
    return image[y1:y2, x1:x2].copy()


def draw_plate_bbox(
    image: np.ndarray,
    detection: DetectionResult,
    label: str | None = None,
) -> np.ndarray:
    """Draw the detected plate bounding box and optional text label."""

    annotated = image.copy()
    bbox = detection.get("bbox", [])
    if len(bbox) != 4:
        return annotated

    x1, y1, x2, y2 = clamp_bbox(bbox, image.shape)
    color = (51, 181, 155)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated
