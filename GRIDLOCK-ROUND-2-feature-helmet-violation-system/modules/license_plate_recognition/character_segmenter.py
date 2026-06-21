"""Character segmentation from cropped license plate images."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

try:
    from .config import DEFAULT_CONFIG, LPRConfig
    from .preprocessing import adaptive_threshold, enhance_image, grayscale_image
    from .utils import get_logger
except ImportError:  # pragma: no cover - supports direct script execution
    from config import DEFAULT_CONFIG, LPRConfig
    from preprocessing import adaptive_threshold, enhance_image, grayscale_image
    from utils import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class CharacterRegion:
    """A segmented character image and its bounding box."""

    bbox: tuple[int, int, int, int]
    image: np.ndarray


def _prepare_plate(
    plate_image: np.ndarray,
    config: LPRConfig,
) -> np.ndarray:
    plate = cv2.resize(
        plate_image,
        config.plate_size,
        interpolation=cv2.INTER_AREA,
    )
    return enhance_image(plate, config=config)


def _remove_border(binary: np.ndarray, config: LPRConfig) -> np.ndarray:
    cleaned = binary.copy()
    border = config.plate_border_size
    cleaned[:border, :] = 0
    cleaned[-border:, :] = 0
    cleaned[:, :border] = 0
    cleaned[:, -border:] = 0
    return cleaned


def _threshold_variants(
    plate_image: np.ndarray,
    config: LPRConfig,
) -> list[np.ndarray]:
    gray = grayscale_image(plate_image)
    blurred = cv2.GaussianBlur(gray, config.threshold_blur_kernel, 0)
    _, otsu_inverse = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    adaptive_inverse = adaptive_threshold(
        blurred,
        invert=True,
        config=config,
    )
    variants = [otsu_inverse, adaptive_inverse]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config.morph_kernel_size)
    cleaned_variants = []
    for binary in variants:
        cleaned = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel,
            iterations=config.character_morph_iterations,
        )
        cleaned_variants.append(_remove_border(cleaned, config))
    return cleaned_variants


def _iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[0] + first[2], second[0] + second[2])
    y2 = min(first[1] + first[3], second[1] + second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = first[2] * first[3]
    second_area = second[2] * second[3]
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _format_character(
    binary: np.ndarray,
    bbox: tuple[int, int, int, int],
    config: LPRConfig,
) -> np.ndarray:
    x, y, width, height = bbox
    character = binary[y : y + height, x : x + width]
    return cv2.resize(
        character,
        config.image_size,
        interpolation=cv2.INTER_AREA,
    )


def _deduplicate_boxes(
    boxes: list[tuple[int, int, int, int]],
    config: LPRConfig,
) -> list[tuple[int, int, int, int]]:
    boxes = sorted(boxes, key=lambda box: box[2] * box[3], reverse=True)
    selected: list[tuple[int, int, int, int]] = []
    for box in boxes:
        if all(_iou(box, kept) < config.char_iou_threshold for kept in selected):
            selected.append(box)
    return sorted(selected, key=lambda box: box[0])


def _find_boxes(
    binary: np.ndarray,
    config: LPRConfig,
) -> list[tuple[int, int, int, int]]:
    plate_height, plate_width = binary.shape[:2]
    min_width = config.char_min_width_ratio * plate_height
    max_width = config.char_max_width_ratio * plate_height
    min_height = config.char_min_height_ratio * plate_height
    max_height = config.char_max_height_ratio * plate_height
    min_area = config.char_min_area_ratio * plate_width * plate_height

    contours, _ = cv2.findContours(
        binary.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[
        : config.max_char_candidates
    ]

    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = width * height
        if not min_width <= width <= max_width:
            continue
        if not min_height <= height <= max_height:
            continue
        if area < min_area:
            continue
        boxes.append((x, y, width, height))
    return _deduplicate_boxes(boxes, config)


def _score_boxes(
    boxes: list[tuple[int, int, int, int]],
    config: LPRConfig,
) -> float:
    count = len(boxes)
    if count == 0:
        return 0.0

    if config.expected_min_chars <= count <= config.expected_max_chars:
        count_score = 1.0
    else:
        distance = min(
            abs(count - config.expected_min_chars),
            abs(count - config.expected_max_chars),
        )
        count_score = max(0.0, 1.0 - (distance / config.expected_max_chars))

    heights = np.array([box[3] for box in boxes], dtype=np.float32)
    consistency = 1.0 - min(float(np.std(heights) / np.mean(heights)), 1.0)
    return (count_score * 0.65) + (consistency * 0.35)


def extract_character_regions(
    plate_image: np.ndarray,
    config: LPRConfig = DEFAULT_CONFIG,
) -> list[CharacterRegion]:
    """Extract sorted character regions from a cropped plate image."""

    if plate_image is None or plate_image.size == 0:
        raise ValueError("Expected a non-empty plate image.")

    plate = _prepare_plate(plate_image, config)
    best_binary: np.ndarray | None = None
    best_boxes: list[tuple[int, int, int, int]] = []
    best_score = 0.0

    for binary in _threshold_variants(plate, config):
        boxes = _find_boxes(binary, config)
        score = _score_boxes(boxes, config)
        if score > best_score:
            best_binary = binary
            best_boxes = boxes
            best_score = score

    if best_binary is None or not best_boxes:
        logger.warning("No character regions were found on the plate.")
        return []

    return [
        CharacterRegion(
            bbox=box,
            image=_format_character(best_binary, box, config),
        )
        for box in best_boxes
    ]


def segment_characters(
    plate_image: np.ndarray,
    config: LPRConfig = DEFAULT_CONFIG,
) -> list[np.ndarray]:
    """Return segmented character images sorted from left to right."""

    return [region.image for region in extract_character_regions(plate_image, config)]
