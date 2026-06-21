"""Shared helpers for the license plate recognition module."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from .config import DEFAULT_CONFIG, LPRConfig
except ImportError:  # pragma: no cover - supports direct script execution
    from config import DEFAULT_CONFIG, LPRConfig


LOGGER_NAME = "license_plate_recognition"


class LPRException(Exception):
    """Base exception for license plate recognition errors."""


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the module logger."""

    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the LPR namespace."""

    configure_logging()
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_module_directories(config: LPRConfig = DEFAULT_CONFIG) -> None:
    """Create model, dataset, and output directories used by the module."""

    for directory in (config.model_dir, config.dataset_dir, config.output_dir):
        ensure_directory(directory)


def load_image(image_path: str | Path) -> np.ndarray:
    """Read an image from disk using OpenCV."""

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(str(path))
    if image is None:
        raise LPRException(f"Unable to read image: {path}")
    return image


def save_image(image_path: str | Path, image: np.ndarray) -> Path:
    """Write an image to disk and return the destination path."""

    path = Path(image_path)
    ensure_directory(path.parent)
    saved = cv2.imwrite(str(path), image)
    if not saved:
        raise LPRException(f"Unable to save image: {path}")
    return path


def write_json(output_path: str | Path, payload: dict[str, Any]) -> Path:
    """Persist a JSON payload with stable formatting."""

    path = Path(output_path)
    ensure_directory(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def clamp_bbox(
    bbox: list[int] | tuple[int, int, int, int],
    image_shape: tuple[int, ...],
) -> list[int]:
    """Clamp a bounding box to image boundaries."""

    height, width = image_shape[:2]
    x1, y1, x2, y2 = [int(value) for value in bbox]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return [x1, y1, x2, y2]


def normalize_plate_text(text: str, config: LPRConfig = DEFAULT_CONFIG) -> str:
    """Keep only supported registration characters."""

    allowed = set(config.characters)
    return "".join(char for char in text.upper() if char in allowed)


def class_label_from_name(
    folder_name: str,
    config: LPRConfig = DEFAULT_CONFIG,
) -> str | None:
    """Resolve labels from folders named ``A`` or ``class_A``."""

    label = folder_name.upper()
    if label.startswith("CLASS_"):
        label = label.removeprefix("CLASS_")
    return label if label in config.characters else None
