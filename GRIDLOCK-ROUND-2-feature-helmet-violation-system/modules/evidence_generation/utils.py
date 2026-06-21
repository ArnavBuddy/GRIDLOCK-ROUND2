"""Shared helpers for evidence generation."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

try:
    from .config import DEFAULT_CONFIG, EvidenceConfig
except ImportError:  # pragma: no cover - supports direct script execution
    from config import DEFAULT_CONFIG, EvidenceConfig


LOGGER_NAME = "evidence_generation"


class EvidenceGenerationError(Exception):
    """Base exception for evidence generation failures."""


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure the module logger."""

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
    """Return a child logger under the evidence namespace."""

    configure_logging()
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def ensure_directory(path: str | Path) -> Path:
    """Create a directory when needed."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_image(image_path: str | Path) -> np.ndarray:
    """Read an image from disk."""

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(str(path))
    if image is None:
        raise EvidenceGenerationError(f"Unable to read image: {path}")
    return image


def save_image(image_path: str | Path, image: np.ndarray) -> Path:
    """Write an image to disk."""

    path = Path(image_path)
    ensure_directory(path.parent)
    saved = cv2.imwrite(str(path), image)
    if not saved:
        raise EvidenceGenerationError(f"Unable to save image: {path}")
    return path


def read_json(path: str | Path) -> dict | list:
    """Read a JSON file."""

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    return json.loads(file_path.read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict) -> Path:
    """Write a JSON file with stable formatting."""

    file_path = Path(path)
    ensure_directory(file_path.parent)
    file_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return file_path


def append_jsonl(path: str | Path, payload: dict) -> Path:
    """Append a JSON object to a JSONL file."""

    file_path = Path(path)
    ensure_directory(file_path.parent)
    with file_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")
    return file_path


def current_timestamp(config: EvidenceConfig = DEFAULT_CONFIG) -> str:
    """Return the current timestamp in the configured timezone."""

    timezone = ZoneInfo(config.default_timezone)
    return datetime.now(timezone).strftime(config.metadata_datetime_format)


def clamp_bbox(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    """Clamp a bounding box to image boundaries."""

    height, width = image_shape[:2]
    x1, y1, x2, y2 = [int(value) for value in bbox]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2
