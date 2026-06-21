"""Configuration for the evidence generation module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class EvidenceConfig:
    """Runtime configuration for evidence generation."""

    module_dir: Path = MODULE_DIR
    output_dir: Path = MODULE_DIR / "outputs"
    default_timezone: str = "Asia/Kolkata"
    annotated_prefix: str = "evidence"
    annotated_extension: str = ".jpg"
    metadata_extension: str = ".json"
    manifest_filename: str = "violation_events.jsonl"
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")

    box_color: tuple[int, int, int] = (0, 0, 255)
    box_thickness: int = 2
    label_background_color: tuple[int, int, int] = (0, 0, 180)
    label_text_color: tuple[int, int, int] = (255, 255, 255)
    label_font_scale: float = 0.55
    label_thickness: int = 1
    metadata_datetime_format: str = "%Y-%m-%dT%H:%M:%S%z"

    @property
    def manifest_path(self) -> Path:
        """Path to the append-only violation metadata manifest."""

        return self.output_dir / self.manifest_filename


DEFAULT_CONFIG = EvidenceConfig()
