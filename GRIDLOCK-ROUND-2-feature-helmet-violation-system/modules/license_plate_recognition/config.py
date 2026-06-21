"""Configuration values for the license plate recognition module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class LPRConfig:
    """Runtime and training configuration for the LPR pipeline."""

    module_dir: Path = MODULE_DIR
    model_dir: Path = MODULE_DIR / "models"
    dataset_dir: Path = MODULE_DIR / "datasets"
    output_dir: Path = MODULE_DIR / "outputs"
    cascade_path: Path = MODULE_DIR / "datasets" / "indian_license_plate.xml"
    model_filename: str = "license_plate_cnn.h5"
    fallback_cascade_name: str = "haarcascade_russian_plate_number.xml"

    characters: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    image_size: tuple[int, int] = (28, 28)
    plate_size: tuple[int, int] = (333, 75)
    character_canvas_size: tuple[int, int] = (24, 44)
    character_inner_size: tuple[int, int] = (20, 40)
    character_border: int = 2
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")

    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)
    low_light_mean_threshold: float = 90.0
    low_light_gamma: float = 1.45
    denoise_h: int = 10
    denoise_template_window_size: int = 7
    denoise_search_window_size: int = 21
    adaptive_block_size: int = 31
    adaptive_c: int = 15
    threshold_blur_kernel: tuple[int, int] = (5, 5)
    morph_kernel_size: tuple[int, int] = (3, 3)
    character_morph_iterations: int = 1
    plate_border_size: int = 3

    cascade_scale_factor: float = 1.2
    cascade_min_neighbors: int = 7
    cascade_min_size: tuple[int, int] = (60, 20)
    min_plate_aspect: float = 2.0
    max_plate_aspect: float = 6.5
    expected_plate_aspect: float = 4.4
    min_plate_area_ratio: float = 0.002
    max_plate_area_ratio: float = 0.35
    preferred_plate_area_ratio: float = 0.045
    confidence_threshold: float = 0.45
    cascade_confidence_scale: float = 0.95
    contour_confidence_scale: float = 0.85
    bilateral_filter_diameter: int = 9
    bilateral_sigma_color: int = 75
    bilateral_sigma_space: int = 75
    canny_threshold1: int = 60
    canny_threshold2: int = 180
    plate_morph_kernel: tuple[int, int] = (17, 3)
    plate_morph_iterations: int = 2
    max_plate_candidates: int = 20

    max_char_candidates: int = 25
    expected_min_chars: int = 4
    expected_max_chars: int = 12
    char_min_width_ratio: float = 0.08
    char_max_width_ratio: float = 0.55
    char_min_height_ratio: float = 0.25
    char_max_height_ratio: float = 0.95
    char_min_area_ratio: float = 0.003
    char_iou_threshold: float = 0.25

    train_batch_size: int = 32
    epochs: int = 80
    learning_rate: float = 0.0001
    validation_split: float = 0.2
    early_stop_accuracy: float = 0.99
    random_seed: int = 42

    @property
    def model_path(self) -> Path:
        """Default path for saved CNN weights."""

        return self.model_dir / self.model_filename

    @property
    def evaluation_report_path(self) -> Path:
        """Default path for evaluation metrics."""

        return self.output_dir / "evaluation_report.json"

    @property
    def annotated_image_path(self) -> Path:
        """Default path for annotated inference output."""

        return self.output_dir / "annotated_image.jpg"


DEFAULT_CONFIG = LPRConfig()
