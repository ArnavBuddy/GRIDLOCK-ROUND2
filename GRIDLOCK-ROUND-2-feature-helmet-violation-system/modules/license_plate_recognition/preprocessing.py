"""Image preprocessing operations for license plate recognition."""

from __future__ import annotations

import cv2
import numpy as np

try:
    from .config import DEFAULT_CONFIG, LPRConfig
except ImportError:  # pragma: no cover - supports direct script execution
    from config import DEFAULT_CONFIG, LPRConfig


def _validate_image(image: np.ndarray) -> None:
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Expected a non-empty OpenCV image array.")


def grayscale_image(image: np.ndarray) -> np.ndarray:
    """Convert a BGR, BGRA, or grayscale image to grayscale."""

    _validate_image(image)
    if image.ndim == 2:
        return image.copy()
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_image(
    image: np.ndarray,
    size: tuple[int, int] | None = None,
    interpolation: int = cv2.INTER_AREA,
    config: LPRConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """Resize an image to a configured or provided ``(width, height)`` size."""

    _validate_image(image)
    target_size = size or config.image_size
    return cv2.resize(image, target_size, interpolation=interpolation)


def denoise_image(
    image: np.ndarray,
    config: LPRConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """Reduce sensor noise while preserving edges needed for characters."""

    _validate_image(image)
    if image.ndim == 2:
        return cv2.fastNlMeansDenoising(
            image,
            None,
            config.denoise_h,
            config.denoise_template_window_size,
            config.denoise_search_window_size,
        )
    return cv2.fastNlMeansDenoisingColored(
        image,
        None,
        config.denoise_h,
        config.denoise_h,
        config.denoise_template_window_size,
        config.denoise_search_window_size,
    )


def _apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    inverse_gamma = 1.0 / gamma
    table = np.array(
        [((value / 255.0) ** inverse_gamma) * 255 for value in range(256)]
    ).astype("uint8")
    return cv2.LUT(image, table)


def enhance_image(
    image: np.ndarray,
    config: LPRConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """Improve contrast for low light, shadows, blur, and uneven exposure."""

    _validate_image(image)
    denoised = denoise_image(image, config=config)

    if denoised.ndim == 2:
        enhanced = cv2.createCLAHE(
            clipLimit=config.clahe_clip_limit,
            tileGridSize=config.clahe_tile_grid_size,
        ).apply(denoised)
    else:
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        lightness = cv2.createCLAHE(
            clipLimit=config.clahe_clip_limit,
            tileGridSize=config.clahe_tile_grid_size,
        ).apply(lightness)
        enhanced = cv2.cvtColor(
            cv2.merge((lightness, channel_a, channel_b)),
            cv2.COLOR_LAB2BGR,
        )

    gray_mean = float(np.mean(grayscale_image(enhanced)))
    if gray_mean < config.low_light_mean_threshold:
        enhanced = _apply_gamma(enhanced, config.low_light_gamma)
    return enhanced


def adaptive_threshold(
    image: np.ndarray,
    invert: bool = True,
    config: LPRConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """Create a binary image resilient to shadows and non-uniform lighting."""

    gray = grayscale_image(image)
    blur = cv2.GaussianBlur(gray, config.threshold_blur_kernel, 0)
    threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    return cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        threshold_type,
        config.adaptive_block_size,
        config.adaptive_c,
    )
