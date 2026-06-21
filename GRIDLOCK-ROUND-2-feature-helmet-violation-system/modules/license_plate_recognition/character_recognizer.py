"""CNN character recognition for segmented license plate characters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from .config import DEFAULT_CONFIG, LPRConfig
    from .utils import normalize_plate_text
except ImportError:  # pragma: no cover - supports direct script execution
    from config import DEFAULT_CONFIG, LPRConfig
    from utils import normalize_plate_text


def _import_tensorflow() -> Any:
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "TensorFlow is required for character recognition. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return tf


def build_cnn_model(config: LPRConfig = DEFAULT_CONFIG) -> Any:
    """Build the CNN architecture adapted from the reference notebook."""

    tf = _import_tensorflow()
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                16,
                (22, 22),
                input_shape=(*config.image_size, 3),
                activation="relu",
                padding="same",
            ),
            tf.keras.layers.Conv2D(
                32,
                (16, 16),
                activation="relu",
                padding="same",
            ),
            tf.keras.layers.Conv2D(
                64,
                (8, 8),
                activation="relu",
                padding="same",
            ),
            tf.keras.layers.Conv2D(
                64,
                (4, 4),
                activation="relu",
                padding="same",
            ),
            tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(len(config.characters), activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_model(
    model_path: str | Path | None = None,
    config: LPRConfig = DEFAULT_CONFIG,
) -> Any:
    """Load trained CNN weights from disk."""

    tf = _import_tensorflow()
    path = Path(model_path) if model_path else config.model_path
    if not path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {path}. Run python train.py first."
        )
    return tf.keras.models.load_model(str(path), compile=False)


def _prepare_character_image(
    character_image: np.ndarray,
    config: LPRConfig,
) -> np.ndarray:
    image = cv2.resize(
        character_image,
        config.image_size,
        interpolation=cv2.INTER_AREA,
    )
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)


def predict_character(
    character_image: np.ndarray,
    model: Any,
    config: LPRConfig = DEFAULT_CONFIG,
) -> tuple[str, float]:
    """Predict a single segmented character and confidence."""

    if character_image is None or character_image.size == 0:
        raise ValueError("Expected a non-empty character image.")

    batch = _prepare_character_image(character_image, config)
    probabilities = model.predict(batch, verbose=0)[0]
    class_index = int(np.argmax(probabilities))
    character = config.characters[class_index]
    confidence = float(probabilities[class_index])
    return character, confidence


def correct_indian_plate_text(text: str) -> str:
    """Fix common OCR confusions using Indian registration number structure."""

    normalized = normalize_plate_text(text)
    if len(normalized) < 7:
        return normalized

    digit_map = {
        "B": "8",
        "D": "0",
        "G": "6",
        "I": "1",
        "L": "1",
        "O": "0",
        "Q": "0",
        "S": "5",
        "T": "7",
        "Z": "2",
    }
    letter_map = {
        "0": "O",
        "1": "I",
        "2": "Z",
        "5": "S",
        "6": "G",
        "7": "T",
        "8": "B",
    }

    chars = list(normalized)

    for index in range(min(2, len(chars))):
        chars[index] = letter_map.get(chars[index], chars[index])

    for index in range(max(0, len(chars) - 4), len(chars)):
        chars[index] = digit_map.get(chars[index], chars[index])

    middle_start = 2
    middle_end = max(middle_start, len(chars) - 4)
    if middle_start < middle_end:
        chars[middle_start] = digit_map.get(chars[middle_start], chars[middle_start])

    if middle_start + 1 < middle_end:
        second_middle = chars[middle_start + 1]
        if second_middle.isdigit() or second_middle in digit_map:
            chars[middle_start + 1] = digit_map.get(second_middle, second_middle)

    series_start = middle_start + 1
    if middle_start + 1 < middle_end and chars[middle_start + 1].isdigit():
        series_start = middle_start + 2

    for index in range(series_start, middle_end):
        chars[index] = letter_map.get(chars[index], chars[index])

    return "".join(chars)


def recognize_plate(
    character_images: list[np.ndarray],
    model: Any,
    config: LPRConfig = DEFAULT_CONFIG,
    return_confidence: bool = False,
) -> str | tuple[str, float]:
    """Recognize the registration text from segmented character images."""

    if not character_images:
        raise ValueError("No segmented characters were provided.")

    predictions = [
        predict_character(character_image, model, config=config)
        for character_image in character_images
    ]
    raw_text = normalize_plate_text("".join(character for character, _ in predictions))
    text = correct_indian_plate_text(raw_text)
    confidence = float(np.mean([score for _, score in predictions]))

    if return_confidence:
        return text, confidence
    return text
