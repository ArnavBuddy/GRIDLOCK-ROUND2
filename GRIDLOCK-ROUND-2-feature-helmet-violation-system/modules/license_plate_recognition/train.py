"""Training script for the license plate character CNN."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .character_recognizer import build_cnn_model
    from .config import DEFAULT_CONFIG, LPRConfig
    from .utils import ensure_directory, ensure_module_directories, get_logger
    from .utils import class_label_from_name
except ImportError:  # pragma: no cover - supports direct script execution
    from character_recognizer import build_cnn_model
    from config import DEFAULT_CONFIG, LPRConfig
    from utils import ensure_directory, ensure_module_directories, get_logger
    from utils import class_label_from_name


logger = get_logger(__name__)


@dataclass(frozen=True)
class DatasetLayout:
    """Resolved training dataset layout."""

    train_dir: Path
    val_dir: Path | None
    use_validation_split: bool = False


def _import_tensorflow() -> Any:
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "TensorFlow is required for training. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return tf


class StopAtAccuracy:
    """Callback factory mirroring the notebook's early stop behavior."""

    def __init__(self, target_accuracy: float) -> None:
        self.target_accuracy = target_accuracy

    def build(self) -> Any:
        """Build the TensorFlow callback after TensorFlow is available."""

        tf = _import_tensorflow()
        target_accuracy = self.target_accuracy

        class _Callback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
                metrics = logs or {}
                validation_accuracy = metrics.get("val_accuracy")
                if validation_accuracy and validation_accuracy >= target_accuracy:
                    self.model.stop_training = True

        return _Callback()


def _has_class_folders(path: Path, config: LPRConfig) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any(
        child.is_dir() and class_label_from_name(child.name, config) is not None
        for child in path.iterdir()
    )


def _dataset_help(data_dir: Path) -> str:
    return (
        f"No training images were found under {data_dir}.\n\n"
        "Expected one of these layouts:\n"
        "  datasets/train/A/*.png and datasets/val/A/*.png\n"
        "  datasets/train/class_A/*.png and datasets/val/class_A/*.png\n"
        "  datasets/data/data/train/class_A/*.png and "
        "datasets/data/data/val/class_A/*.png\n"
        "  datasets/A/*.png or datasets/class_A/*.png, ... "
        "(the script will create an in-memory validation split)\n\n"
        "Put labelled character images into folders named 0-9/A-Z or "
        "class_0/class_A, or run:\n"
        "  python train.py --data-dir <path-to-your-character-dataset>"
    )


def _resolve_dataset_layout(
    data_dir: Path,
    config: LPRConfig = DEFAULT_CONFIG,
) -> DatasetLayout:
    """Resolve common dataset layouts used by the notebook and this module."""

    layout_roots = [
        data_dir,
        data_dir / "data",
        data_dir / "data" / "data",
    ]

    for root in layout_roots:
        train_dir = root / "train"
        val_dir = root / "val"
        if _has_class_folders(train_dir, config) and _has_class_folders(
            val_dir,
            config,
        ):
            return DatasetLayout(train_dir=train_dir, val_dir=val_dir)
        if _has_class_folders(train_dir, config):
            return DatasetLayout(
                train_dir=train_dir,
                val_dir=None,
                use_validation_split=True,
            )

    if data_dir.name.lower() == "train":
        val_dir = data_dir.parent / "val"
        if _has_class_folders(data_dir, config) and _has_class_folders(
            val_dir,
            config,
        ):
            return DatasetLayout(train_dir=data_dir, val_dir=val_dir)

    if _has_class_folders(data_dir, config):
        return DatasetLayout(
            train_dir=data_dir,
            val_dir=None,
            use_validation_split=True,
        )

    raise FileNotFoundError(_dataset_help(data_dir))


def train_model(
    data_dir: str | Path | None = None,
    output_model: str | Path | None = None,
    config: LPRConfig = DEFAULT_CONFIG,
) -> Path:
    """Train the CNN and save model weights."""

    ensure_module_directories(config)
    dataset_dir = Path(data_dir) if data_dir else config.dataset_dir
    layout = _resolve_dataset_layout(dataset_dir, config=config)
    tf = _import_tensorflow()

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        validation_split=(
            config.validation_split if layout.use_validation_split else 0.0
        ),
    )

    train_generator = train_datagen.flow_from_directory(
        str(layout.train_dir),
        target_size=config.image_size,
        batch_size=config.train_batch_size,
        class_mode="sparse",
        color_mode="rgb",
        shuffle=True,
        seed=config.random_seed,
        subset="training" if layout.use_validation_split else None,
    )

    if layout.val_dir is None:
        validation_generator = train_datagen.flow_from_directory(
            str(layout.train_dir),
            target_size=config.image_size,
            batch_size=config.train_batch_size,
            class_mode="sparse",
            color_mode="rgb",
            shuffle=False,
            subset="validation",
        )
    else:
        validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255.0
        )
        validation_generator = validation_datagen.flow_from_directory(
            str(layout.val_dir),
            target_size=config.image_size,
            batch_size=config.train_batch_size,
            class_mode="sparse",
            color_mode="rgb",
            shuffle=False,
        )

    model = build_cnn_model(config)
    callbacks = [StopAtAccuracy(config.early_stop_accuracy).build()]
    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=config.epochs,
        callbacks=callbacks,
    )

    model_path = Path(output_model) if output_model else config.model_path
    ensure_directory(model_path.parent)
    model.save(str(model_path))
    logger.info("Saved trained model to %s", model_path)
    return model_path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Train the LPR character CNN.")
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_CONFIG.dataset_dir),
        help="Dataset root containing train and val class folders.",
    )
    parser.add_argument(
        "--output-model",
        default=str(DEFAULT_CONFIG.model_path),
        help="Path where trained model weights will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    try:
        train_model(data_dir=args.data_dir, output_model=args.output_model)
    except (FileNotFoundError, ImportError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
