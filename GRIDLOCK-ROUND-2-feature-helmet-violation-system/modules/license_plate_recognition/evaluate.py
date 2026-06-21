"""Evaluation utilities for the license plate character recognizer."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from .character_recognizer import load_model, predict_character
    from .config import DEFAULT_CONFIG, LPRConfig
    from .utils import (
        class_label_from_name,
        ensure_module_directories,
        get_logger,
        load_image,
        write_json,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from character_recognizer import load_model, predict_character
    from config import DEFAULT_CONFIG, LPRConfig
    from utils import (
        class_label_from_name,
        ensure_module_directories,
        get_logger,
        load_image,
        write_json,
    )


logger = get_logger(__name__)


def _import_metrics() -> Any:
    try:
        from sklearn.metrics import (  # type: ignore
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "scikit-learn is required for evaluation. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return accuracy_score, precision_score, recall_score, f1_score


def _iter_labelled_images(
    data_dir: Path,
    config: LPRConfig,
) -> list[tuple[Path, str]]:
    samples: list[tuple[Path, str]] = []
    for image_path in data_dir.rglob("*"):
        if image_path.suffix.lower() not in config.image_extensions:
            continue
        label = class_label_from_name(image_path.parent.name, config)
        if label is not None:
            samples.append((image_path, label))
    return samples


def _has_labelled_images(data_dir: Path, config: LPRConfig) -> bool:
    return bool(_iter_labelled_images(data_dir, config))


def _evaluation_help(data_dir: Path) -> str:
    return (
        f"No labelled evaluation images found under {data_dir}.\n\n"
        "Expected one of these layouts:\n"
        "  datasets/test/class_A/*.jpg\n"
        "  datasets/val/class_A/*.jpg\n"
        "  datasets/data/data/test/class_A/*.jpg\n"
        "  datasets/data/data/val/class_A/*.jpg\n\n"
        "Your archive appears to use a validation folder, so run:\n"
        "  python evaluate.py --data-dir "
        "\"C:\\Users\\Arnav Majithia'\\Downloads\\archive\\data\\data\\val\""
    )


def _resolve_evaluation_dir(
    data_dir: Path,
    config: LPRConfig = DEFAULT_CONFIG,
) -> Path:
    """Resolve common evaluation dataset layouts."""

    split_candidates: list[Path] = []
    root_candidates: list[Path] = []
    for root in (data_dir, data_dir / "data", data_dir / "data" / "data"):
        split_candidates.extend([root / "test", root / "val"])
        root_candidates.append(root)

    if data_dir.name.lower() == "train":
        split_candidates.extend([data_dir.parent / "test", data_dir.parent / "val"])

    seen: set[Path] = set()
    for candidate in [*split_candidates, *root_candidates]:
        if candidate in seen:
            continue
        seen.add(candidate)
        if _has_labelled_images(candidate, config):
            return candidate

    raise FileNotFoundError(_evaluation_help(data_dir))


def evaluate_model(
    data_dir: str | Path | None = None,
    model_path: str | Path | None = None,
    output_path: str | Path | None = None,
    config: LPRConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Evaluate the trained model on labelled character folders."""

    ensure_module_directories(config)
    dataset_dir = _resolve_evaluation_dir(
        Path(data_dir) if data_dir else config.dataset_dir,
        config=config,
    )
    samples = _iter_labelled_images(dataset_dir, config)

    model = load_model(model_path=model_path, config=config)
    y_true: list[str] = []
    y_pred: list[str] = []

    for image_path, label in samples:
        image = load_image(image_path)
        prediction, _ = predict_character(image, model, config=config)
        y_true.append(label)
        y_pred.append(prediction)

    accuracy_score, precision_score, recall_score, f1_score = _import_metrics()
    report = {
        "samples": len(samples),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(
            float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            4,
        ),
        "recall": round(
            float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            4,
        ),
        "f1_score": round(
            float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            4,
        ),
    }

    report_path = Path(output_path) if output_path else config.evaluation_report_path
    write_json(report_path, report)
    logger.info("Saved evaluation report to %s", report_path)
    return report


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Evaluate the LPR CNN model.")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Evaluation dataset root containing test, val, or labelled class folders.",
    )
    parser.add_argument("--model", default=None, help="Path to trained CNN weights.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_CONFIG.evaluation_report_path),
        help="Path for evaluation_report.json.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    try:
        report = evaluate_model(
            data_dir=args.data_dir,
            model_path=args.model,
            output_path=args.output,
        )
    except (FileNotFoundError, ImportError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(report)


if __name__ == "__main__":
    main()
