"""End-to-end inference pipeline for license plate recognition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .character_recognizer import load_model, recognize_plate
    from .character_segmenter import segment_characters
    from .config import DEFAULT_CONFIG, LPRConfig
    from .plate_detector import crop_plate, detect_license_plate, draw_plate_bbox
    from .preprocessing import enhance_image
    from .utils import (
        ensure_directory,
        ensure_module_directories,
        get_logger,
        load_image,
        save_image,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from character_recognizer import load_model, recognize_plate
    from character_segmenter import segment_characters
    from config import DEFAULT_CONFIG, LPRConfig
    from plate_detector import crop_plate, detect_license_plate, draw_plate_bbox
    from preprocessing import enhance_image
    from utils import (
        ensure_directory,
        ensure_module_directories,
        get_logger,
        load_image,
        save_image,
    )


logger = get_logger(__name__)


def run_inference(
    image_path: str | Path,
    model_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    config: LPRConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Run the full LPR pipeline for one input image."""

    ensure_module_directories(config)
    image = load_image(image_path)
    processed = enhance_image(image, config=config)

    detection = detect_license_plate(processed, config=config)
    if not detection.get("bbox"):
        raise RuntimeError("No license plate was detected in the input image.")

    plate = crop_plate(processed, detection=detection, config=config)
    characters = segment_characters(plate, config=config)
    if not characters:
        raise RuntimeError("No characters were segmented from the detected plate.")

    model = load_model(model_path=model_path, config=config)
    plate_number, recognition_confidence = recognize_plate(
        characters,
        model,
        config=config,
        return_confidence=True,
    )

    detection_confidence = float(detection["confidence"])
    overall_confidence = float((detection_confidence + recognition_confidence) / 2)
    output_path = Path(output_dir) if output_dir else config.output_dir
    annotated_path = output_path / "annotated_image.jpg"
    annotated = draw_plate_bbox(image, detection, label=plate_number)
    save_image(annotated_path, annotated)

    result = {
        "plate_number": plate_number,
        "confidence": round(overall_confidence, 4),
        "plate_detection": detection,
        "segmented_characters": len(characters),
        "annotated_image": str(annotated_path),
    }
    logger.info("Recognized plate %s", plate_number)
    return result


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Run license plate recognition.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--model", default=None, help="Path to trained CNN weights.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for annotated inference outputs.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_CONFIG.output_dir
    ensure_directory(output_dir)
    try:
        result = run_inference(
            image_path=args.image,
            model_path=args.model,
            output_dir=output_dir,
        )
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
