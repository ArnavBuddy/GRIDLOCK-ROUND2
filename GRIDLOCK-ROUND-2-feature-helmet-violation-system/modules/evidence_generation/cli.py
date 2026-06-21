"""Command line interface for evidence generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .config import DEFAULT_CONFIG
    from .generator import generate_evidence
    from .schemas import Violation
    from .utils import read_json
except ImportError:  # pragma: no cover - supports direct script execution
    from config import DEFAULT_CONFIG
    from generator import generate_evidence
    from schemas import Violation
    from utils import read_json


def _load_violations(path: str | Path) -> list[Violation]:
    payload = read_json(path)
    if isinstance(payload, dict):
        raw_violations = payload.get("violations")
    else:
        raw_violations = payload

    if not isinstance(raw_violations, list):
        raise ValueError("Violations JSON must be a list or contain a violations list.")
    return [Violation.from_dict(item) for item in raw_violations]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Generate annotated violation evidence.")
    parser.add_argument("--image", required=True, help="Path to the source image.")
    parser.add_argument(
        "--violations",
        required=True,
        help="JSON file containing violation records.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_CONFIG.output_dir),
        help="Directory for annotated images and metadata.",
    )
    parser.add_argument("--frame-id", default=None, help="Optional frame identifier.")
    parser.add_argument("--camera-id", default=None, help="Optional camera identifier.")
    parser.add_argument("--location", default=None, help="Optional camera location.")
    args, unknown = parser.parse_known_args()
    if unknown:
        parser.error(
            "unrecognized arguments: "
            f"{' '.join(unknown)}\n"
            "Tip: wrap Windows paths that contain spaces in double quotes."
        )
    return args


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    try:
        record = generate_evidence(
            image_path=args.image,
            violations=_load_violations(args.violations),
            output_dir=args.output_dir,
            frame_id=args.frame_id,
            camera_id=args.camera_id,
            location=args.location,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc

    print(json.dumps(record.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
