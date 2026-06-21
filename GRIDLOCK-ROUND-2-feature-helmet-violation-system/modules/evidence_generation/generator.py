"""End-to-end evidence generation pipeline."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

try:
    from .annotator import annotate_image
    from .config import DEFAULT_CONFIG, EvidenceConfig
    from .metadata_store import save_evidence_metadata
    from .schemas import EvidenceRecord, Violation
    from .utils import current_timestamp, ensure_directory, load_image, save_image
except ImportError:  # pragma: no cover - supports direct script execution
    from annotator import annotate_image
    from config import DEFAULT_CONFIG, EvidenceConfig
    from metadata_store import save_evidence_metadata
    from schemas import EvidenceRecord, Violation
    from utils import current_timestamp, ensure_directory, load_image, save_image


def _build_evidence_id(
    source_image: str | Path,
    generated_at: str,
    frame_id: str | None,
) -> str:
    stem = Path(source_image).stem.replace(" ", "_")
    clean_timestamp = (
        generated_at.replace(":", "").replace("-", "").replace("+", "_")
    )
    suffix = frame_id or uuid4().hex[:8]
    return f"{stem}_{clean_timestamp}_{suffix}"


def generate_evidence(
    image_path: str | Path,
    violations: list[Violation],
    output_dir: str | Path | None = None,
    frame_id: str | None = None,
    camera_id: str | None = None,
    location: str | None = None,
    config: EvidenceConfig = DEFAULT_CONFIG,
) -> EvidenceRecord:
    """Create an annotated image and metadata for a set of violations."""

    if not violations:
        raise ValueError("At least one violation must be supplied.")

    image = load_image(image_path)
    generated_at = current_timestamp(config)
    evidence_id = _build_evidence_id(image_path, generated_at, frame_id)
    target_dir = ensure_directory(output_dir or config.output_dir)

    annotated_path = (
        target_dir / f"{config.annotated_prefix}_{evidence_id}{config.annotated_extension}"
    )
    metadata_path = target_dir / f"{config.annotated_prefix}_{evidence_id}{config.metadata_extension}"

    annotated = annotate_image(image, violations, config=config)
    save_image(annotated_path, annotated)

    record = EvidenceRecord(
        evidence_id=evidence_id,
        source_image=str(Path(image_path)),
        annotated_image=str(annotated_path),
        metadata_path=str(metadata_path),
        generated_at=generated_at,
        violations=violations,
        frame_id=frame_id,
        camera_id=camera_id,
        location=location,
    )
    save_evidence_metadata(record, metadata_path, config=config)
    return record
