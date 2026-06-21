"""Metadata persistence for generated violation evidence."""

from __future__ import annotations

from pathlib import Path

try:
    from .config import DEFAULT_CONFIG, EvidenceConfig
    from .schemas import EvidenceRecord
    from .utils import append_jsonl, write_json
except ImportError:  # pragma: no cover - supports direct script execution
    from config import DEFAULT_CONFIG, EvidenceConfig
    from schemas import EvidenceRecord
    from utils import append_jsonl, write_json


def save_evidence_metadata(
    record: EvidenceRecord,
    metadata_path: str | Path,
    config: EvidenceConfig = DEFAULT_CONFIG,
    append_manifest: bool = True,
) -> Path:
    """Save one metadata JSON file and optionally append to the manifest."""

    payload = record.to_dict()
    path = write_json(metadata_path, payload)
    if append_manifest:
        append_jsonl(config.manifest_path, payload)
    return path
