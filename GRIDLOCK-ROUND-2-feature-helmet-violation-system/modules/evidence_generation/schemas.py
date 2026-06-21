"""Typed data models for evidence generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Violation:
    """One traffic violation to annotate and store."""

    violation_type: str
    bbox: tuple[int, int, int, int]
    confidence: float | None = None
    track_id: str | None = None
    description: str | None = None
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Violation":
        """Build a violation from JSON-compatible input."""

        bbox = payload.get("bbox")
        if not isinstance(bbox, list | tuple) or len(bbox) != 4:
            raise ValueError("Each violation requires bbox: [x1, y1, x2, y2].")

        violation_type = payload.get("violation_type") or payload.get("type")
        if not violation_type:
            raise ValueError("Each violation requires violation_type.")

        return cls(
            violation_type=str(violation_type),
            bbox=tuple(int(value) for value in bbox),
            confidence=(
                float(payload["confidence"])
                if payload.get("confidence") is not None
                else None
            ),
            track_id=(
                str(payload["track_id"]) if payload.get("track_id") is not None else None
            ),
            description=(
                str(payload["description"])
                if payload.get("description") is not None
                else None
            ),
            timestamp=(
                str(payload["timestamp"]) if payload.get("timestamp") is not None else None
            ),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible violation data."""

        data = asdict(self)
        data["bbox"] = list(self.bbox)
        return data


@dataclass(frozen=True)
class EvidenceRecord:
    """Generated evidence artifact and metadata."""

    evidence_id: str
    source_image: str
    annotated_image: str
    metadata_path: str
    generated_at: str
    violations: list[Violation]
    frame_id: str | None = None
    camera_id: str | None = None
    location: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible evidence record data."""

        data = asdict(self)
        data["violations"] = [violation.to_dict() for violation in self.violations]
        return data
