"""Tests for evaluation dataset layout detection."""

from pathlib import Path

import pytest

from modules.license_plate_recognition.evaluate import _resolve_evaluation_dir


def _touch_sample(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"sample")


def test_resolve_evaluation_prefers_test_split(tmp_path: Path) -> None:
    _touch_sample(tmp_path / "test" / "class_A" / "1.jpg")
    _touch_sample(tmp_path / "val" / "class_B" / "1.jpg")

    assert _resolve_evaluation_dir(tmp_path) == tmp_path / "test"


def test_resolve_evaluation_accepts_notebook_val_layout(tmp_path: Path) -> None:
    _touch_sample(tmp_path / "data" / "data" / "val" / "class_A" / "1.jpg")

    assert _resolve_evaluation_dir(tmp_path) == tmp_path / "data" / "data" / "val"


def test_missing_evaluation_dataset_has_actionable_message(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Expected one of these layouts"):
        _resolve_evaluation_dir(tmp_path)
