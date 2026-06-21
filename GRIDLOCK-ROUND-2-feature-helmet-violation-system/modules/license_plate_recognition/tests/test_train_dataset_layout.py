"""Tests for training dataset layout detection."""

from pathlib import Path

import pytest

from modules.license_plate_recognition.train import _resolve_dataset_layout


def _touch_sample(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"sample")


def test_resolve_notebook_dataset_layout(tmp_path: Path) -> None:
    _touch_sample(tmp_path / "data" / "data" / "train" / "class_A" / "1.png")
    _touch_sample(tmp_path / "data" / "data" / "val" / "class_A" / "1.png")

    layout = _resolve_dataset_layout(tmp_path)

    assert layout.train_dir == tmp_path / "data" / "data" / "train"
    assert layout.val_dir == tmp_path / "data" / "data" / "val"
    assert layout.use_validation_split is False


def test_resolve_single_folder_layout_uses_split(tmp_path: Path) -> None:
    _touch_sample(tmp_path / "A" / "1.png")

    layout = _resolve_dataset_layout(tmp_path)

    assert layout.train_dir == tmp_path
    assert layout.val_dir is None
    assert layout.use_validation_split is True


def test_missing_dataset_has_actionable_message(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Expected one of these layouts"):
        _resolve_dataset_layout(tmp_path)
