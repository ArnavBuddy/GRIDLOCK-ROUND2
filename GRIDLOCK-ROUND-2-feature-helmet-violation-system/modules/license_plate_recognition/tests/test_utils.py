"""Tests for shared utility helpers."""

from modules.license_plate_recognition.utils import class_label_from_name
from modules.license_plate_recognition.character_recognizer import (
    correct_indian_plate_text,
)


def test_class_label_from_name_accepts_bare_and_prefixed_labels() -> None:
    assert class_label_from_name("A") == "A"
    assert class_label_from_name("class_A") == "A"
    assert class_label_from_name("class_7") == "7"
    assert class_label_from_name("not_a_label") is None


def test_correct_indian_plate_text_handles_common_ocr_confusions() -> None:
    assert correct_indian_plate_text("DLBCAF5D3O") == "DL8CAF5030"
