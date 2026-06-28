"""Helpers for parsing ExDark ground-truth annotation files."""

from __future__ import annotations

from pathlib import Path

from sentinel.evaluation.metrics import Box, GroundTruthRecord


_EXDARK_CLASS_MAP = {
    "people": "person",
    "person": "person",
}


def parse_exdark_annotation_file(
    annotation_path: Path,
    image_id: str,
    target_class: str = "person",
) -> list[GroundTruthRecord]:
    """
    Parse an ExDark `.txt` annotation file.

    Format (per line, after optional header):
      People  l  t  w  h  ...
    Bounding box is [left, top, width, height] in pixels.
    """
    records: list[GroundTruthRecord] = []
    if not annotation_path.exists():
        return records

    with open(annotation_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            class_name = parts[0].lower()
            normalised = _EXDARK_CLASS_MAP.get(class_name)
            if normalised != target_class:
                continue
            left, top, width, height = map(float, parts[1:5])
            bbox: Box = (left, top, left + width, top + height)
            records.append(
                GroundTruthRecord(
                    image_id=image_id,
                    class_name="person",
                    bbox=bbox,
                )
            )
    return records
