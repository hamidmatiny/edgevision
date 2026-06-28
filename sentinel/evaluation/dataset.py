"""Load evaluation datasets from JSON manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sentinel.evaluation.metrics import Box, GroundTruthRecord


@dataclass(frozen=True)
class EvalImageEntry:
    image_id: str
    image_path: Path
    ground_truths: tuple[GroundTruthRecord, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class EvalDataset:
    name: str
    description: str
    provenance: str
    split: str
    target_class: str
    images: tuple[EvalImageEntry, ...]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_eval_dataset(manifest_path: str | Path) -> EvalDataset:
    manifest_path = Path(manifest_path)
    with open(manifest_path, encoding="utf-8") as f:
        raw = json.load(f)

    root = _repo_root()
    images: list[EvalImageEntry] = []
    for item in raw["images"]:
        image_path = root / item["image_path"]
        gts = tuple(
            GroundTruthRecord(
                image_id=item["image_id"],
                class_name=gt["class_name"],
                bbox=tuple(gt["bbox"]),
            )
            for gt in item["ground_truths"]
        )
        images.append(
            EvalImageEntry(
                image_id=item["image_id"],
                image_path=image_path,
                ground_truths=gts,
                metadata=item.get("metadata", {}),
            )
        )

    return EvalDataset(
        name=raw["name"],
        description=raw["description"],
        provenance=raw["provenance"],
        split=raw["split"],
        target_class=raw.get("target_class", "person"),
        images=tuple(images),
    )
