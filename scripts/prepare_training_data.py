#!/usr/bin/env python3
"""
Prepare Phase 2 commercial-safe low-light training data.

Pipeline:
  1. Resolve Intel IoT DevKit sample-videos (CC-BY 4.0) — NOT ExDark, NOT bus.jpg.
  2. Extract frames from each video.
  3. Pseudo-label person boxes on bright source frames with stock yolo11n.pt.
  4. Apply Albumentations low-light recipes (4× expansion per parent frame).
  5. Write YOLO dataset + versioned training manifest.

Usage:
    python scripts/prepare_training_data.py
    python scripts/prepare_training_data.py --frame-interval 6 --variants 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "training" / "lowlight_yolo"
DEFAULT_CACHE = REPO_ROOT / "data" / "training" / "sources"

sys.path.insert(0, str(REPO_ROOT))

from sentinel.detection.detector import Detector  # noqa: E402
from sentinel.training.augmentations import (  # noqa: E402
    LOW_LIGHT_RECIPES,
    apply_low_light_recipe,
    write_yolo_label,
)
from sentinel.training.intel_sources import (  # noqa: E402
    INTEL_ATTRIBUTION,
    INTEL_TRAINING_VIDEOS,
    INTEL_VIDEO_LICENSE,
    intel_video_url,
    resolve_local_video_path,
)


@dataclass
class ParentFrame:
    parent_id: str
    video_filename: str
    frame_index: int
    image_bgr: object  # np.ndarray — kept in memory only during build
    boxes: list[tuple[float, float, float, float]]


def download_video(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    print(f"  Downloading {dest.name} ...")
    urllib.request.urlretrieve(url, dest)
    return dest


def ensure_intel_videos(repo_root: Path, cache_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for video in INTEL_TRAINING_VIDEOS:
        local = resolve_local_video_path(video, repo_root, cache_dir)
        if local is None:
            local = download_video(intel_video_url(video.filename), cache_dir / video.filename)
        paths[video.filename] = local
        print(f"  {video.filename} -> {local}")
    return paths


def extract_and_label_frames(
    video_path: Path,
    video_filename: str,
    detector: Detector,
    frame_interval: int,
    label_confidence: float,
) -> list[ParentFrame]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    parents: list[ParentFrame] = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % frame_interval != 0:
            continue

        detections = detector.detect(frame)
        person_boxes = [
            tuple(det["bbox"])
            for det in detections
            if det["class"] == "person" and det["confidence"] >= label_confidence
        ]
        if not person_boxes:
            continue

        parent_id = f"{video_filename.stem}_f{frame_idx:06d}"
        parents.append(
            ParentFrame(
                parent_id=parent_id,
                video_filename=video_filename,
                frame_index=frame_idx,
                image_bgr=frame.copy(),
                boxes=person_boxes,
            )
        )
    cap.release()
    return parents


def val_split(parent_id: str, val_ratio: float) -> bool:
    """Deterministic hash split — entire parent + all variants stay in same split."""
    digest = hashlib.sha256(parent_id.encode()).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return bucket < val_ratio


def build_dataset(
    output_dir: Path,
    repo_root: Path,
    frame_interval: int = 6,
    variants: int = 4,
    val_ratio: float = 0.15,
    label_confidence: float = 0.5,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = DEFAULT_CACHE
    video_paths = ensure_intel_videos(repo_root, cache_dir)

    detector = Detector(
        model_name="yolo11n.pt",
        confidence_threshold=0.25,
        device="cpu",
        target_classes=["person"],
    )

    all_parents: list[ParentFrame] = []
    print("Extracting and pseudo-labeling frames (stock yolo11n on bright source frames)...")
    for video in INTEL_TRAINING_VIDEOS:
        path = video_paths[video.filename]
        parents = extract_and_label_frames(
            path,
            Path(video.filename),
            detector,
            frame_interval=frame_interval,
            label_confidence=label_confidence,
        )
        print(f"  {video.filename}: {len(parents)} labeled parent frames")
        all_parents.extend(parents)

    if not all_parents:
        raise RuntimeError("No labeled parent frames extracted — check videos and label confidence.")

    recipes = [name for name, _ in LOW_LIGHT_RECIPES[:variants]]
    manifest_rows: list[dict] = []
    train_count = val_count = 0

    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    print(f"Generating {variants} low-light variants per parent ({len(recipes)} recipes)...")
    for parent in all_parents:
        is_val = val_split(parent.parent_id, val_ratio)
        split = "val" if is_val else "train"
        h, w = parent.image_bgr.shape[:2]

        for recipe_name in recipes:
            aug = apply_low_light_recipe(parent.image_bgr, parent.boxes, recipe_name)
            if aug is None:
                continue
            aug_img, aug_boxes = aug
            image_id = f"{parent.parent_id}__{recipe_name}"
            img_path = output_dir / "images" / split / f"{image_id}.jpg"
            lbl_path = output_dir / "labels" / split / f"{image_id}.txt"

            cv2.imwrite(str(img_path), aug_img)
            if not write_yolo_label(lbl_path, aug_boxes, w, h):
                img_path.unlink(missing_ok=True)
                continue

            manifest_rows.append(
                {
                    "image_id": image_id,
                    "image_path": str(img_path.relative_to(repo_root)),
                    "label_path": str(lbl_path.relative_to(repo_root)),
                    "split": split,
                    "source": "intel_sample_video_augmented",
                    "license": INTEL_VIDEO_LICENSE,
                    "attribution": INTEL_ATTRIBUTION,
                    "parent_image_id": parent.parent_id,
                    "parent_video": str(parent.video_filename),
                    "parent_frame_index": parent.frame_index,
                    "synthetic": True,
                    "augmentations": [recipe_name],
                    "label_method": "pseudo_label_stock_yolo11n_on_bright_frame",
                    "num_boxes": len(aug_boxes),
                }
            )
            if split == "train":
                train_count += 1
            else:
                val_count += 1

    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "person"},
        "nc": 1,
    }
    with open(output_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    manifest = {
        "name": "lowlight_intel_augmented",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Low-light synthetic training set from Intel IoT DevKit sample-videos "
            "(CC-BY 4.0) with Albumentations augmentation. ExDark NOT included. "
            "Ultralytics bus.jpg NOT included."
        ),
        "license_policy": {
            "training_sources": "Intel sample-videos (CC-BY 4.0) only",
            "exdark": "benchmark only — not in this manifest",
            "ultralytics_sample_images": "excluded (bus.jpg not used)",
        },
        "attribution": INTEL_ATTRIBUTION,
        "source_videos": [
            {
                "filename": v.filename,
                "description": v.description,
                "license": INTEL_VIDEO_LICENSE,
                "url": intel_video_url(v.filename),
            }
            for v in INTEL_TRAINING_VIDEOS
        ],
        "augmentation_recipes": recipes,
        "frame_interval": frame_interval,
        "variants_per_parent": variants,
        "val_ratio": val_ratio,
        "num_parent_frames": len(all_parents),
        "num_train_images": train_count,
        "num_val_images": val_count,
        "images": manifest_rows,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frame-interval", type=int, default=6)
    parser.add_argument("--variants", type=int, default=4, choices=range(1, 5))
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--label-confidence", type=float, default=0.5)
    args = parser.parse_args()

    print("Phase 2 Step 2 — prepare training data")
    print("Sources: Intel IoT DevKit sample-videos (CC-BY 4.0) ONLY")
    print("Excluded: ExDark, Ultralytics bus.jpg, COCO images\n")

    manifest = build_dataset(
        output_dir=args.output,
        repo_root=REPO_ROOT,
        frame_interval=args.frame_interval,
        variants=args.variants,
        val_ratio=args.val_ratio,
        label_confidence=args.label_confidence,
    )

    print("\nTraining dataset ready:")
    print(f"  Parent frames:     {manifest['num_parent_frames']}")
    print(f"  Train images:      {manifest['num_train_images']}")
    print(f"  Val images:        {manifest['num_val_images']}")
    print(f"  Manifest:          {args.output / 'manifest.json'}")
    print(f"  YOLO data.yaml:    {args.output / 'data.yaml'}")


if __name__ == "__main__":
    main()
