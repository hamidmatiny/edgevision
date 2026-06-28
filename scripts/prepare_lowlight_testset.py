#!/usr/bin/env python3
"""
Prepare the Phase 2 held-out low-light evaluation set.

Primary source: ExDark (Exclusively Dark Image Dataset) — real low-light images
with human-annotated bounding boxes. Uses the official People class, Testing split.

Downloads (if missing):
  - ExDark groundtruth zip (~5 MB)
  - ExDark image zip (~1.5 GB) — required once; cached under data/exdark/

Outputs:
  evaluation/datasets/exdark_people_test/manifest.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "exdark"
OUTPUT_MANIFEST = REPO_ROOT / "evaluation" / "datasets" / "exdark_people_test" / "manifest.json"

GROUNDTRUTH_URL = "https://drive.google.com/uc?id=1P3iO3UYn7KoBi5jiUkogJq96N6maZS1i"
IMAGES_URL = "https://drive.google.com/uc?id=1BHmPgu8EsHoFDDkMGLVoXIlCth2dW6Yx"
IMAGECLASSLIST_URL = (
    "https://raw.githubusercontent.com/cs-chan/Exclusively-Dark-Image-Dataset/"
    "master/Groundtruth/imageclasslist.txt"
)

sys.path.insert(0, str(REPO_ROOT))

from sentinel.evaluation.exdark import parse_exdark_annotation_file  # noqa: E402


def _run_gdown(url: str, output: Path) -> None:
    if output.exists():
        print(f"Already present: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    gdown = REPO_ROOT / ".venv" / "bin" / "gdown"
    cmd = [str(gdown), url, "-O", str(output)]
    print("Downloading:", output.name)
    subprocess.run(cmd, check=True)


def _ensure_exdark_assets() -> None:
    gt_zip = DATA_DIR / "exdark_groundtruth.zip"
    img_zip = DATA_DIR / "exdark_images.zip"
    classlist = DATA_DIR / "imageclasslist.txt"

    _run_gdown(GROUNDTRUTH_URL, gt_zip)
    _run_gdown(IMAGES_URL, img_zip)

    if not classlist.exists():
        import urllib.request

        classlist.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(IMAGECLASSLIST_URL, classlist)

    gt_dir = DATA_DIR / "groundtruth" / "ExDark_Annno"
    if not gt_dir.exists():
        print("Extracting groundtruth...")
        with zipfile.ZipFile(gt_zip) as zf:
            zf.extractall(DATA_DIR / "groundtruth")

    img_root = DATA_DIR / "images" / "ExDark" / "People"
    if not img_root.exists():
        print("Extracting images (this may take a minute)...")
        with zipfile.ZipFile(img_zip) as zf:
            zf.extractall(DATA_DIR / "images")


def _load_people_testing_filenames(classlist_path: Path) -> list[str]:
    filenames: list[str] = []
    with open(classlist_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Name"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            filename, class_id, _light, _inout, split_id = parts[:5]
            if int(class_id) == 11 and int(split_id) == 3:
                filenames.append(filename)
    return sorted(filenames)


def _annotation_path(filename: str, anno_root: Path) -> Path:
    stem = Path(filename).name
    candidates = [
        anno_root / "People" / f"{stem}.txt",
        anno_root / "People" / f"{stem.lower()}.txt",
        anno_root / "People" / f"{stem.upper()}.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def build_manifest(max_images: int | None = None) -> Path:
    _ensure_exdark_assets()

    classlist = DATA_DIR / "imageclasslist.txt"
    anno_root = DATA_DIR / "groundtruth" / "ExDark_Annno"
    image_root = DATA_DIR / "images" / "ExDark" / "People"

    filenames = _load_people_testing_filenames(classlist)
    if max_images is not None:
        filenames = filenames[:max_images]

    images = []
    skipped = 0
    for filename in filenames:
        image_path = image_root / filename
        if not image_path.exists():
            alt = image_root / filename.lower()
            if alt.exists():
                image_path = alt
            else:
                skipped += 1
                continue

        anno_path = _annotation_path(filename, anno_root)
        gts = parse_exdark_annotation_file(anno_path, image_id=filename, target_class="person")
        if not gts:
            skipped += 1
            continue

        rel_image_path = image_path.relative_to(REPO_ROOT)
        images.append(
            {
                "image_id": filename,
                "image_path": str(rel_image_path),
                "ground_truths": [
                    {"class_name": gt.class_name, "bbox": list(gt.bbox)}
                    for gt in gts
                ],
                "metadata": {
                    "source": "ExDark",
                    "class_folder": "People",
                    "split": "testing",
                    "lighting": "low-light (real)",
                },
            }
        )

    manifest = {
        "name": "exdark_people_test",
        "description": (
            "ExDark People class, official Testing split. Real low-light/night "
            "images with human-annotated person bounding boxes."
        ),
        "provenance": (
            "Exclusively Dark Image Dataset (ExDark), non-commercial research use. "
            "See https://github.com/cs-chan/Exclusively-Dark-Image-Dataset"
        ),
        "split": "testing",
        "target_class": "person",
        "iou_threshold": 0.5,
        "images": images,
        "stats": {
            "requested_filenames": len(filenames),
            "included_images": len(images),
            "skipped_missing_or_empty": skipped,
            "total_person_boxes": sum(len(img["ground_truths"]) for img in images),
        },
    }

    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote manifest: {OUTPUT_MANIFEST}")
    print(
        f"  images={len(images)}  person_boxes={manifest['stats']['total_person_boxes']}  "
        f"skipped={skipped}"
    )
    return OUTPUT_MANIFEST


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap for quick smoke runs (default: full Testing split)",
    )
    args = parser.parse_args()
    build_manifest(max_images=args.max_images)


if __name__ == "__main__":
    main()
