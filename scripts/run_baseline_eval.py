#!/usr/bin/env python3
"""
Run Phase 2 Step 1 baseline evaluation: stock YOLO weights on the held-out
low-light test set.

Usage:
    python scripts/prepare_lowlight_testset.py
    python scripts/run_baseline_eval.py
    python scripts/run_baseline_eval.py --manifest evaluation/datasets/exdark_people_test/manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "evaluation" / "datasets" / "exdark_people_test" / "manifest.json"
DEFAULT_METRICS_MD = REPO_ROOT / "evaluation" / "baseline_metrics.md"

sys.path.insert(0, str(REPO_ROOT))

from sentinel.detection.detector import Detector  # noqa: E402
from sentinel.evaluation.dataset import load_eval_dataset  # noqa: E402
from sentinel.evaluation.metrics import DetectionRecord, evaluate_dataset  # noqa: E402


def run_baseline(
    manifest_path: Path,
    model_name: str = "yolo11n.pt",
    confidence_threshold: float = 0.4,
    device: str = "cpu",
    target_class: str = "person",
) -> dict:
    dataset = load_eval_dataset(manifest_path)
    detector = Detector(
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        device=device,
        target_classes=[target_class],
    )

    detections: list[DetectionRecord] = []
    ground_truths = []
    for entry in dataset.images:
        ground_truths.extend(entry.ground_truths)
        frame = cv2.imread(str(entry.image_path))
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {entry.image_path}")

        for det in detector.detect(frame):
            if det["class"] != target_class:
                continue
            detections.append(
                DetectionRecord(
                    image_id=entry.image_id,
                    class_name=det["class"],
                    bbox=tuple(det["bbox"]),
                    confidence=float(det["confidence"]),
                )
            )

    result = evaluate_dataset(ground_truths, detections, iou_threshold=0.5)
    return {
        "dataset": {
            "name": dataset.name,
            "manifest_path": str(manifest_path.relative_to(REPO_ROOT)),
            "description": dataset.description,
            "provenance": dataset.provenance,
            "split": dataset.split,
            "num_images": result.num_images,
            "num_ground_truth_boxes": result.num_ground_truths,
        },
        "model": {
            "weights": model_name,
            "confidence_threshold": confidence_threshold,
            "device": device,
            "target_class": target_class,
        },
        "metrics": {
            "iou_threshold": 0.5,
            "precision": round(result.precision, 4),
            "recall": round(result.recall, 4),
            "f1": round(result.f1, 4),
            "ap50": round(result.ap50, 4),
            "true_positives": result.true_positives,
            "false_positives": result.false_positives,
            "false_negatives": result.false_negatives,
            "num_predictions": result.num_predictions,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_baseline_markdown(report: dict, output_path: Path) -> None:
    ds = report["dataset"]
    model = report["model"]
    m = report["metrics"]

    content = f"""# Phase 2 — Baseline Detector Metrics (Step 1)

Generated: {report["generated_at_utc"]}

This file records **stock pretrained YOLO weights** performance on the held-out
low-light test set **before any Phase 2 fine-tuning**. Phase 2 Step 3 will add a
fine-tuned comparison row to this document.

## Test set

| Field | Value |
|---|---|
| Name | `{ds["name"]}` |
| Manifest | `{ds["manifest_path"]}` |
| Split | {ds["split"]} |
| Images | {ds["num_images"]} |
| Person GT boxes | {ds["num_ground_truth_boxes"]} |
| Provenance | {ds["provenance"]} |

{ds["description"]}

**Important:** This is **real low-light imagery** from ExDark (not a synthetic
darkening proxy). ExDark is licensed for non-commercial research use only.

## Evaluation setup

| Setting | Value |
|---|---|
| Model | `{model["weights"]}` (stock Ultralytics pretrained) |
| Confidence threshold | {model["confidence_threshold"]} (matches pipeline default) |
| Device | {model["device"]} |
| Target class | {model["target_class"]} |
| IoU threshold | {m["iou_threshold"]} (standard detection matching) |
| Metrics | Precision, Recall, F1, AP@0.5 (101-point interpolated PR) |

Matching is greedy per image: predictions sorted by confidence, each matched to
the best unmatched ground-truth box with IoU ≥ {m["iou_threshold"]}.

## Baseline results (stock weights)

| Model | Precision | Recall | F1 | AP@0.5 | TP | FP | FN | Predictions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `{model["weights"]}` | **{m["precision"]:.4f}** | **{m["recall"]:.4f}** | **{m["f1"]:.4f}** | **{m["ap50"]:.4f}** | {m["true_positives"]} | {m["false_positives"]} | {m["false_negatives"]} | {m["num_predictions"]} |

## Fine-tuned comparison (Step 3 — pending)

| Model | Precision | Recall | F1 | AP@0.5 | Notes |
|---|---:|---:|---:|---:|---|
| *(not yet run)* | — | — | — | — | Added after Step 3 fine-tuning |

## Reproduce

```bash
python scripts/prepare_lowlight_testset.py
python scripts/run_baseline_eval.py
cat evaluation/baseline_metrics.md
```

Machine-readable JSON snapshot: `evaluation/baseline_metrics.json`
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-md", type=Path, default=DEFAULT_METRICS_MD)
    parser.add_argument("--output-json", type=Path, default=REPO_ROOT / "evaluation" / "baseline_metrics.json")
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"Manifest not found: {args.manifest}")
        print("Run: python scripts/prepare_lowlight_testset.py")
        sys.exit(1)

    report = run_baseline(
        manifest_path=args.manifest,
        model_name=args.model,
        confidence_threshold=args.confidence,
        device=args.device,
    )

    write_baseline_markdown(report, args.output_md)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    m = report["metrics"]
    print("Baseline evaluation complete (stock weights)")
    print(f"  Precision={m['precision']:.4f}  Recall={m['recall']:.4f}  "
          f"F1={m['f1']:.4f}  AP@0.5={m['ap50']:.4f}")
    print(f"  TP={m['true_positives']}  FP={m['false_positives']}  FN={m['false_negatives']}")
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
