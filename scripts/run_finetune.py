#!/usr/bin/env python3
"""
Fine-tune YOLO on the commercial-safe low-light training set (Phase 2 Step 3).

Requires: python scripts/prepare_training_data.py

Does NOT use ExDark or Ultralytics sample images for training.

Usage:
    python scripts/prepare_training_data.py
    python scripts/run_finetune.py
    python scripts/run_finetune.py --epochs 30 --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "data" / "training" / "lowlight_yolo" / "data.yaml"
DEFAULT_PROJECT = REPO_ROOT / "data" / "training" / "runs"
DEFAULT_NAME = "lowlight_finetune"
DEFAULT_MANIFEST = REPO_ROOT / "data" / "training" / "lowlight_yolo" / "manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--base-model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    if not args.data.exists():
        print(f"Training data not found: {args.data}")
        print("Run: python scripts/prepare_training_data.py")
        sys.exit(1)

    if not DEFAULT_MANIFEST.exists():
        print(f"Training manifest not found: {DEFAULT_MANIFEST}")
        sys.exit(1)

    args.project.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics required: pip install ultralytics")
        sys.exit(1)

    print("Phase 2 Step 3 — fine-tune on Intel CC-BY augmented data")
    print(f"  Base model:  {args.base_model}")
    print(f"  Data:        {args.data}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Device:      {args.device}")
    print("  ExDark:      NOT in training set")
    print("  bus.jpg:     NOT in training set\n")

    model = YOLO(args.base_model)
    results = model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project.resolve()),
        name=args.name,
        patience=args.patience,
        exist_ok=True,
        verbose=True,
        # Single-class person fine-tune
        classes=[0],
    )

    run_dir = Path(results.save_dir) if results is not None else args.project / args.name
    best_weights = run_dir / "weights" / "best.pt"
    last_weights = run_dir / "weights" / "last.pt"

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "training_data": str(args.data.relative_to(REPO_ROOT)),
        "training_manifest": str(DEFAULT_MANIFEST.relative_to(REPO_ROOT)),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "run_dir": str(run_dir.relative_to(REPO_ROOT)),
        "best_weights": str(best_weights.relative_to(REPO_ROOT)) if best_weights.exists() else None,
        "last_weights": str(last_weights.relative_to(REPO_ROOT)) if last_weights.exists() else None,
        "license_note": "Trained on Intel sample-videos (CC-BY 4.0) augmented data only. ExDark not used.",
    }
    meta_path = run_dir / "finetune_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nFine-tuning complete.")
    print(f"  Best weights: {best_weights}")
    print(f"  Metadata:     {meta_path}")
    print("\nNext: python scripts/run_finetuned_eval.py")


if __name__ == "__main__":
    main()
