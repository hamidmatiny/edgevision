#!/usr/bin/env python3
"""
Evaluate fine-tuned weights on the ExDark benchmark (same protocol as Step 1).

Updates the fine-tuned comparison row in evaluation/baseline_metrics.md.

Usage:
    python scripts/run_finetuned_eval.py
    python scripts/run_finetuned_eval.py --weights data/training/runs/lowlight_finetune/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "evaluation" / "datasets" / "exdark_people_test" / "manifest.json"
DEFAULT_METRICS_MD = REPO_ROOT / "evaluation" / "baseline_metrics.md"
DEFAULT_METRICS_JSON = REPO_ROOT / "evaluation" / "baseline_metrics.json"
DEFAULT_WEIGHTS = (
    REPO_ROOT / "data" / "training" / "runs" / "lowlight_finetune" / "weights" / "best.pt"
)

sys.path.insert(0, str(REPO_ROOT))

from scripts.run_baseline_eval import run_baseline  # noqa: E402


def update_comparison_markdown(
    md_path: Path,
    weights_path: Path,
    report: dict,
    training_manifest: Path | None,
) -> None:
    content = md_path.read_text(encoding="utf-8")
    m = report["metrics"]
    model_label = str(weights_path.relative_to(REPO_ROOT))
    notes = (
        "Intel CC-BY augmented training; ExDark benchmark only. "
        "See README_PHASE2.md for synthetic-data limitations."
    )
    if training_manifest and training_manifest.exists():
        notes += f" Training manifest: `{training_manifest.relative_to(REPO_ROOT)}`."

    new_row = (
        f"| `{model_label}` | **{m['precision']:.4f}** | **{m['recall']:.4f}** | "
        f"**{m['f1']:.4f}** | **{m['ap50']:.4f}** | {notes} |"
    )

    pattern = re.compile(
        r"(\| Model \| Precision \| Recall \| F1 \| AP@0\.5 \| Notes \|\n"
        r"\|---\|---:|---:|---:|---:|---\|\n)"
        r"\| \*\(not yet run\)\* \|[^\n]+\|\n",
        re.MULTILINE,
    )
    if not pattern.search(content):
        raise RuntimeError("Could not find pending fine-tuned comparison row in baseline_metrics.md")

    content = pattern.sub(rf"\1{new_row}\n", content, count=1)

    # Update section title from pending to complete
    content = content.replace(
        "## Fine-tuned comparison (Step 3 — pending)",
        "## Fine-tuned comparison (Step 3)",
    )
    content = content.replace(
        "Phase 2 Step 3 will add a\nfine-tuned comparison row to this document.",
        "Phase 2 Step 3 fine-tuned comparison is recorded below.",
    )

    md_path.write_text(content, encoding="utf-8")


def append_finetuned_json(json_path: Path, weights_path: Path, report: dict) -> None:
    payload: dict
    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = {}

    payload["finetuned"] = {
        "weights": str(weights_path.relative_to(REPO_ROOT)),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **report,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-md", type=Path, default=DEFAULT_METRICS_MD)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_METRICS_JSON)
    parser.add_argument(
        "--training-manifest",
        type=Path,
        default=REPO_ROOT / "data" / "training" / "lowlight_yolo" / "manifest.json",
    )
    args = parser.parse_args()

    if not args.weights.exists():
        print(f"Weights not found: {args.weights}")
        print("Run: python scripts/run_finetune.py")
        sys.exit(1)

    if not args.manifest.exists():
        print(f"Benchmark manifest not found: {args.manifest}")
        sys.exit(1)

    report = run_baseline(
        manifest_path=args.manifest,
        model_name=str(args.weights.resolve()),
        confidence_threshold=args.confidence,
        device=args.device,
    )

    update_comparison_markdown(
        args.output_md,
        args.weights,
        report,
        args.training_manifest,
    )
    append_finetuned_json(args.output_json, args.weights, report)

    m = report["metrics"]
    baseline_json = args.output_json
    delta_recall = delta_ap = None
    if baseline_json.exists():
        with open(baseline_json, encoding="utf-8") as f:
            baseline = json.load(f)
        if "metrics" in baseline:
            delta_recall = m["recall"] - baseline["metrics"]["recall"]
            delta_ap = m["ap50"] - baseline["metrics"]["ap50"]

    print("Fine-tuned evaluation complete (ExDark benchmark — same protocol as Step 1)")
    print(f"  Weights:   {args.weights.relative_to(REPO_ROOT)}")
    print(f"  Precision={m['precision']:.4f}  Recall={m['recall']:.4f}  "
          f"F1={m['f1']:.4f}  AP@0.5={m['ap50']:.4f}")
    print(f"  TP={m['true_positives']}  FP={m['false_positives']}  FN={m['false_negatives']}")
    if delta_recall is not None:
        print(f"  Δ Recall vs baseline: {delta_recall:+.4f}")
        print(f"  Δ AP@0.5 vs baseline: {delta_ap:+.4f}")
    print(f"Updated {args.output_md}")


if __name__ == "__main__":
    main()
