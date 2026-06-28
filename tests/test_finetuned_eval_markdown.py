"""Tests for idempotent fine-tuned eval markdown updates."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_finetuned_eval import update_comparison_markdown  # noqa: E402


def test_update_comparison_markdown_overwrites_existing_row(tmp_path: Path):
    md = tmp_path / "baseline_metrics.md"
    md.write_text(
        """## Fine-tuned comparison (Step 3)

| Model | Precision | Recall | F1 | AP@0.5 | Notes |
|---|---:|---:|---:|---:|---|
| `data/training/runs/lowlight_finetune/weights/best.pt` | **0.5232** | **0.2102** | **0.2999** | **0.1691** | old notes |
""",
        encoding="utf-8",
    )
    report = {
        "metrics": {
            "precision": 0.6000,
            "recall": 0.3000,
            "f1": 0.4000,
            "ap50": 0.2500,
        }
    }
    weights = Path("data/training/runs/lowlight_finetune/weights/best.pt")

    update_comparison_markdown(md, weights, report, training_manifest=None)

    text = md.read_text(encoding="utf-8")
    assert "**0.6000**" in text
    assert "**0.3000**" in text
    assert "0.5232" not in text
    assert text.count("| `data/training/runs/lowlight_finetune/weights/best.pt` |") == 1


def test_update_comparison_markdown_replaces_pending_placeholder(tmp_path: Path):
    md = tmp_path / "baseline_metrics.md"
    md.write_text(
        """## Fine-tuned comparison (Step 3 — pending)

| Model | Precision | Recall | F1 | AP@0.5 | Notes |
|---|---:|---:|---:|---:|---|
| *(not yet run)* | — | — | — | — | Added after Step 3 fine-tuning |
""",
        encoding="utf-8",
    )
    report = {
        "metrics": {
            "precision": 0.5232,
            "recall": 0.2102,
            "f1": 0.2999,
            "ap50": 0.1691,
        }
    }
    weights = Path("data/training/runs/lowlight_finetune/weights/best.pt")

    update_comparison_markdown(md, weights, report, training_manifest=None)

    text = md.read_text(encoding="utf-8")
    assert "## Fine-tuned comparison (Step 3)" in text
    assert "*(not yet run)*" not in text
    assert "**0.5232**" in text
