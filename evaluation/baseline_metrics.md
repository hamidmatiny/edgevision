# Phase 2 — Baseline Detector Metrics (Step 1)

Generated: 2026-06-28T01:52:52.223457+00:00

This file records **stock pretrained YOLO weights** performance on the held-out
low-light test set **before any Phase 2 fine-tuning**. Phase 2 Step 3 will add a
fine-tuned comparison row to this document.

## Test set

| Field | Value |
|---|---|
| Name | `exdark_people_test` |
| Manifest | `evaluation/datasets/exdark_people_test/manifest.json` |
| Split | testing |
| Images | 209 |
| Person GT boxes | 590 |
| Provenance | Exclusively Dark Image Dataset (ExDark), non-commercial research use. See https://github.com/cs-chan/Exclusively-Dark-Image-Dataset |

ExDark People class, official Testing split. Real low-light/night images with human-annotated person bounding boxes.

**Important:** This is **real low-light imagery** from ExDark (not a synthetic
darkening proxy). ExDark is licensed for non-commercial research use only.

## Evaluation setup

| Setting | Value |
|---|---|
| Model | `yolo11n.pt` (stock Ultralytics pretrained) |
| Confidence threshold | 0.4 (matches pipeline default) |
| Device | cpu |
| Target class | person |
| IoU threshold | 0.5 (standard detection matching) |
| Metrics | Precision, Recall, F1, AP@0.5 (101-point interpolated PR) |

Matching is greedy per image: predictions sorted by confidence, each matched to
the best unmatched ground-truth box with IoU ≥ 0.5.

## Baseline results (stock weights)

| Model | Precision | Recall | F1 | AP@0.5 | TP | FP | FN | Predictions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `yolo11n.pt` | **0.9080** | **0.6525** | **0.7594** | **0.6360** | 385 | 39 | 205 | 424 |

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
