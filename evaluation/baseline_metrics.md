# Phase 2 — Baseline Detector Metrics (Step 1)

Generated: 2026-06-28T01:52:52.223457+00:00

This file records **stock pretrained YOLO weights** performance on the held-out
low-light test set **before any Phase 2 fine-tuning**. Phase 2 Step 3 fine-tuned comparison is recorded below.

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

## Licensing — benchmark only (not training data)

ExDark is licensed for **non-commercial research use only**. See the ExDark dataset README:
https://github.com/cs-chan/Exclusively-Dark-Image-Dataset

| Use | Allowed? |
|---|---|
| Measure stock / fine-tuned detector on this held-out test set | **Yes** — normal benchmarking |
| Include ExDark images in YOLO fine-tuning | **No** — would create commercially encumbered weights |
| Deploy fine-tuned weights trained on ExDark to paying customers | **No** |

**This project targets a commercial product.** ExDark remains our **evaluation benchmark** for before/after recall/precision comparisons. It must **never** appear in the Phase 2 training manifest. Training data will use commercial-safe sources only (self-recorded footage and/or synthetic augmentation of owned images). See [`README_PHASE2.md`](../README_PHASE2.md).

**Important:** This is **real low-light imagery** from ExDark (not a synthetic darkening proxy).

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

## Fine-tuned comparison (Step 3)

| Model | Precision | Recall | F1 | AP@0.5 | Notes |
|---|---:|---:|---:|---:|---|
| `data/training/runs/lowlight_finetune/weights/best.pt` | **0.5232** | **0.2102** | **0.2999** | **0.1691** | Intel CC-BY augmented training; ExDark benchmark only. See README_PHASE2.md for synthetic-data limitations. Training manifest: `data/training/lowlight_yolo/manifest.json`. |

## Reproduce

```bash
python scripts/prepare_lowlight_testset.py
python scripts/run_baseline_eval.py
python scripts/prepare_training_data.py
python scripts/run_finetune.py
python scripts/run_finetuned_eval.py
cat evaluation/baseline_metrics.md
```

Machine-readable JSON snapshot: `evaluation/baseline_metrics.json`
