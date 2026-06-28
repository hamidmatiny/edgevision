# Phase 2 — Synthetic Data & Detector Fine-Tuning

**Status:** Step 1 complete (baseline measured). Step 2+ pending data-plan approval.

## Licensing policy (read before any training work)

### ExDark — benchmark only, never training data

The held-out test set [`evaluation/datasets/exdark_people_test/manifest.json`](evaluation/datasets/exdark_people_test/manifest.json) uses the [Exclusively Dark (ExDark) dataset](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset).

| Allowed | Not allowed |
|---|---|
| Running `scripts/run_baseline_eval.py` to **measure** precision/recall/AP@0.5 | Including ExDark images in YOLO fine-tuning |
| Before/after comparisons on the **same** ExDark test split | Shipping model weights trained on ExDark |
| Internal R&D benchmarking | Any commercial deployment where weights are derivatives of ExDark |

**Why:** ExDark is licensed for **non-commercial research use only**. Fine-tuning production detector weights on ExDark would create a derivative work with real legal exposure at sale, customer deployment, or diligence.

Step 1 baseline numbers in [`evaluation/baseline_metrics.md`](evaluation/baseline_metrics.md) remain valid as an **evaluation benchmark**. They do not authorize training on ExDark.

### Training data (Step 2+) — commercial-safe sources only

Any image/frame used to **update model weights** must be:

1. Footage/images **you own** (self-recorded), or
2. Explicitly licensed for **commercial use and derivative models**, with license documented in the training manifest, or
3. Fully **synthetic** (procedurally generated or augmented from commercial-safe inputs only)

When a new external dataset is proposed, the license must be stated explicitly in this document **before** it enters the training manifest — same standard as the ExDark flag above.

## Step 1 complete — baseline (stock weights)

See [`evaluation/baseline_metrics.md`](evaluation/baseline_metrics.md).

| Metric | Stock `yolo11n.pt` on ExDark People test |
|---|---:|
| Precision | 0.9080 |
| Recall | 0.6525 |
| AP@0.5 | 0.6360 |

Fine-tuning (Step 3) must beat these numbers on the **same ExDark benchmark** without training on ExDark.

## Proposed training data plan (awaiting approval)

**Do not implement until approved.**

### Primary source — self-recorded footage (recommended)

| Item | Proposal |
|---|---|
| **License** | You own the footage → full commercial rights |
| **Volume** | 8–12 clips × 30–60 s (≈4–8 min total) |
| **Scenarios** | (1) person loitering 5+ s in frame, (2) person walking through scene, (3) partial occlusion / edge of frame, (4) one vehicle pass if available |
| **Lighting** | At least half recorded at dusk/night or dim indoor — matches product use case |
| **Resolution** | 720p or 1080p; static camera angle similar to a fixed security cam |
| **Labels** | Extract frames (e.g. 2 fps), label person boxes in CVAT/Label Studio **or** pseudo-label from stock YOLO on a bright duplicate exposure (document which); all labels stored in a versioned manifest |

### Secondary source — synthetic low-light augmentation (commercial-safe)

| Item | Proposal |
|---|---|
| **License** | Derived only from **self-recorded** frames (or other approved commercial-safe inputs) |
| **Method** | Extend Phase 1 compositing pattern + [Albumentations](https://albumentations.ai/) transforms: underexposure, gamma, Gaussian noise, contrast compression, color-temperature shift |
| **Output** | 3–5× expansion of owned frames; manifest records `source_image_id`, `augmentation_pipeline`, `synthetic=true` |
| **Not used** | ExDark; random web images without license check |

### Explicitly not in training mix

| Source | Reason |
|---|---|
| **ExDark** | Non-commercial research only |
| **Ultralytics sample images (`bus.jpg`, etc.)** | Sourced from COCO-like collections; not verified for commercial derivative training |
| **COCO train/val images** | Per-image license heterogeneity; unsafe default for commercial weight training without legal review |
| **NightOwls** | Oxford license: non-commercial research, no redistribution — same class of problem as ExDark |

### Evaluation split (unchanged)

| Set | Role | Source |
|---|---|---|
| `exdark_people_test` | Held-out **benchmark** only | ExDark People / Testing |
| User footage holdout | Optional training-time val (not ExDark) | 15–20% of your clips, never augmented into train |

### Provenance tracking (Step 2 implementation)

Manifest JSON per training image (DVC deferred for Phase 2 unless you prefer it now):

```json
{
  "image_id": "...",
  "source": "self_recorded | synthetic_augment",
  "license": "owner: Hamid Matiny / commercial OK",
  "parent_image_id": null,
  "augmentations": ["RandomGamma", "GaussNoise"]
}
```

### Scope note (unchanged from phase plan)

We are **not** building Omniverse/Unity simulation in Phase 2. Heavier simulation is deferred until augmentation + owned footage fails to close the recall gap on the ExDark benchmark.

## Files (Step 1)

| Path | Purpose |
|---|---|
| `sentinel/evaluation/metrics.py` | Detection metrics (P/R/F1/AP@0.5) |
| `scripts/prepare_lowlight_testset.py` | Build ExDark **benchmark** manifest |
| `scripts/run_baseline_eval.py` | Stock-weight evaluation |
| `evaluation/baseline_metrics.md` | Baseline numbers + license notes |
| `tests/test_evaluation_metrics.py` | Metric unit tests |

## Reproduce baseline (benchmark only)

```bash
python scripts/prepare_lowlight_testset.py   # downloads ExDark to data/exdark/ (gitignored)
python scripts/run_baseline_eval.py
cat evaluation/baseline_metrics.md
```

## Known limitations

- ExDark benchmark ≠ your deployment camera FOV/lighting; it is a proxy for low-light person detection difficulty.
- ExDark **cannot** train production weights (see licensing policy above).
- Baseline uses `yolo11n.pt` at conf=0.4 on CPU; GPU numbers may differ slightly.

## Deferred

- Step 2 training pipeline implementation (pending data-plan approval)
- Step 3 fine-tuning and before/after table row
- Step 4 wiring improved weights into `detector.py`
- DVC versioning (optional; manifest JSON minimum for Phase 2)
