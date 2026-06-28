# Phase 2 — Synthetic Data & Detector Fine-Tuning

**Status:** ✅ Complete — **negative result, valuable diagnosis; stock `yolo11n.pt` retained in pipeline.**

Fine-tuned weights exist under `data/training/runs/lowlight_finetune/weights/best.pt` for reference but are **not** wired into `sentinel/detection/detector.py`.

## Findings (Phase 2 close-out)

### ExDark benchmark (out-of-domain) — did not improve

| Model | Precision | Recall | F1 | AP@0.5 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stock `yolo11n.pt` | 0.9080 | **0.6525** | 0.7594 | **0.6360** | 385 | 39 | 205 |
| Fine-tuned (Intel CC-BY aug) | 0.5232 | 0.2102 | 0.2999 | 0.1691 | 124 | 113 | 466 |

Fine-tuning **hurt** real low-light recall on ExDark (Δ recall −0.44, Δ AP@0.5 −0.47). Stock weights remain in production code.

### In-domain validation (augmented Intel holdout) — strong throughout

Training val split draws from the **same augmented-Intel-frame pool** as training (`data/training/lowlight_yolo/images/val`, 200 images). Per-epoch metrics from `data/training/runs/lowlight_finetune/results.csv`:

| Epoch | Val precision | Val recall | Val mAP@0.5 | Val mAP@0.5:0.95 |
|---:|---:|---:|---:|---:|
| 1 | 0.8920 | 0.5615 | 0.6637 | 0.4286 |
| 15 | 0.9148 | 0.7664 | 0.8535 | 0.6995 |
| 30 (final) | **0.9514** | **0.8016** | **0.8841** | **0.7584** |

Val metrics were **strong and stable for the entire run** — improving from epoch 1 through 30 with no train/val divergence and no late-epoch collapse. Val losses (`val/box_loss`, `val/cls_loss`, `val/dfl_loss`) trended down consistently.

### Diagnosis — data diversity / domain transfer, not optimization

This rules out a **training-procedure failure** (bad learning rate, missing early stopping, broken labels, etc.) as the primary cause. The model successfully learned its **narrow training distribution** (294 parent frames, 4 Intel CC-BY videos, synthetically darkened via Albumentations) extremely well — and that distribution **does not overlap enough** with real ExDark night imagery to transfer.

This is a **synthetic-data-diversity / domain-transfer limitation**, not a fixable hyperparameter issue. Retraining on the same Intel-derived pool with different epochs, LR, or augment settings would likely repeat the same outcome.

**Implication for future low-light work:** the next serious attempt at improving low-light person recall needs **genuinely diverse real night footage** (or much more varied synthetic data with broader scene/lighting coverage) — not another pass on this Intel-derived pool with tweaked settings. Phase 2's primary FP-reduction path for the product is **Phase 3 (VLM contextual verifier)** on top of stock YOLO, not these fine-tuned weights.

## Licensing policy (read before any training work)

### ExDark — benchmark only, never training data

The held-out test set [`evaluation/datasets/exdark_people_test/manifest.json`](evaluation/datasets/exdark_people_test/manifest.json) uses the [Exclusively Dark (ExDark) dataset](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset).

| Allowed | Not allowed |
|---|---|
| Running evaluation scripts to **measure** precision/recall/AP@0.5 | Including ExDark images in YOLO fine-tuning |
| Before/after comparisons on the **same** ExDark test split | Shipping model weights trained on ExDark |
| Internal R&D benchmarking | Any commercial deployment where weights are derivatives of ExDark |

**Why:** ExDark is licensed for **non-commercial research use only**. Fine-tuning production detector weights on ExDark would create a derivative work with real legal exposure at sale, customer deployment, or diligence.

### Training data — Intel sample-videos (CC-BY 4.0) + Albumentations

**Approved training source:** frames extracted from [Intel IoT DevKit sample-videos](https://github.com/intel-iot-devkit/sample-videos), licensed **CC-BY 4.0 (Attribution)** — commercial use permitted with attribution.

**Attribution (required):** Sample video footage from Intel IoT DevKit sample-videos (https://github.com/intel-iot-devkit/sample-videos), licensed under CC-BY 4.0. See also [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

**Explicitly excluded from Step 2 training:**

| Source | Status |
|---|---|
| ExDark | Benchmark only |
| Ultralytics `bus.jpg` and other sample images | **NOT used** — only in Phase 1 `generate_smoke_test_video.py` |
| COCO / random web images | Not used |

The augmentation pipeline (`scripts/prepare_training_data.py`) resolves videos only via `sentinel/training/intel_sources.py`. It does not import or download Ultralytics sample images.

## Synthetic-data limitation (honest assessment)

Albumentations-based low-light augmentation **approximates** darkness (gamma, brightness, noise, color cast, blur). It does **not** fully replicate real low-light sensor behaviour:

- Real ISO noise patterns and demosaicing artefacts
- Motion blur from long exposures in actual night footage
- Artificial-light glare, bloom, and mixed colour temperature
- HDR/auto-exposure behaviour of security cameras

**ExDark remains the real-world benchmark precisely because of this gap.** It contains genuine low-light/night imagery with human-annotated boxes.

If augmentation-only training on Intel CC-BY footage does not transfer to real low-light benchmarks, that is a **legitimate outcome** — documented in [`evaluation/baseline_metrics.md`](evaluation/baseline_metrics.md) and the Findings section above. Phase 2 confirmed this empirically.

Future low-light detector work requires **broader real or synthetic night data** — not hyperparameter tuning on the same Intel pool.

## Step 1 — baseline (stock weights)

See [`evaluation/baseline_metrics.md`](evaluation/baseline_metrics.md).

| Metric | Stock `yolo11n.pt` on ExDark People test |
|---|---:|
| Precision | 0.9080 |
| Recall | 0.6525 |
| AP@0.5 | 0.6360 |

## Step 2 — training data pipeline

```bash
python scripts/prepare_training_data.py
```

**What it does:**

1. Resolves Intel sample-videos (downloads to `data/training/sources/` if missing; uses repo-root `test_video.mp4` alias for `person-bicycle-car-detection.mp4`).
2. Extracts every 6th frame (~2 fps at 12 fps source).
3. Pseudo-labels **person** boxes on **bright source frames** with stock `yolo11n.pt` (conf ≥ 0.5).
4. Applies **4 Albumentations low-light recipes** per labeled parent frame (~4× expansion).
5. Writes YOLO dataset to `data/training/lowlight_yolo/` + versioned `manifest.json`.

Each manifest row includes: `source`, `license`, `parent_image_id`, `augmentations`, `synthetic`, `attribution`.

**Intel videos used:**

| File | Notes |
|---|---|
| `person-bicycle-car-detection.mp4` | Also available as `test_video.mp4` |
| `people-detection.mp4` | |
| `one-by-one-person-detection.mp4` | |
| `face-demographics-walking.mp4` | |

## Step 3 — fine-tune + benchmark eval

```bash
python scripts/run_finetune.py              # writes best.pt to data/training/runs/lowlight_finetune/
python scripts/run_finetuned_eval.py        # same ExDark protocol as Step 1; updates baseline_metrics.md
cat evaluation/baseline_metrics.md
```

Fine-tuned weights are **not** wired into `sentinel/detection/detector.py`. Stock `yolo11n.pt` remains the pipeline default (see root `README.md`).

## Files

| Path | Purpose |
|---|---|
| `sentinel/training/intel_sources.py` | Approved CC-BY video list + license constants |
| `sentinel/training/augmentations.py` | Albumentations low-light recipes |
| `scripts/prepare_training_data.py` | Build YOLO dataset + manifest |
| `scripts/run_finetune.py` | Fine-tune on augmented Intel data |
| `scripts/run_finetuned_eval.py` | ExDark before/after comparison |
| `scripts/run_baseline_eval.py` | Step 1 stock-weight eval |
| `evaluation/baseline_metrics.md` | Baseline + fine-tuned comparison table |
| `THIRD_PARTY_NOTICES.md` | License attributions |
| `tests/test_training_augmentations.py` | Augmentation unit tests |

## Reproduce full Phase 2 eval loop

```bash
pip install -e ".[dev,train]"
python scripts/prepare_lowlight_testset.py   # ExDark benchmark (eval only)
python scripts/run_baseline_eval.py          # Step 1 baseline
python scripts/prepare_training_data.py      # Step 2 training data
python scripts/run_finetune.py               # Step 3 fine-tune
python scripts/run_finetuned_eval.py         # Step 3 ExDark eval + comparison row
```

## Phase 2 outcome

**Closed.** Negative result on ExDark transfer; in-domain training succeeded. Stock detector retained. Step 4 (wire fine-tuned weights) **will not be pursued** on this data pool.

## Deferred

- DVC versioning (optional; manifest JSON is minimum provenance)
- Future low-light detector attempt (requires new data sources — see Findings)
