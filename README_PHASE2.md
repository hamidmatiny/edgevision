# Phase 2 — Synthetic Data & Detector Fine-Tuning

**Status:** Steps 2–3 complete. Fine-tuned weights **not wired into pipeline** (Step 4 deferred — and current numbers do not justify wiring).

## Results summary (ExDark benchmark — same protocol as Step 1)

| Model | Precision | Recall | F1 | AP@0.5 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stock `yolo11n.pt` | 0.9080 | **0.6525** | 0.7594 | **0.6360** | 385 | 39 | 205 |
| Fine-tuned (Intel CC-BY aug) | 0.5232 | 0.2102 | 0.2999 | 0.1691 | 124 | 113 | 466 |

**Honest read:** Augmentation-only training on well-lit Intel sample footage **did not** close the ExDark recall gap — it moved in the wrong direction. This is a legitimate, expected possible outcome given the synthetic/real domain gap documented below. **Do not wire `best.pt` into production** based on these numbers. Next options: self-recorded night footage, heavier simulation, or different training strategy — not Step 4 wiring.

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

If augmentation-only training on Intel CC-BY footage **does not meaningfully close the recall gap** on ExDark, that is a **legitimate, expected possible outcome** — not a failure of the implementation. Results are reported honestly either way in [`evaluation/baseline_metrics.md`](evaluation/baseline_metrics.md).

Heavier approaches (self-recorded night footage, simulation) are deferred unless this path proves insufficient.

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

Fine-tuned weights are **not** wired into `sentinel/detection/detector.py` until you verify the numbers (Step 4 — deferred).

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

## Deferred

- Step 4: wire fine-tuned weights into `detector.py` (after you verify numbers)
- DVC versioning (optional; manifest JSON is minimum provenance)
