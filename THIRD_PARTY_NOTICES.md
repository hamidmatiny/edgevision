# Third-Party Notices

This file documents external data, models, and assets used by EdgeVision / SENTINEL
with licensing implications for commercial deployment.

---

## Intel IoT DevKit sample-videos (Phase 2 training data)

**Used for:** Phase 2 low-light detector fine-tuning — frame extraction and Albumentations augmentation.

**Repository:** https://github.com/intel-iot-devkit/sample-videos

**License:** Creative Commons Attribution 4.0 International (**CC-BY 4.0**)

**Commercial use:** Permitted with attribution.

**Attribution (required):** Sample video footage from Intel IoT DevKit sample-videos (https://github.com/intel-iot-devkit/sample-videos), licensed under CC-BY 4.0.

**Videos used in training pipeline** (see `sentinel/training/intel_sources.py`):

| File | Purpose |
|---|---|
| `person-bicycle-car-detection.mp4` | Outdoor pedestrian / vehicle scene |
| `people-detection.mp4` | Multi-person indoor-style scene |
| `one-by-one-person-detection.mp4` | Sequential single-person entries |
| `face-demographics-walking.mp4` | People walking at varying distances |

**Training manifest:** `data/training/lowlight_yolo/manifest.json` (generated locally; not committed).

---

## Exclusively Dark Image Dataset — ExDark (evaluation / benchmark only)

**Used for:** Phase 2 held-out **benchmark** only (`evaluation/datasets/exdark_people_test/`).

**Repository:** https://github.com/cs-chan/Exclusively-Dark-Image-Dataset

**License:** **Non-commercial research use only**

**Commercial use / fine-tuning:** **NOT permitted.** ExDark images must never appear in training manifests or fine-tuning runs for production weights.

---

## Ultralytics YOLO pretrained weights

**Used for:** Stock detector (`yolo11n.pt`), pseudo-labeling source frames, fine-tune base weights.

**Package:** https://github.com/ultralytics/ultralytics (AGPL-3.0 for the software)

**Note:** Pretrained weights are used as a starting checkpoint. Phase 2 fine-tuning updates weights using Intel CC-BY augmented data only.

---

## Ultralytics sample images — NOT used for training

**Examples:** `bus.jpg` and other images at https://ultralytics.com/images/

**Status:** Used only by `scripts/generate_smoke_test_video.py` for Phase 1 smoke-test video generation. **Explicitly excluded** from Phase 2 training pipeline (`scripts/prepare_training_data.py`).

---

## Albumentations

**Used for:** Low-light image augmentation in Phase 2 training data pipeline.

**Repository:** https://github.com/albumentations-team/albumentations

**License:** MIT License

---

## Moondream 2 — proposed Phase 3 VLM (not yet integrated)

**Planned use:** Stage 3 contextual verifier (local inference only).

**Repository:** https://huggingface.co/vikhyatk/moondream2

**Revision (pinned for production):** `2025-06-21`

**License:** **Apache License 2.0**

**Commercial use:** Permitted with attribution and license notice preservation. See [`README_PHASE3.md`](README_PHASE3.md) Step 3.1 review.

**Attribution (required when integrated):** Vision-language verification uses Moondream 2 (vikhyatk/moondream2), licensed under Apache 2.0.

**Not used:** Moondream cloud API (`api.moondream.ai`) — on-prem requirement.

---

## SmolVLM-256M-Instruct — Phase 3 fallback candidate (not integrated)

**Repository:** https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct

**License:** **Apache License 2.0**

**Commercial use:** Permitted with attribution. Fallback only if Moondream latency/quality trade-off requires it.

