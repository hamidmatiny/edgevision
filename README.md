# EdgeVision — SENTINEL

**AI-native, VMS-agnostic edge security analytics.**  
A production-grade anomaly & threat detection layer for cameras you already own.

---

## The problem

90–99% of security camera alerts are false positives. Operators stop trusting the system. Real threats get missed. Legacy motion detection is the culprit — it fires on wind, shadows, animals, and headlights.

SENTINEL is built around one metric: **< 5% false positive rate** — the industry benchmark that most deployed systems never hit.

## How it works

A multi-stage verification pipeline runs entirely on-prem:

```
RTSP/ONVIF Camera Streams
        │
        ▼
[Stage 1] Fast Detector (YOLO + ByteTrack)
        — person/vehicle detection + stable track IDs
        │
        ▼
[Stage 2] Rule Engine
        — zone geometry, dwell time, schedule
        — only forwards events that genuinely crossed a boundary
        │
        ▼
[Stage 3] VLM Contextual Verifier  ← Phase 3 (coming)
        — vision-language model confirms: "is this actually a person
          climbing a fence, or a shadow the detector misread?"
        │
        ▼
[Stage 4] Agentic Decision Layer
        — structured incident record + evidence clip
        — audit trail: detector confidence → rule trigger → VLM reasoning
```

Footage never leaves the site. Only structured metadata + optional low-res clips sync to the cloud control plane. Privacy-first by architecture, not just marketing.

## Current status

**Phase 1 complete** — core perception pipeline (detection + tracking + zone engine + evidence capture), verified with unit tests and local video smoke runs.

**Phase 2 complete** — synthetic low-light fine-tuning experiment. **Negative result:** ExDark recall regressed; in-domain val was strong (mAP@0.5 0.88 at epoch 30). Diagnosis: domain-transfer / data-diversity gap, not a training bug. **Stock `yolo11n.pt` retained** in `sentinel/detection/detector.py` — fine-tuned weights are **not** used in the pipeline.

**Phase 3 next** — VLM contextual verifier. Step 3.1 (model + license + latency spike) complete — see [`README_PHASE3.md`](README_PHASE3.md) for review before integration.

See [SPEC.md](SPEC.md) for success metrics, [README_PHASE1.md](README_PHASE1.md) and [README_PHASE2.md](README_PHASE2.md) for prior phase notes.

## Quickstart

```bash
# Install (Python 3.10–3.12 recommended)
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Download / regenerate the Phase 1 smoke-test clip (optional if using committed fixture)
python scripts/generate_smoke_test_video.py

# Run against a local video file
python scripts/run_local.py \
  --source tests/fixtures/smoke_test_loiter.mp4 \
  --camera-id cam1 \
  --frame-skip 2

# Run unit tests
pytest tests/ -v
```

Configure your camera sources in `config/cameras.yaml` and zone polygons in `config/zones.yaml`.

## Tech stack

| Layer | Choice |
|---|---|
| Detector | YOLO (Ultralytics) — exportable to TensorRT/ONNX for edge |
| Tracker | ByteTrack via `supervision` |
| Zone logic | Shapely (polygon) + custom dwell-time state machine |
| Video I/O | OpenCV |
| Edge runtime | Docker Compose (coming Phase 4) |
| Cloud backend | FastAPI + Postgres (coming Phase 5) |
| Dashboard | React + Tailwind (coming Phase 5) |

## Roadmap

| Phase | Status | Description |
|---|---|---|
| 0 | ✅ Done | Spec lock, repo scaffold |
| 1 | ✅ Done | Core perception pipeline |
| 2 | ✅ Done | Synthetic low-light fine-tune — **negative ExDark transfer; stock YOLO retained** |
| 3 | 🔜 Next | VLM contextual verifier |
| 4 | 🔜 | Edge containerisation (Docker) |
| 5 | 🔜 | Cloud control plane + dashboard |
| 6 | 🔜 | Explainability & audit trail |
| 7 | 🔜 | MLOps & continuous learning loop |
| 8 | 🔜 | Pilot packaging & GTM |

## License

MIT
