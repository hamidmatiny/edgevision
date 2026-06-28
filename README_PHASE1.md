# SENTINEL — Phase 1 Build Notes

## What was built

Phase 1 delivers the core local perception pipeline:

**Stage 1a — Ingestion (`sentinel/ingestion/stream_reader.py`)**
Handles both RTSP streams and local video files uniformly. Configurable frame-skip to control processing FPS. RTSP reconnect with exponential backoff — never crashes on connectivity loss. Yields a **stream timeline** timestamp (`raw_frame_index / fps`) alongside wall clock time so downstream stages measure dwell in source time, not processing speed.

**Stage 1b — Detection + Tracking (`sentinel/detection/`)**
- `detector.py`: Wraps Ultralytics YOLO (pretrained, no fine-tuning yet). Filters to person + vehicle classes only. Returns normalised dicts with bounding box, centroid, class, confidence.
- `tracker.py`: Wraps ByteTrack via the `supervision` library. Assigns stable track IDs across frames. Detections without a stable ID are returned with `track_id = -1`; the zone engine ignores these so they cannot start or reset dwell state.

**Stage 2 — Zone/Rule Engine (`sentinel/rules/zone_engine.py`)**
The core IP of the product. Evaluates each tracked detection against per-camera zone configs:
- Polygon containment check (Shapely)
- Schedule check ("always" or HH:MM-HH:MM with midnight-wrap support)
- Dwell-time state machine on the **stream timeline**: a track must be continuously inside a zone for ≥ N seconds of source time before becoming a candidate event. Re-entry resets the timer.
- Zone configs are loaded from the top-level `zones:` key in `config/zones.yaml` via `load_zones_for_camera()`.

**Stage 3 stub (`sentinel/pipeline.py` → `VLMVerifierStub`)**
Pass-through placeholder that always confirms candidates. Swap in the real VLM verifier in Phase 3 without touching any other code.

**Stage 4 — Evidence (`sentinel/events/`)**
- `event_builder.py`: Builds a structured IncidentRecord JSON (schema v1.0) with full audit chain.
- `clip_writer.py`: Maintains a rolling pre-trigger frame buffer. On trigger, accumulates post-trigger frames and writes an MP4 evidence clip. When the clip finishes writing, the pipeline patches the event JSON with the clip path.

## How to run

### 1. Install dependencies

Requires Python 3.10–3.12 (recommended). From the repo root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Get a test video

**Recommended — Intel sample (people + vehicles, 768×432):**
```bash
curl -L -o test_video.mp4 \
  "https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4"
```

**Alternative — record your own (30 seconds, walk through the configured zone):**
```bash
python -c "
import cv2, time
cap = cv2.VideoCapture(0)
out = cv2.VideoWriter('test_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
t = time.time()
while time.time() - t < 30:
    ret, frame = cap.read()
    if ret: out.write(frame)
cap.release(); out.release()
print('Saved test_video.mp4')
"
```

### 3. Zone config

Open `config/zones.yaml`. Default `cam1` zones:

| Zone | Polygon (approx.) | Schedule | Dwell | Classes |
|---|---|---|---|---|
| `perimeter_east` | Centre-left rectangle | `always` (dev) | 3s | person, vehicle |
| `restricted_storage` | Right side of frame | `always` | 2s | person |

For production after-hours perimeter monitoring, set `perimeter_east` schedule to `"22:00-06:00"`. Adjust polygons to match your camera resolution and scene layout.

### 4. Run the pipeline

```bash
python scripts/run_local.py \
  --source test_video.mp4 \
  --camera-id cam1 \
  --frame-skip 2 \
  --log-level INFO
```

### 5. Expected output

After a person or vehicle dwells inside a zone for ≥ configured seconds **of video time**:

```
2026-06-27 22:15:18 [INFO] sentinel.rules.zone_engine — CANDIDATE EVENT: track=1 zone='restricted_storage' camera=cam1 class=person dwell=3.7s conf=0.83 centroid=(694.7, 186.4)
2026-06-27 22:15:18 [INFO] sentinel.events.event_builder — Event record saved: evidence/cam1_<uuid>.json
2026-06-27 22:15:18 [INFO] sentinel.pipeline — [cam1] INCIDENT CONFIRMED — event_id=<uuid> ...
2026-06-27 22:15:19 [INFO] sentinel.pipeline — [cam1] Clip ready: evidence/cam1_<uuid>_<ts>.mp4
```

**Files created in `evidence/`:**
- `cam1_<uuid>.json` — structured incident record (`clip_path` populated after clip write)
- `cam1_<uuid>_<timestamp>.mp4` — short video clip (pre + post trigger)

**Example event JSON (after clip is written):**
```json
{
  "schema_version": "1.0",
  "event_id": "a3f7c2d1-...",
  "camera_id": "cam1",
  "zone_name": "restricted_storage",
  "track_id": 1,
  "detection_class": "person",
  "confidence": 0.8343,
  "centroid": [694.67, 186.45],
  "trigger_wall_time": 1782609318.7485702,
  "trigger_wall_time_iso": "2026-06-28T01:15:18.748570+00:00",
  "dwell_elapsed_seconds": 3.667,
  "clip_path": "evidence/cam1_a3f7c2d1-..._1782609319.mp4",
  "vlm_verification": null,
  "audit_chain": {
    "stage1_confidence": 0.8343,
    "stage2_zone_rule": "zone='restricted_storage' dwell=3.67s",
    "stage3_vlm": null,
    "final_decision": "candidate"
  }
}
```

### 6. Run unit tests

```bash
pytest tests/ -v
```

All zone engine, config loader, stream timeline, event builder, and pipeline helper tests should pass with no GPU or real video required.

## Known limitations (Phase 1)

- **Stock YOLO weights only.** No fine-tuning on industrial/night footage yet. Expect higher FP rate in challenging lighting. Phase 2 addresses this with synthetic data fine-tuning.
- **False positive rate not yet measured.** FP benchmark harness is Phase 3 (requires VLM verifier to compare Stage1+2 vs Stage1+2+3).
- **Stage 3 VLM is a stub.** All candidates are auto-confirmed. Phase 3 adds the contextual reasoning layer that is the primary FP reduction mechanism.
- **No cloud dashboard.** Phase 5.
- **Single-camera only.** Multi-camera: run one process per camera, or spawn a thread per camera — the pipeline is per-camera by design.
- **ByteTrack via `supervision` is deprecated** as of supervision v0.28 and will be removed in v0.30 — plan a tracker migration before upgrading supervision.
- **Default zone polygons are scene-specific.** Adjust `config/zones.yaml` for your camera FOV; the committed defaults target the Intel sample video and a centre/right split layout.

## Verification checkpoint

After running against a real test video, confirm:
1. Terminal output showing at least one `CANDIDATE EVENT` log line
2. Contents of one `evidence/*.json` file with a non-null `clip_path`
3. Matching `evidence/*.mp4` clip written (file size > 0)

Do not advance to Phase 2 until real output has been confirmed.
