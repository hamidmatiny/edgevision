# SENTINEL — Phase 1 Build Notes

## What was built

Phase 1 delivers the core local perception pipeline:

**Stage 1a — Ingestion (`sentinel/ingestion/stream_reader.py`)**
Handles both RTSP streams and local video files uniformly. Configurable frame-skip to control processing FPS. RTSP reconnect with exponential backoff — never crashes on connectivity loss.

**Stage 1b — Detection + Tracking (`sentinel/detection/`)**
- `detector.py`: Wraps Ultralytics YOLO (pretrained, no fine-tuning yet). Filters to person + vehicle classes only. Returns normalised dicts with bounding box, centroid, class, confidence.
- `tracker.py`: Wraps ByteTrack via the `supervision` library. Assigns stable track IDs across frames. Handles occlusion and brief ID loss gracefully.

**Stage 2 — Zone/Rule Engine (`sentinel/rules/zone_engine.py`)**
The core IP of the product. Evaluates each tracked detection against per-camera zone configs:
- Polygon containment check (Shapely)
- Schedule check ("always" or HH:MM-HH:MM with midnight-wrap support)
- Dwell-time state machine: a detection must be continuously inside a zone for ≥ N seconds before becoming a candidate event. Re-entry resets the timer.

**Stage 3 stub (`sentinel/pipeline.py` → `VLMVerifierStub`)**
Pass-through placeholder that always confirms candidates. Swap in the real VLM verifier in Phase 3 without touching any other code.

**Stage 4 — Evidence (`sentinel/events/`)**
- `event_builder.py`: Builds a structured IncidentRecord JSON (schema v1.0) with full audit chain.
- `clip_writer.py`: Maintains a rolling pre-trigger frame buffer. On trigger, accumulates post-trigger frames and writes an MP4 evidence clip.

## How to run

### 1. Install dependencies

```bash
pip install -e ".[dev]"
```

### 2. Get a test video

**Option A — download a public sample:**
```bash
# Any MP4 with people walking works. Example (replace with any video URL):
wget -O test_video.mp4 "https://www.pexels.com/video/[any-free-stock-video]"
```

**Option B — record your own (30 seconds, walk in front of webcam):**
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

### 3. Edit the zone config

Open `config/zones.yaml`. The `cam1` zone is already set to cover `[[100,200],[400,200],[400,500],[100,500]]`. For a 640×480 video, adjust the polygon to cover the region you walk through, or use the default and walk in the centre of frame.

Set `schedule: "always"` for daytime testing.

### 4. Run the pipeline

```bash
python scripts/run_local.py \
  --source test_video.mp4 \
  --camera-id cam1 \
  --frame-skip 2 \
  --log-level INFO
```

### 5. Expected output

In your terminal (after ~3 seconds of a person in the zone):
```
2024-01-15 14:32:01 [INFO] sentinel.rules.zone_engine — CANDIDATE EVENT: track=1 zone='perimeter_east' camera=cam1 class=person dwell=3.2s conf=0.87 centroid=(250.3, 350.1)
2024-01-15 14:32:01 [INFO] sentinel.events.event_builder — Event record saved: evidence/cam1_<uuid>.json
2024-01-15 14:32:04 [INFO] sentinel.pipeline — Clip ready: evidence/cam1_<event_id>_<ts>.mp4
```

**Files created in `evidence/`:**
- `cam1_<uuid>.json` — structured incident record
- `cam1_<uuid>_<timestamp>.mp4` — short video clip (pre + post trigger)

**Example event JSON:**
```json
{
  "schema_version": "1.0",
  "event_id": "a3f7c2d1-...",
  "camera_id": "cam1",
  "zone_name": "perimeter_east",
  "track_id": 1,
  "detection_class": "person",
  "confidence": 0.8721,
  "centroid": [250.3, 350.1],
  "trigger_wall_time": 1719000123.45,
  "trigger_wall_time_iso": "2024-06-21T20:02:03.450000+00:00",
  "dwell_elapsed_seconds": 3.2,
  "clip_path": null,
  "vlm_verification": null,
  "audit_chain": {
    "stage1_confidence": 0.8721,
    "stage2_zone_rule": "zone='perimeter_east' dwell=3.2s",
    "stage3_vlm": null,
    "final_decision": "candidate"
  }
}
```

### 6. Run unit tests

```bash
pytest tests/ -v
```

All zone engine and event builder tests should pass with no dependencies on GPU or real video.

## Known limitations (Phase 1)

- **Stock YOLO weights only.** No fine-tuning on industrial/night footage yet. Expect higher FP rate in challenging lighting. Phase 2 addresses this with synthetic data fine-tuning.
- **False positive rate not yet measured.** FP benchmark harness is Phase 3 (requires VLM verifier to compare Stage1+2 vs Stage1+2+3).
- **Stage 3 VLM is a stub.** All candidates are auto-confirmed. Phase 3 adds the contextual reasoning layer that is the primary FP reduction mechanism.
- **No cloud dashboard.** Phase 5.
- **Single-camera only.** Multi-camera: run one process per camera, or spawn a thread per camera — the pipeline is per-camera by design.
- **clip_path in event JSON is null** after initial write; the clip is written asynchronously. A future update will patch the JSON file once the clip path is known.

## Verification checkpoint

After running against a real test video, paste the following back to confirm Phase 1 is working:
1. Terminal output showing at least one CANDIDATE EVENT log line
2. Contents of one `evidence/*.json` file
3. Confirmation that an `evidence/*.mp4` clip was written (file size > 0)

Do not advance to Phase 2 until real output has been confirmed.
