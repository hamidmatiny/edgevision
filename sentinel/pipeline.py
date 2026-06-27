"""
pipeline.py — Main Run Loop

Wires together all stages:
  Stage 1a: StreamReader      (ingestion)
  Stage 1b: Detector + Tracker (YOLO + ByteTrack)
  Stage 2:  ZoneEngine         (rules: zone, schedule, dwell)
  [Stage 3: VLM Verifier]      ← hook point; not yet built (Phase 3)
  Stage 4:  EventBuilder + ClipWriter (evidence)

Architecture note:
  Each camera runs its own pipeline instance. For multi-camera deployments,
  launch one process/thread per camera. This module is single-camera.

  Stage 3 is stubbed as a pass-through (always confirms candidates) so the
  architecture slot exists for Phase 3 without changing anything downstream.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import yaml

from sentinel.detection.detector import Detector
from sentinel.detection.tracker import Tracker
from sentinel.events.clip_writer import ClipWriter
from sentinel.events.event_builder import build_event, save_event, update_clip_path
from sentinel.ingestion.stream_reader import StreamReader
from sentinel.rules.zone_engine import ZoneEngine, parse_zone_config

logger = logging.getLogger(__name__)

# How often to purge stale track IDs from zone engine memory (every N processed frames)
PURGE_INTERVAL_FRAMES = 300


# ---------------------------------------------------------------------------
# Stage 3 stub — swap out in Phase 3
# ---------------------------------------------------------------------------

class VLMVerifierStub:
    """
    Placeholder for the Stage 3 VLM contextual verifier (Phase 3).

    Currently a pass-through: every candidate is confirmed.
    Replace this class with a real VLM wrapper in Phase 3 without
    touching the pipeline.py call site.
    """

    def verify(self, candidate, frame_bgr) -> tuple[bool, Optional[str]]:
        """
        Returns (confirmed: bool, reasoning: str | None).
        Pass-through: always confirms.
        """
        return True, None   # (confirmed, vlm_reasoning_text)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_zones_for_camera(zones_yaml_path: str, camera_id: str) -> list:
    with open(zones_yaml_path, "r") as f:
        raw = yaml.safe_load(f)
    camera_zones = raw.get(camera_id, [])
    if not camera_zones:
        logger.warning("No zones configured for camera '%s'.", camera_id)
    return [parse_zone_config(z) for z in camera_zones]


def load_camera_config(cameras_yaml_path: str, camera_id: str) -> dict:
    with open(cameras_yaml_path, "r") as f:
        raw = yaml.safe_load(f)
    if camera_id not in raw.get("cameras", {}):
        raise ValueError(f"Camera '{camera_id}' not found in {cameras_yaml_path}")
    return raw["cameras"][camera_id]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class CameraPipeline:
    """
    Single-camera end-to-end pipeline.

    Usage:
        pipeline = CameraPipeline(
            camera_id="cam1",
            source="rtsp://...",
            zones_yaml="config/zones.yaml",
            evidence_dir="evidence/",
        )
        pipeline.run()
    """

    def __init__(
        self,
        camera_id: str,
        source: str,
        zones_yaml: str,
        evidence_dir: str = "evidence",
        frame_skip: int = 3,
        model_name: str = "yolo11n.pt",
        confidence_threshold: float = 0.4,
        device: str = "cpu",
        pre_trigger_seconds: float = 3.0,
        post_trigger_seconds: float = 3.0,
    ):
        self.camera_id = camera_id
        self.evidence_dir = Path(evidence_dir)

        # Stage 1
        self.reader = StreamReader(source=source, frame_skip=frame_skip, camera_id=camera_id)
        self.detector = Detector(
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            device=device,
        )
        self.tracker = Tracker()

        # Stage 2
        self.zones = load_zones_for_camera(zones_yaml, camera_id)
        self.zone_engine = ZoneEngine(camera_id=camera_id)

        # Stage 3 stub
        self.vlm_verifier = VLMVerifierStub()

        # Evidence
        self.clip_writer = ClipWriter(
            camera_id=camera_id,
            output_dir=evidence_dir,
            pre_trigger_seconds=pre_trigger_seconds,
            post_trigger_seconds=post_trigger_seconds,
        )

        self._frame_count = 0

    def run(self, max_frames: Optional[int] = None) -> None:
        """
        Main loop. Blocks until source ends (local file) or is interrupted (RTSP).
        Press Ctrl+C to stop.

        Args:
            max_frames: Stop after this many processed frames (useful for testing).
        """
        logger.info(
            "[%s] Pipeline starting. Zones loaded: %d. Evidence dir: %s",
            self.camera_id, len(self.zones), self.evidence_dir,
        )

        if not self.zones:
            logger.warning(
                "[%s] No zones configured — pipeline will run detection/tracking "
                "but no events will be triggered.", self.camera_id,
            )

        try:
            for frame_idx, wall_time, frame_bgr in self.reader:
                self._process_frame(frame_idx, wall_time, frame_bgr)
                self._frame_count += 1

                if max_frames is not None and self._frame_count >= max_frames:
                    logger.info("[%s] Reached max_frames=%d, stopping.", self.camera_id, max_frames)
                    break

        except KeyboardInterrupt:
            logger.info("[%s] Pipeline interrupted by user (Ctrl+C).", self.camera_id)
        finally:
            leftover_clips = self.clip_writer.flush_all()
            if leftover_clips:
                logger.info("[%s] Flushed %d pending clip(s).", self.camera_id, len(leftover_clips))
            self.reader.release()
            logger.info("[%s] Pipeline stopped. Processed %d frames.", self.camera_id, self._frame_count)

    def _process_frame(self, frame_idx: int, wall_time: float, frame_bgr) -> None:
        mono_time = time.monotonic()
        h, w = frame_bgr.shape[:2]

        # --- Stage 1a: Detect ---
        raw_detections = self.detector.detect(frame_bgr)

        # --- Stage 1b: Track ---
        tracked_detections = self.tracker.update(raw_detections, frame_shape=(h, w))

        # Feed frame into clip pre-buffer (regardless of detections)
        completed_clips = self.clip_writer.push(frame_bgr)
        self._handle_completed_clips(completed_clips)

        if not tracked_detections:
            return

        # Periodic purge of stale track states
        if self._frame_count % PURGE_INTERVAL_FRAMES == 0:
            active_ids = {d["track_id"] for d in tracked_detections if d.get("track_id", -1) != -1}
            self.zone_engine.purge_stale_tracks(active_ids)

        # --- Stage 2: Zone/rule engine ---
        candidates = self.zone_engine.evaluate(
            frame_time=mono_time,
            detections=tracked_detections,
            zones=self.zones,
        )

        if not candidates:
            return

        for candidate in candidates:
            # --- Stage 3: VLM verification (stub — always confirms) ---
            confirmed, vlm_reasoning = self.vlm_verifier.verify(candidate, frame_bgr)

            if not confirmed:
                logger.info(
                    "[%s] VLM rejected candidate: track=%d zone='%s' reason=%s",
                    self.camera_id, candidate.track_id, candidate.zone_name, vlm_reasoning,
                )
                continue

            # --- Stage 4: Build and save event record ---
            event_record = build_event(candidate, wall_time=wall_time)
            event_path = save_event(event_record, output_dir=self.evidence_dir)

            # Trigger clip capture
            event_id = event_record["event_id"]
            self.clip_writer.trigger(event_id=event_id)

            logger.info(
                "[%s] INCIDENT CONFIRMED — event_id=%s zone='%s' track=%d "
                "class=%s conf=%.2f | record: %s",
                self.camera_id, event_id,
                candidate.zone_name, candidate.track_id,
                candidate.detection_class, candidate.confidence,
                event_path,
            )

    def _handle_completed_clips(self, clip_paths: list[str]) -> None:
        """Log completed clips (future: update event record with clip_path)."""
        for clip_path in clip_paths:
            logger.info("[%s] Clip ready: %s", self.camera_id, clip_path)
