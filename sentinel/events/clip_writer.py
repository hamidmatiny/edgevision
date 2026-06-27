"""
clip_writer.py — Evidence Clip Capture

Maintains a rolling frame buffer per camera and, on trigger, writes a
short video clip (pre-trigger + post-trigger frames) to the evidence/ folder.

Design:
  - Rolling deque of the last N seconds of raw frames (pre-buffer).
  - On trigger, starts accumulating post-trigger frames.
  - Writes the combined clip to disk as an MP4 via OpenCV VideoWriter.
  - Non-blocking trigger: post-trigger frames are accumulated by subsequent
    calls to push(); the clip is finalised automatically once post-buffer fills.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Codec for output clips (MP4 / H.264 where available)
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")


@dataclass
class _PendingClip:
    """State for a clip that has been triggered and is accumulating post-trigger frames."""
    trigger_event_id: str
    pre_frames: list[np.ndarray]
    post_frames: list[np.ndarray] = field(default_factory=list)
    target_post_count: int = 0


class ClipWriter:
    """
    Rolling-buffer clip writer for one camera stream.

    Usage:
        writer = ClipWriter(camera_id="cam1", output_dir="evidence/", fps=10.0)

        # In the frame loop:
        writer.push(frame)

        # On a candidate event:
        clip_path = writer.trigger(event_id="evt_001")

        # Must keep calling push() with new frames; clip finalises automatically.
    """

    def __init__(
        self,
        camera_id: str,
        output_dir: str | Path = "evidence",
        fps: float = 10.0,
        pre_trigger_seconds: float = 3.0,
        post_trigger_seconds: float = 3.0,
    ):
        self.camera_id = camera_id
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.pre_trigger_seconds = pre_trigger_seconds
        self.post_trigger_seconds = post_trigger_seconds

        pre_buffer_size = max(1, int(pre_trigger_seconds * fps))
        self._pre_buffer: deque[np.ndarray] = deque(maxlen=pre_buffer_size)
        self._post_count = max(1, int(post_trigger_seconds * fps))
        self._pending: dict[str, _PendingClip] = {}  # event_id → pending clip

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def push(self, frame: np.ndarray) -> list[str]:
        """
        Feed the next frame into the rolling buffer (and any pending clips).

        Returns a list of clip file paths that were COMPLETED this call
        (usually empty, occasionally has one entry when a post-trigger buffer fills).
        """
        self._pre_buffer.append(frame.copy())

        completed_paths: list[str] = []

        for event_id, pending in list(self._pending.items()):
            pending.post_frames.append(frame.copy())
            if len(pending.post_frames) >= pending.target_post_count:
                path = self._write_clip(pending)
                completed_paths.append(path)
                del self._pending[event_id]
                logger.info(
                    "[%s] Evidence clip written: %s (%d frames)",
                    self.camera_id, path,
                    len(pending.pre_frames) + len(pending.post_frames),
                )

        return completed_paths

    def trigger(self, event_id: str) -> None:
        """
        Mark a candidate event — begin accumulating post-trigger frames.
        The clip is written automatically once enough post frames accumulate.

        Call push() with subsequent frames to fill the post-trigger buffer.
        """
        if event_id in self._pending:
            logger.debug("[%s] Duplicate trigger for event %s — ignoring.", self.camera_id, event_id)
            return

        pre_frames = list(self._pre_buffer)  # snapshot of current pre-buffer
        self._pending[event_id] = _PendingClip(
            trigger_event_id=event_id,
            pre_frames=pre_frames,
            target_post_count=self._post_count,
        )
        logger.debug(
            "[%s] Clip triggered for event %s (%d pre-frames buffered).",
            self.camera_id, event_id, len(pre_frames),
        )

    def flush_all(self) -> list[str]:
        """
        Force-write all pending clips immediately (even if post-buffer not full).
        Call at pipeline shutdown to avoid losing evidence.
        """
        paths = []
        for event_id, pending in list(self._pending.items()):
            if pending.pre_frames or pending.post_frames:
                path = self._write_clip(pending)
                paths.append(path)
                logger.info("[%s] Flushed incomplete clip: %s", self.camera_id, path)
            del self._pending[event_id]
        return paths

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_clip(self, pending: _PendingClip) -> str:
        all_frames = pending.pre_frames + pending.post_frames
        if not all_frames:
            logger.warning("[%s] No frames to write for event %s", self.camera_id, pending.trigger_event_id)
            return ""

        h, w = all_frames[0].shape[:2]
        ts = int(time.time())
        filename = f"{self.camera_id}_{pending.trigger_event_id}_{ts}.mp4"
        out_path = self.output_dir / filename

        writer = cv2.VideoWriter(str(out_path), FOURCC, self.fps, (w, h))
        for frame in all_frames:
            writer.write(frame)
        writer.release()

        return str(out_path)
