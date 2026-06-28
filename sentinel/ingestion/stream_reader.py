"""
stream_reader.py — Stage 0: Video Ingestion

Handles both RTSP streams and local video files uniformly via OpenCV.
Key features:
  - Transparent abstraction: callers don't care if source is RTSP or file.
  - Frame-skip: only yields every Nth frame (configurable) to control processing FPS.
  - Retry with exponential backoff for RTSP disconnections (does NOT crash).
  - Yields (frame_index, wall_time, stream_time, frame_bgr) tuples.
    stream_time is seconds on the source timeline (raw_frame_index / fps),
    independent of how fast the processing loop runs.
"""

from __future__ import annotations

import logging
import time
from typing import Generator, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Maximum backoff between RTSP reconnect attempts (seconds)
MAX_RETRY_BACKOFF = 30.0
BASE_RETRY_BACKOFF = 1.0


class StreamReader:
    """
    Iterable video source that yields (frame_index, wall_time, stream_time, frame_bgr).

    Usage:
        reader = StreamReader(source="rtsp://...", frame_skip=3)
        for frame_idx, wall_time, stream_time, frame in reader:
            process(frame)
    """

    def __init__(
        self,
        source: str,
        frame_skip: int = 1,
        camera_id: str = "unknown",
    ):
        """
        Args:
            source: RTSP URL (e.g. "rtsp://...") or local file path ("/path/to/video.mp4").
            frame_skip: Yield every Nth frame. 1 = every frame, 3 = every 3rd frame.
            camera_id: For logging only.
        """
        if frame_skip < 1:
            raise ValueError("frame_skip must be >= 1")
        self.source = source
        self.frame_skip = frame_skip
        self.camera_id = camera_id
        self._is_rtsp = source.lower().startswith("rtsp://")
        self._cap: Optional[cv2.VideoCapture] = None
        self._source_fps: float = 30.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __iter__(self) -> Generator[tuple[int, float, float, np.ndarray], None, None]:
        """Yield (frame_index, wall_time, stream_time, bgr_frame) tuples."""
        yield from self._read_loop()

    def release(self) -> None:
        """Release the underlying VideoCapture. Safe to call multiple times."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _open(self) -> cv2.VideoCapture:
        """Open and return a VideoCapture, setting RTSP-appropriate buffer options."""
        logger.info("[%s] Opening source: %s", self.camera_id, self.source)
        cap = cv2.VideoCapture(self.source)

        if self._is_rtsp:
            # Reduce RTSP buffer to minimise latency (use latest frames, not buffered ones)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            raise IOError(f"[{self.camera_id}] Cannot open source: {self.source}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        self._source_fps = fps if fps and fps > 0 else 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(
            "[%s] Stream opened: %dx%d @ %.1f fps (processing every %d frame(s))",
            self.camera_id, width, height, self._source_fps, self.frame_skip,
        )
        return cap

    def _stream_time_seconds(self, raw_frame_idx: int) -> float:
        """Map a 1-based raw frame index to seconds on the source timeline."""
        return (raw_frame_idx - 1) / self._source_fps

    def _read_loop(self) -> Generator[tuple[int, float, float, np.ndarray], None, None]:
        raw_frame_idx = 0   # counts every raw frame read from source
        processed_count = 0 # counts frames actually yielded
        retry_backoff = BASE_RETRY_BACKOFF

        while True:
            # --- Ensure capture is open ---
            if self._cap is None or not self._cap.isOpened():
                try:
                    self._cap = self._open()
                    retry_backoff = BASE_RETRY_BACKOFF  # reset on successful open
                except IOError as exc:
                    if self._is_rtsp:
                        logger.warning(
                            "[%s] Source unavailable (%s). Retrying in %.1fs ...",
                            self.camera_id, exc, retry_backoff,
                        )
                        time.sleep(retry_backoff)
                        retry_backoff = min(retry_backoff * 2, MAX_RETRY_BACKOFF)
                        continue
                    else:
                        # Local files that don't open are a hard error
                        raise

            # --- Read one raw frame ---
            ret, frame = self._cap.read()

            if not ret:
                if self._is_rtsp:
                    logger.warning(
                        "[%s] Lost stream connection. Retrying in %.1fs ...",
                        self.camera_id, retry_backoff,
                    )
                    self.release()
                    time.sleep(retry_backoff)
                    retry_backoff = min(retry_backoff * 2, MAX_RETRY_BACKOFF)
                    continue
                else:
                    # End of local file
                    logger.info(
                        "[%s] End of file reached after %d raw frames (%d processed).",
                        self.camera_id, raw_frame_idx, processed_count,
                    )
                    break

            wall_time = time.time()
            raw_frame_idx += 1
            stream_time = self._stream_time_seconds(raw_frame_idx)

            # --- Frame skip ---
            if (raw_frame_idx % self.frame_skip) != 0:
                continue

            yield processed_count, wall_time, stream_time, frame
            processed_count += 1

        self.release()
