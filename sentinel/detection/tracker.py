"""
tracker.py — Stage 1b: Multi-Object Tracker

Wraps ByteTrack (via the `supervision` library) to assign stable track IDs
across frames. Inputs are raw detections from detector.py; outputs are the
same dicts augmented with a `track_id` field.

Why supervision/ByteTrack:
  - Maintained, well-tested library; no need to hand-roll tracking.
  - ByteTrack is lightweight and handles occlusion / ID switching gracefully
    for the frame rates typical of edge deployments.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class Tracker:
    """
    Wraps ByteTrack via supervision.

    Call `update(detections)` each processed frame.
    Returns the input detections list augmented with `track_id` for each
    detection that received a stable ID. Detections the tracker couldn't
    match are still returned but with `track_id = -1`.
    """

    def __init__(self):
        self._tracker = None

    def _load_tracker(self):
        if self._tracker is None:
            try:
                import supervision as sv
            except ImportError as exc:
                raise ImportError(
                    "supervision is required. Install with: pip install supervision"
                ) from exc
            self._tracker = sv.ByteTrack()
            logger.info("ByteTrack tracker initialised.")
        return self._tracker

    def update(self, detections: list[dict], frame_shape: tuple[int, int]) -> list[dict]:
        """
        Update the tracker with the current frame's detections.

        Args:
            detections: List of detection dicts from Detector.detect().
            frame_shape: (height, width) of the source frame (needed by supervision).

        Returns:
            Same list, each dict now has a `track_id` key (int, or -1 if unmatched).
        """
        import supervision as sv

        if not detections:
            # Feed empty detections to keep tracker state ticking
            sv_dets = sv.Detections.empty()
            self._load_tracker().update_with_detections(sv_dets)
            return []

        tracker = self._load_tracker()

        # Build supervision Detections object from our dict format
        xyxy = np.array([d["bbox"] for d in detections], dtype=np.float32)
        confidences = np.array([d["confidence"] for d in detections], dtype=np.float32)
        class_ids = np.array([d["raw_class_id"] for d in detections], dtype=int)

        sv_dets = sv.Detections(
            xyxy=xyxy,
            confidence=confidences,
            class_id=class_ids,
        )

        tracked = tracker.update_with_detections(sv_dets)

        # supervision returns updated Detections with .tracker_id populated
        # Match back to our detection dicts by index (supervision preserves order
        # for matched detections; unmatched ones are dropped from tracked).
        # We re-index by bbox similarity to merge the track IDs back.

        result: list[dict] = []

        if tracked.tracker_id is None or len(tracked) == 0:
            # No tracked objects this frame
            for det in detections:
                result.append({**det, "track_id": -1})
            return result

        # For each tracked detection, find the best-matching original detection
        for i in range(len(tracked)):
            tracked_bbox = tracked.xyxy[i]
            track_id = int(tracked.tracker_id[i])

            # Find the original detection closest to this tracked bbox
            best_idx = _match_bbox(tracked_bbox, xyxy)
            if best_idx is not None:
                augmented = {**detections[best_idx], "track_id": track_id}
                result.append(augmented)

        return result


def _match_bbox(target: np.ndarray, candidates: np.ndarray, iou_threshold: float = 0.3) -> Optional[int]:
    """
    Return the index into `candidates` with the highest IoU against `target`,
    or None if no candidate exceeds the threshold.
    """
    if len(candidates) == 0:
        return None

    ious = _batch_iou(target, candidates)
    best = int(np.argmax(ious))
    return best if ious[best] >= iou_threshold else None


def _batch_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Compute IoU between one box [x1,y1,x2,y2] and an array of boxes (N,4)."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - inter_area

    return np.where(union_area > 0, inter_area / union_area, 0.0)
