"""
detector.py — Stage 1a: YOLO Object Detector

Wraps Ultralytics YOLO and returns per-frame detections in a normalised format.
Only person and vehicle classes are returned by default (configurable).

Output per detection:
  {
    "bbox": [x1, y1, x2, y2],         # pixel coordinates
    "centroid": (cx, cy),              # float pixel coords of bbox center
    "class": "person" | "vehicle",    # normalised class name
    "confidence": float,               # 0..1
    "raw_class_id": int,               # original YOLO class index
  }

Note: track IDs are NOT assigned here — that's tracker.py's job.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# COCO class names that we treat as "person"
_PERSON_CLASSES = {"person"}

# COCO class names that we treat as "vehicle"
_VEHICLE_CLASSES = {
    "car", "truck", "bus", "motorcycle", "bicycle",
    "van", "vehicle",
}


def _normalise_class(class_name: str) -> Optional[str]:
    """Map a raw YOLO class name to our 2-class schema, or None to filter out."""
    name = class_name.lower()
    if name in _PERSON_CLASSES:
        return "person"
    if name in _VEHICLE_CLASSES:
        return "vehicle"
    return None


class Detector:
    """
    YOLO-based object detector.

    Lazy-loads the model on first call to avoid import-time GPU initialisation.
    Swappable by subclassing and overriding `detect()`.
    """

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence_threshold: float = 0.4,
        device: str = "cpu",
        target_classes: Optional[list[str]] = None,
    ):
        """
        Args:
            model_name: Ultralytics model identifier or path to weights file.
            confidence_threshold: Minimum confidence to include a detection.
            device: 'cpu', 'cuda', 'cuda:0', 'mps', etc.
            target_classes: Which normalised classes to keep ("person", "vehicle").
                            Defaults to both.
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.target_classes = set(target_classes) if target_classes else {"person", "vehicle"}
        self._model = None

    def _load_model(self):
        """Lazy-load the YOLO model (only on first call)."""
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise ImportError(
                    "ultralytics is required. Install with: pip install ultralytics"
                ) from exc

            logger.info("Loading YOLO model: %s on device=%s", self.model_name, self.device)
            self._model = YOLO(self.model_name)
            logger.info("YOLO model loaded.")
        return self._model

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        """
        Run inference on one BGR frame and return filtered detections.

        Returns:
            List of dicts:
              { bbox, centroid, class, confidence, raw_class_id }
        """
        model = self._load_model()

        results = model.predict(
            source=frame_bgr,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )

        detections: list[dict] = []

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes
            for i in range(len(boxes)):
                raw_class_id = int(boxes.cls[i].item())
                class_name = model.names[raw_class_id]
                normalised = _normalise_class(class_name)

                if normalised is None or normalised not in self.target_classes:
                    continue

                conf = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].tolist()   # [x1, y1, x2, y2]
                x1, y1, x2, y2 = xyxy
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "centroid": (cx, cy),
                    "class": normalised,
                    "confidence": conf,
                    "raw_class_id": raw_class_id,
                })

        return detections
