#!/usr/bin/env python3
"""
Generate smoke_test_loiter.mp4 — a deterministic local test clip for Phase 1 e2e.

The video is 768×432 @ 12 fps, 15 seconds. A real person photograph (Ultralytics
`bus.jpg`, verified detectable by yolo11n) is composited into the cam1
`restricted_storage` zone ([500,100]–[700,400]) and held stationary so dwell
time exceeds the default 5s threshold on the stream timeline.

Usage (from repo root):
    python scripts/generate_smoke_test_video.py
    python scripts/generate_smoke_test_video.py --output tests/fixtures/smoke_test_loiter.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import requests

# Match config/zones.yaml cam1 restricted_storage and typical dev resolution.
WIDTH = 768
HEIGHT = 432
FPS = 12
DURATION_SECONDS = 15

DEFAULT_OUTPUT = Path("tests/fixtures/smoke_test_loiter.mp4")
# Verified: yolo11n detects persons in this image when scaled into restricted_storage.
SOURCE_IMAGE_URL = "https://ultralytics.com/images/bus.jpg"


def download_image_bgr(url: str) -> np.ndarray:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    arr = np.frombuffer(resp.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to decode image from {url}")
    return img


def composite_frame(source_bgr: np.ndarray) -> np.ndarray:
    frame = np.full((HEIGHT, WIDTH, 3), 48, dtype=np.uint8)
    cv2.rectangle(frame, (500, 100), (700, 400), (70, 70, 70), 2)
    cv2.putText(
        frame, "restricted_storage", (510, 130),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA,
    )

    # Scale so person height ~360px; anchor bottom-right of crop inside the zone.
    src_h, src_w = source_bgr.shape[:2]
    scale = 360 / src_h
    new_w, new_h = int(src_w * scale), int(src_h * scale)
    resized = cv2.resize(source_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    x1 = 500
    y1 = max(20, 400 - new_h + 20)
    x2 = min(WIDTH, x1 + new_w)
    y2 = min(HEIGHT, y1 + new_h)
    crop_w, crop_h = x2 - x1, y2 - y1
    frame[y1:y2, x1:x2] = resized[:crop_h, :crop_w]
    return frame


def generate(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source = download_image_bgr(SOURCE_IMAGE_URL)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {output_path}")

    frame = composite_frame(source)
    total_frames = FPS * DURATION_SECONDS
    for _ in range(total_frames):
        writer.write(frame)
    writer.release()

    print(f"Wrote {total_frames} frames ({DURATION_SECONDS}s @ {FPS} fps) -> {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate(args.output)


if __name__ == "__main__":
    main()
