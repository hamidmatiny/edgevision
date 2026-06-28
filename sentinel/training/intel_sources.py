"""
Intel IoT DevKit sample-videos — CC-BY 4.0 training sources for Phase 2.

Repository: https://github.com/intel-iot-devkit/sample-videos
License:    Creative Commons Attribution 4.0 International (CC-BY 4.0)

These videos are the ONLY approved real-footage source for Step 2 training.
ExDark is evaluation-only. Ultralytics sample images (bus.jpg) are NOT used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

INTEL_SAMPLE_VIDEOS_BASE = (
    "https://github.com/intel-iot-devkit/sample-videos/raw/master"
)

INTEL_VIDEO_LICENSE = "CC-BY-4.0"
INTEL_ATTRIBUTION = (
    "Sample video footage from Intel IoT DevKit sample-videos "
    "(https://github.com/intel-iot-devkit/sample-videos), "
    "licensed under CC-BY 4.0."
)


@dataclass(frozen=True)
class IntelTrainingVideo:
    filename: str
    description: str
    local_aliases: tuple[str, ...] = ()


INTEL_TRAINING_VIDEOS: tuple[IntelTrainingVideo, ...] = (
    IntelTrainingVideo(
        filename="person-bicycle-car-detection.mp4",
        description="Outdoor pedestrian, bicycle, and car scene (768×432 @ 12 fps).",
        local_aliases=("test_video.mp4",),
    ),
    IntelTrainingVideo(
        filename="people-detection.mp4",
        description="Multiple people in a retail/indoor-style scene.",
    ),
    IntelTrainingVideo(
        filename="one-by-one-person-detection.mp4",
        description="Single person entering frame sequentially.",
    ),
    IntelTrainingVideo(
        filename="face-demographics-walking.mp4",
        description="People walking past camera at varying distances.",
    ),
)


def intel_video_url(filename: str) -> str:
    return f"{INTEL_SAMPLE_VIDEOS_BASE}/{filename}"


def resolve_local_video_path(
    video: IntelTrainingVideo,
    repo_root: Path,
    cache_dir: Path,
) -> Path | None:
    """Return an existing local path for this video, if already present."""
    for alias in video.local_aliases:
        candidate = repo_root / alias
        if candidate.exists():
            return candidate
    cached = cache_dir / video.filename
    if cached.exists():
        return cached
    return None
