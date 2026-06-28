"""Ensure Phase 2 training pipeline excludes Ultralytics sample images."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_prepare_training_data_excludes_bus_jpg():
    script = (REPO_ROOT / "scripts" / "prepare_training_data.py").read_text(encoding="utf-8")
    assert "ultralytics.com/images" not in script.lower()
    assert "SOURCE_IMAGE_URL" not in script
    assert "download_image_bgr" not in script


def test_intel_sources_only():
    from sentinel.training.intel_sources import INTEL_TRAINING_VIDEOS

    assert len(INTEL_TRAINING_VIDEOS) >= 4
    for video in INTEL_TRAINING_VIDEOS:
        assert video.filename.endswith(".mp4")
