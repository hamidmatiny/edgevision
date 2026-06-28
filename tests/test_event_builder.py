"""
Unit tests for event_builder.py

Tests cover:
  - build_event produces a well-formed IncidentRecord
  - All required schema fields are present
  - save_event writes a valid JSON file to disk
  - update_clip_path returns an updated record without mutating the original
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from sentinel.events.event_builder import (
    SCHEMA_VERSION,
    build_event,
    rewrite_event,
    save_event,
    update_clip_path,
)
from sentinel.rules.zone_engine import CandidateEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_candidate(
    camera_id="cam1",
    zone_name="test_zone",
    track_id=42,
    detection_class="person",
    confidence=0.87,
    centroid=(123.4, 456.7),
    trigger_time=1000.0,
    dwell_elapsed=4.5,
) -> CandidateEvent:
    return CandidateEvent(
        camera_id=camera_id,
        zone_name=zone_name,
        track_id=track_id,
        detection_class=detection_class,
        confidence=confidence,
        centroid=centroid,
        trigger_time=trigger_time,
        dwell_elapsed=dwell_elapsed,
    )


# ---------------------------------------------------------------------------
# build_event
# ---------------------------------------------------------------------------

class TestBuildEvent:
    def test_required_fields_present(self):
        candidate = make_candidate()
        record = build_event(candidate, wall_time=1719000000.0)

        required_keys = [
            "schema_version", "event_id", "camera_id", "zone_name",
            "track_id", "detection_class", "confidence", "centroid",
            "trigger_wall_time", "trigger_wall_time_iso",
            "dwell_elapsed_seconds", "clip_path",
            "vlm_verification", "audit_chain",
        ]
        for key in required_keys:
            assert key in record, f"Missing required key: {key}"

    def test_schema_version(self):
        record = build_event(make_candidate(), wall_time=1719000000.0)
        assert record["schema_version"] == SCHEMA_VERSION

    def test_event_id_is_uuid(self):
        import uuid
        record = build_event(make_candidate(), wall_time=1719000000.0)
        # Should not raise
        parsed = uuid.UUID(record["event_id"])
        assert str(parsed) == record["event_id"]

    def test_values_match_candidate(self):
        candidate = make_candidate()
        record = build_event(candidate, wall_time=1719000000.0)
        assert record["camera_id"] == "cam1"
        assert record["zone_name"] == "test_zone"
        assert record["track_id"] == 42
        assert record["detection_class"] == "person"
        assert record["confidence"] == pytest.approx(0.87, abs=1e-4)
        assert record["dwell_elapsed_seconds"] == pytest.approx(4.5, abs=1e-3)

    def test_centroid_is_list_of_two_floats(self):
        record = build_event(make_candidate(), wall_time=1719000000.0)
        assert isinstance(record["centroid"], list)
        assert len(record["centroid"]) == 2

    def test_clip_path_none_by_default(self):
        record = build_event(make_candidate(), wall_time=1719000000.0)
        assert record["clip_path"] is None

    def test_clip_path_set_when_provided(self):
        record = build_event(make_candidate(), wall_time=1719000000.0, clip_path="/evidence/test.mp4")
        assert record["clip_path"] == "/evidence/test.mp4"

    def test_vlm_verification_placeholder_is_none(self):
        record = build_event(make_candidate(), wall_time=1719000000.0)
        assert record["vlm_verification"] is None

    def test_audit_chain_structure(self):
        record = build_event(make_candidate(), wall_time=1719000000.0)
        chain = record["audit_chain"]
        assert "stage1_confidence" in chain
        assert "stage2_zone_rule" in chain
        assert "stage3_vlm" in chain
        assert chain["stage3_vlm"] is None
        assert chain["final_decision"] == "candidate"

    def test_iso_timestamp_format(self):
        from datetime import datetime, timezone
        record = build_event(make_candidate(), wall_time=1719000000.0)
        # Should parse without error
        dt = datetime.fromisoformat(record["trigger_wall_time_iso"])
        assert dt.tzinfo is not None  # must be timezone-aware

    def test_two_events_have_different_ids(self):
        candidate = make_candidate()
        r1 = build_event(candidate, wall_time=1.0)
        r2 = build_event(candidate, wall_time=1.0)
        assert r1["event_id"] != r2["event_id"]


# ---------------------------------------------------------------------------
# save_event
# ---------------------------------------------------------------------------

class TestSaveEvent:
    def test_file_is_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            record = build_event(make_candidate(), wall_time=1719000000.0)
            path = save_event(record, output_dir=tmpdir)
            assert os.path.isfile(path)

    def test_saved_file_is_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            record = build_event(make_candidate(), wall_time=1719000000.0)
            path = save_event(record, output_dir=tmpdir)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["event_id"] == record["event_id"]

    def test_filename_contains_camera_and_event_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            record = build_event(make_candidate(camera_id="cam_test"), wall_time=1719000000.0)
            path = save_event(record, output_dir=tmpdir)
            filename = os.path.basename(path)
            assert "cam_test" in filename
            assert record["event_id"] in filename

    def test_output_dir_created_if_not_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = os.path.join(tmpdir, "new", "nested", "dir")
            record = build_event(make_candidate(), wall_time=1719000000.0)
            path = save_event(record, output_dir=new_dir)
            assert os.path.isfile(path)


# ---------------------------------------------------------------------------
# update_clip_path
# ---------------------------------------------------------------------------

class TestUpdateClipPath:
    def test_clip_path_is_updated(self):
        record = build_event(make_candidate(), wall_time=1719000000.0)
        updated = update_clip_path(record, "/evidence/clip.mp4")
        assert updated["clip_path"] == "/evidence/clip.mp4"

    def test_original_record_not_mutated(self):
        record = build_event(make_candidate(), wall_time=1719000000.0)
        _ = update_clip_path(record, "/evidence/clip.mp4")
        assert record["clip_path"] is None   # original unchanged

    def test_other_fields_preserved(self):
        record = build_event(make_candidate(), wall_time=1719000000.0)
        updated = update_clip_path(record, "/evidence/clip.mp4")
        assert updated["event_id"] == record["event_id"]
        assert updated["camera_id"] == record["camera_id"]


class TestRewriteEvent:
    def test_rewrite_event_updates_file_on_disk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            record = build_event(make_candidate(), wall_time=1719000000.0)
            path = save_event(record, output_dir=tmpdir)
            updated = update_clip_path(record, f"{tmpdir}/clip.mp4")
            rewrite_event(updated, path)

            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)
            assert loaded["clip_path"] == f"{tmpdir}/clip.mp4"
            assert loaded["event_id"] == record["event_id"]
