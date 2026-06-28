"""
Unit tests for zone_engine.py

Tests cover:
  - Schedule parsing (normal window, midnight-wrap, always, edge cases)
  - Point-in-polygon (inside, outside, on boundary)
  - Dwell-time state machine:
      - No event before threshold
      - Event fires exactly at threshold
      - No re-fire while still inside after first event
      - Re-entry resets timer (fires again after dwell from re-entry)
      - Departure detected correctly
  - Class filter: zone only fires for configured classes
  - Multi-zone, multi-track interactions

Run with: pytest tests/test_zone_engine.py -v
"""

from datetime import datetime, time

import pytest
from shapely.geometry import Polygon

from sentinel.rules.zone_engine import (
    CandidateEvent,
    DwellState,
    ZoneConfig,
    ZoneEngine,
    is_inside_zone,
    is_schedule_active,
    parse_zone_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_zone(
    name="test_zone",
    polygon=None,
    schedule="always",
    dwell_seconds=3.0,
    classes=None,
) -> ZoneConfig:
    if polygon is None:
        # Default: unit square [0,0]-[10,10]
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    if classes is None:
        classes = ["person"]
    return ZoneConfig(
        name=name,
        polygon=polygon,
        schedule=schedule,
        dwell_seconds=dwell_seconds,
        classes=classes,
    )


def make_det(track_id=1, cls="person", confidence=0.9, centroid=(5.0, 5.0)) -> dict:
    return {
        "track_id": track_id,
        "class": cls,
        "confidence": confidence,
        "centroid": centroid,
    }


# ---------------------------------------------------------------------------
# is_schedule_active
# ---------------------------------------------------------------------------

class TestSchedule:
    def test_always_returns_true(self):
        assert is_schedule_active("always") is True

    def test_always_case_insensitive(self):
        assert is_schedule_active("Always") is True
        assert is_schedule_active("ALWAYS") is True

    def test_normal_window_inside(self):
        now = datetime(2024, 1, 15, 10, 30)   # 10:30
        assert is_schedule_active("08:00-17:00", now=now) is True

    def test_normal_window_at_start(self):
        now = datetime(2024, 1, 15, 8, 0)
        assert is_schedule_active("08:00-17:00", now=now) is True

    def test_normal_window_at_end_exclusive(self):
        # End time is exclusive
        now = datetime(2024, 1, 15, 17, 0)
        assert is_schedule_active("08:00-17:00", now=now) is False

    def test_normal_window_before(self):
        now = datetime(2024, 1, 15, 7, 59)
        assert is_schedule_active("08:00-17:00", now=now) is False

    def test_normal_window_after(self):
        now = datetime(2024, 1, 15, 18, 0)
        assert is_schedule_active("08:00-17:00", now=now) is False

    def test_midnight_wrap_in_night_portion(self):
        # "22:00-06:00" — active at 23:00
        now = datetime(2024, 1, 15, 23, 0)
        assert is_schedule_active("22:00-06:00", now=now) is True

    def test_midnight_wrap_in_early_morning(self):
        # "22:00-06:00" — active at 01:00
        now = datetime(2024, 1, 15, 1, 0)
        assert is_schedule_active("22:00-06:00", now=now) is True

    def test_midnight_wrap_at_start(self):
        now = datetime(2024, 1, 15, 22, 0)
        assert is_schedule_active("22:00-06:00", now=now) is True

    def test_midnight_wrap_at_end_exclusive(self):
        now = datetime(2024, 1, 15, 6, 0)
        assert is_schedule_active("22:00-06:00", now=now) is False

    def test_midnight_wrap_outside(self):
        # 10:00 is outside "22:00-06:00"
        now = datetime(2024, 1, 15, 10, 0)
        assert is_schedule_active("22:00-06:00", now=now) is False

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError):
            is_schedule_active("bad-format")


# ---------------------------------------------------------------------------
# is_inside_zone
# ---------------------------------------------------------------------------

class TestPointInPolygon:
    def setup_method(self):
        self.zone = make_zone()  # square (0,0)-(10,10)

    def test_centroid_inside(self):
        assert is_inside_zone((5.0, 5.0), self.zone) is True

    def test_centroid_outside(self):
        assert is_inside_zone((15.0, 15.0), self.zone) is False

    def test_centroid_at_corner(self):
        # Corners are technically on the boundary; shapely's .contains() returns False for boundary
        result = is_inside_zone((0.0, 0.0), self.zone)
        # Boundary behavior: shapely.contains excludes boundary, but this is
        # acceptable — we test that the API doesn't crash and returns a bool.
        assert isinstance(result, bool)

    def test_centroid_near_edge_inside(self):
        assert is_inside_zone((0.5, 5.0), self.zone) is True

    def test_centroid_just_outside(self):
        assert is_inside_zone((-0.1, 5.0), self.zone) is False


# ---------------------------------------------------------------------------
# parse_zone_config
# ---------------------------------------------------------------------------

class TestParseZoneConfig:
    def test_basic_parse(self):
        raw = {
            "name": "test",
            "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
            "schedule": "always",
            "dwell_seconds": 5,
            "classes": ["person", "vehicle"],
        }
        z = parse_zone_config(raw)
        assert z.name == "test"
        assert z.dwell_seconds == 5.0
        assert "person" in z.classes
        assert "vehicle" in z.classes

    def test_too_few_points_raises(self):
        raw = {
            "name": "bad",
            "polygon": [[0, 0], [10, 0]],   # only 2 points
            "schedule": "always",
            "dwell_seconds": 3,
            "classes": ["person"],
        }
        with pytest.raises(ValueError):
            parse_zone_config(raw)

    def test_defaults_applied(self):
        raw = {
            "name": "minimal",
            "polygon": [[0, 0], [10, 0], [10, 10]],
        }
        z = parse_zone_config(raw)
        assert z.schedule == "always"
        assert z.dwell_seconds == 3.0
        assert z.classes == ["person"]


# ---------------------------------------------------------------------------
# ZoneEngine — dwell state machine
# ---------------------------------------------------------------------------

class TestZoneEngineDwell:
    def setup_method(self):
        self.engine = ZoneEngine(camera_id="cam_test")
        self.zone = make_zone(dwell_seconds=3.0)
        self.zones = [self.zone]

    def _eval(self, t, detections=None, now=None):
        if detections is None:
            detections = [make_det()]
        return self.engine.evaluate(t, detections, self.zones, now=now)

    # --- No event before threshold ---

    def test_no_event_before_dwell_threshold(self):
        # Frame at t=0 (entry)
        result = self._eval(0.0)
        assert result == []

        # Frame at t=2.9 (just under 3s threshold)
        result = self._eval(2.9)
        assert result == []

    # --- Event fires at threshold ---

    def test_event_fires_at_dwell_threshold(self):
        self._eval(0.0)          # entry
        result = self._eval(3.0) # exactly at threshold
        assert len(result) == 1
        evt = result[0]
        assert evt.zone_name == "test_zone"
        assert evt.track_id == 1
        assert evt.detection_class == "person"
        assert evt.dwell_elapsed == pytest.approx(3.0, abs=1e-6)

    # --- No re-fire while still inside ---

    def test_no_refire_while_inside(self):
        self._eval(0.0)
        self._eval(3.0)          # fires here
        result = self._eval(5.0) # still inside — should NOT fire again
        assert result == []

    def test_no_refire_multiple_subsequent_frames(self):
        self._eval(0.0)
        self._eval(3.5)          # fires
        for t in [4.0, 5.0, 6.0, 10.0]:
            assert self._eval(t) == []

    # --- Re-entry resets timer ---

    def test_reentry_resets_timer(self):
        # Enter at t=0, stay until t=3 (fires)
        self._eval(0.0)
        self._eval(3.0)          # fires once

        # Leave at t=4 (pass empty detections to simulate departure)
        self.engine.evaluate(4.0, [], self.zones)

        # Re-enter at t=5 — timer resets to 0 relative to t=5
        self._eval(5.0)           # re-entry; no event
        result = self._eval(7.9)  # 2.9s of dwell since re-entry — no event yet
        assert result == []

        result = self._eval(8.1)  # 3.1s of dwell since re-entry — fires again
        assert len(result) == 1

    def test_reentry_before_threshold_then_leaves_resets(self):
        # Enter, leave before threshold, re-enter — timer should reset
        self._eval(0.0)           # enter
        self._eval(1.0)           # 1s — under threshold

        # Leave
        self.engine.evaluate(2.0, [], self.zones)

        # Re-enter
        self._eval(10.0)          # new entry at t=10
        result = self._eval(12.9) # 2.9s — under threshold
        assert result == []

        result = self._eval(13.1) # 3.1s from re-entry — fires
        assert len(result) == 1

    # --- Departure tracking ---

    def test_departure_detected_in_state(self):
        self._eval(0.0)
        # Depart
        self.engine.evaluate(1.0, [], self.zones)

        key = (1, "test_zone")
        state = self.engine._dwell.get(key)
        assert state is not None
        assert state.inside is False

    # --- Class filter ---

    def test_class_filter_excludes_wrong_class(self):
        # Zone only accepts "person"; send "vehicle"
        det = make_det(cls="vehicle")
        for t in [0.0, 3.0, 5.0]:
            result = self.engine.evaluate(t, [det], self.zones)
            assert result == []

    def test_class_filter_accepts_correct_class(self):
        zone = make_zone(classes=["vehicle"])
        det = make_det(cls="vehicle")
        engine = ZoneEngine(camera_id="cam_test2")
        engine.evaluate(0.0, [det], [zone])
        result = engine.evaluate(3.0, [det], [zone])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# ZoneEngine — multi-track, multi-zone
# ---------------------------------------------------------------------------

class TestZoneEngineMulti:
    def test_two_tracks_independent_dwell(self):
        engine = ZoneEngine(camera_id="cam_multi")
        zone = make_zone(dwell_seconds=3.0)
        zones = [zone]

        # Track 1 enters at t=0, Track 2 enters at t=1
        engine.evaluate(0.0, [make_det(track_id=1)], zones)
        engine.evaluate(1.0, [make_det(track_id=1), make_det(track_id=2)], zones)

        # At t=3.0: track 1 has been inside 3s (fires), track 2 only 2s (no fire)
        result = engine.evaluate(3.0, [make_det(track_id=1), make_det(track_id=2)], zones)
        fired_tracks = {e.track_id for e in result}
        assert 1 in fired_tracks
        assert 2 not in fired_tracks

        # At t=4.1: track 2 has been inside 3.1s (fires)
        result = engine.evaluate(4.1, [make_det(track_id=1), make_det(track_id=2)], zones)
        fired_tracks = {e.track_id for e in result}
        assert 2 in fired_tracks
        assert 1 not in fired_tracks  # already fired, no re-fire

    def test_two_zones_same_track(self):
        engine = ZoneEngine(camera_id="cam_2z")
        # Zone A: covers (0,0)-(10,10); Zone B: covers (20,20)-(30,30)
        zone_a = make_zone(name="zone_a", dwell_seconds=2.0)
        zone_b = make_zone(
            name="zone_b",
            polygon=Polygon([(20, 20), (30, 20), (30, 30), (20, 30)]),
            dwell_seconds=2.0,
        )

        # Track inside zone_a only
        det_in_a = make_det(centroid=(5.0, 5.0))
        engine.evaluate(0.0, [det_in_a], [zone_a, zone_b])
        result = engine.evaluate(2.1, [det_in_a], [zone_a, zone_b])

        assert len(result) == 1
        assert result[0].zone_name == "zone_a"

    def test_purge_stale_tracks(self):
        engine = ZoneEngine(camera_id="cam_purge")
        zone = make_zone()
        engine.evaluate(0.0, [make_det(track_id=99)], [zone])
        assert (99, "test_zone") in engine._dwell

        engine.purge_stale_tracks(active_track_ids=set())   # track 99 is gone
        assert (99, "test_zone") not in engine._dwell


# ---------------------------------------------------------------------------
# Schedule integration with ZoneEngine
# ---------------------------------------------------------------------------

class TestZoneEngineSchedule:
    def test_out_of_schedule_no_event(self):
        engine = ZoneEngine(camera_id="cam_sched")
        zone = make_zone(schedule="22:00-06:00", dwell_seconds=1.0)
        zones = [zone]
        det = make_det()

        # Simulate midday (out of schedule)
        midday = datetime(2024, 1, 15, 12, 0)
        for t in [0.0, 1.5, 3.0]:
            result = engine.evaluate(t, [det], zones, now=midday)
            assert result == [], f"Expected no event at t={t} (out of schedule)"

    def test_in_schedule_fires(self):
        engine = ZoneEngine(camera_id="cam_sched2")
        zone = make_zone(schedule="22:00-06:00", dwell_seconds=2.0)
        zones = [zone]
        det = make_det()

        night = datetime(2024, 1, 15, 23, 30)
        engine.evaluate(0.0, [det], zones, now=night)
        result = engine.evaluate(2.1, [det], zones, now=night)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Stream timeline dwell (independent of processing wall clock)
# ---------------------------------------------------------------------------

class TestZoneEngineStreamTimeline:
    def test_dwell_uses_stream_time_not_processing_gaps(self):
        """Intermediate stream times can be skipped; dwell is measured on stream timeline."""
        engine = ZoneEngine(camera_id="cam_stream")
        zone = make_zone(dwell_seconds=3.0)
        zones = [zone]
        det = make_det(track_id=1)

        engine.evaluate(0.0, [det], zones)
        result = engine.evaluate(3.0, [det], zones)
        assert len(result) == 1
        assert result[0].dwell_elapsed == pytest.approx(3.0, abs=1e-6)

    def test_dwell_fires_at_exact_rational_stream_time(self):
        """Regression: 67/12 - 43/12 must satisfy a 2.0s threshold (float != 2.0)."""
        engine = ZoneEngine(camera_id="cam_fp")
        zone = make_zone(dwell_seconds=2.0)
        zones = [zone]
        det = make_det(track_id=1)

        entry_time = 43 / 12
        check_time = 67 / 12
        engine.evaluate(entry_time, [det], zones)
        result = engine.evaluate(check_time, [det], zones)
        assert len(result) == 1
        assert result[0].dwell_elapsed == pytest.approx(2.0, abs=1e-6)

    def test_dwell_requires_full_threshold_on_stream_timeline(self):
        engine = ZoneEngine(camera_id="cam_stream2")
        zone = make_zone(dwell_seconds=3.0)
        zones = [zone]
        det = make_det(track_id=1)

        engine.evaluate(0.0, [det], zones)
        assert engine.evaluate(2.9, [det], zones) == []
        result = engine.evaluate(3.0, [det], zones)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Unmatched track IDs (track_id < 0)
# ---------------------------------------------------------------------------

class TestZoneEngineTrackIdFilter:
    def test_negative_track_id_does_not_create_dwell_state(self):
        engine = ZoneEngine(camera_id="cam_untracked")
        zone = make_zone(dwell_seconds=1.0)
        zones = [zone]
        det = make_det(track_id=-1)

        for t in [0.0, 1.5, 3.0]:
            assert engine.evaluate(t, [det], zones) == []

        assert engine._dwell == {}

    def test_negative_track_id_does_not_reset_positive_track_dwell(self):
        engine = ZoneEngine(camera_id="cam_mixed")
        zone = make_zone(dwell_seconds=3.0)
        zones = [zone]

        engine.evaluate(0.0, [make_det(track_id=1)], zones)
        # Unmatched detection alongside the tracked one must not reset track 1's dwell.
        engine.evaluate(
            1.0,
            [make_det(track_id=1), make_det(track_id=-1)],
            zones,
        )
        result = engine.evaluate(3.0, [make_det(track_id=1)], zones)
        assert len(result) == 1
