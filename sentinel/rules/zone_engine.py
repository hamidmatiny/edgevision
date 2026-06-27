"""
zone_engine.py — Stage 2: Rule Engine

Responsibilities:
  - Point-in-polygon check (is a detection centroid inside a configured zone?)
  - Schedule check (is the zone active right now, given its time window?)
  - Dwell-time state machine (has a tracked object been inside long enough to be a candidate?)

Design notes:
  - Uses shapely for the polygon check; do not hand-roll point-in-polygon.
  - Dwell timer is per (track_id, zone_name, camera_id) triple.
  - If a track LEAVES the zone and RE-ENTERS, the dwell timer RESETS to zero on re-entry.
    This is explicit policy: a transient walk-through that leaves and comes back is treated
    as a new event candidate, not accumulated dwell time.
  - This module is stateful (holds dwell timers) and must be instantiated once per camera
    pipeline and kept alive across frames.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Optional

from shapely.geometry import Point, Polygon

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ZoneConfig:
    """Parsed representation of one zone entry from zones.yaml."""
    name: str
    polygon: Polygon
    schedule: str          # "always" or "HH:MM-HH:MM"
    dwell_seconds: float
    classes: list[str]


@dataclass
class DwellState:
    """Mutable dwell-time tracking for one (track_id, zone) pair."""
    inside: bool = False
    entry_time: Optional[float] = None   # monotonic timestamp (seconds) of when the track entered


@dataclass
class CandidateEvent:
    """Emitted by the zone engine when dwell threshold is crossed."""
    camera_id: str
    zone_name: str
    track_id: int
    detection_class: str
    confidence: float
    centroid: tuple[float, float]
    trigger_time: float        # monotonic timestamp
    dwell_elapsed: float       # seconds spent continuously inside the zone


# ---------------------------------------------------------------------------
# Zone parsing helpers
# ---------------------------------------------------------------------------

def parse_zone_config(raw: dict) -> ZoneConfig:
    """Convert a raw dict (from zones.yaml) into a ZoneConfig."""
    polygon_points = [tuple(pt) for pt in raw["polygon"]]
    if len(polygon_points) < 3:
        raise ValueError(f"Zone '{raw['name']}' must have at least 3 polygon points.")
    return ZoneConfig(
        name=raw["name"],
        polygon=Polygon(polygon_points),
        schedule=raw.get("schedule", "always"),
        dwell_seconds=float(raw.get("dwell_seconds", 3)),
        classes=[c.lower() for c in raw.get("classes", ["person"])],
    )


# ---------------------------------------------------------------------------
# Schedule check
# ---------------------------------------------------------------------------

def _parse_time(t_str: str) -> time:
    """Parse 'HH:MM' into a datetime.time object."""
    h, m = t_str.strip().split(":")
    return time(int(h), int(m))


def is_schedule_active(schedule: str, now: Optional[datetime] = None) -> bool:
    """
    Return True if the current time falls within the zone's active schedule.

    schedule formats:
      - "always"           → always active
      - "HH:MM-HH:MM"     → active within that window (wraps midnight correctly,
                             e.g. "22:00-06:00" is active from 10pm to 6am)
    """
    if schedule.strip().lower() == "always":
        return True

    if now is None:
        now = datetime.now()

    current = now.time().replace(second=0, microsecond=0)

    parts = schedule.split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid schedule format: '{schedule}'. Expected 'HH:MM-HH:MM' or 'always'.")

    start = _parse_time(parts[0])
    end = _parse_time(parts[1])

    if start <= end:
        # Normal window (e.g. "08:00-17:00")
        return start <= current < end
    else:
        # Wraps midnight (e.g. "22:00-06:00")
        return current >= start or current < end


# ---------------------------------------------------------------------------
# Point-in-polygon check
# ---------------------------------------------------------------------------

def is_inside_zone(centroid: tuple[float, float], zone: ZoneConfig) -> bool:
    """Return True if the centroid (x, y) is inside the zone's polygon."""
    return zone.polygon.contains(Point(centroid))


# ---------------------------------------------------------------------------
# Dwell-time state machine
# ---------------------------------------------------------------------------

class ZoneEngine:
    """
    Stateful zone engine for one camera stream.

    Call `evaluate(frame_time, detections, zones)` every processed frame.
    It returns a list of CandidateEvent objects for any tracks that crossed
    the dwell threshold since the last call.

    Thread-safety: not thread-safe. Each camera pipeline should own its own instance.
    """

    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        # Key: (track_id, zone_name) → DwellState
        self._dwell: dict[tuple[int, str], DwellState] = {}
        # Track which (track_id, zone_name) pairs have already fired an event this
        # continuous dwell session (to avoid re-firing on every frame after threshold).
        self._fired: set[tuple[int, str]] = set()

    def evaluate(
        self,
        frame_time: float,           # monotonic time in seconds (e.g. time.monotonic())
        detections: list[dict],      # each: {track_id, class, confidence, centroid: (x,y)}
        zones: list[ZoneConfig],
        now: Optional[datetime] = None,
    ) -> list[CandidateEvent]:
        """
        Evaluate one frame's detections against all zones.

        Returns a list of CandidateEvent for any (track, zone) pairs that:
          1. Are inside the zone polygon
          2. Zone is on schedule
          3. Detection class is in zone's class list
          4. Have been continuously inside for >= dwell_seconds (fires once per continuous session)
        """
        candidates: list[CandidateEvent] = []

        # Build a set of active (track_id, zone_name) pairs this frame
        active_pairs: set[tuple[int, str]] = set()

        for det in detections:
            track_id: int = det["track_id"]
            det_class: str = det["class"].lower()
            confidence: float = det["confidence"]
            centroid: tuple[float, float] = det["centroid"]

            for zone in zones:
                key = (track_id, zone.name)

                # Check class filter
                if det_class not in zone.classes:
                    continue

                # Check schedule
                if not is_schedule_active(zone.schedule, now=now):
                    continue

                # Check spatial containment
                if not is_inside_zone(centroid, zone):
                    continue

                active_pairs.add(key)

                state = self._dwell.get(key)

                if state is None or not state.inside:
                    # Track just entered the zone (or is re-entering after leaving)
                    self._dwell[key] = DwellState(inside=True, entry_time=frame_time)
                    self._fired.discard(key)   # reset fired flag on re-entry
                    logger.debug(
                        "Track %d entered zone '%s' on camera %s at t=%.2f",
                        track_id, zone.name, self.camera_id, frame_time,
                    )
                    continue

                # Track was already inside; check dwell elapsed
                dwell_elapsed = frame_time - state.entry_time  # type: ignore[operator]

                if dwell_elapsed >= zone.dwell_seconds and key not in self._fired:
                    self._fired.add(key)
                    logger.info(
                        "CANDIDATE EVENT: track=%d zone='%s' camera=%s class=%s "
                        "dwell=%.1fs conf=%.2f centroid=%s",
                        track_id, zone.name, self.camera_id, det_class,
                        dwell_elapsed, confidence, centroid,
                    )
                    candidates.append(CandidateEvent(
                        camera_id=self.camera_id,
                        zone_name=zone.name,
                        track_id=track_id,
                        detection_class=det_class,
                        confidence=confidence,
                        centroid=centroid,
                        trigger_time=frame_time,
                        dwell_elapsed=dwell_elapsed,
                    ))

        # Update state for tracks that have LEFT zones since last frame
        all_known_keys = set(self._dwell.keys())
        departed_keys = all_known_keys - active_pairs

        for key in departed_keys:
            state = self._dwell.get(key)
            if state and state.inside:
                track_id, zone_name = key
                logger.debug(
                    "Track %d left zone '%s' on camera %s",
                    track_id, zone_name, self.camera_id,
                )
                # Mark as outside; dwell timer will reset on next re-entry (see above)
                self._dwell[key] = DwellState(inside=False, entry_time=None)
                self._fired.discard(key)

        return candidates

    def purge_stale_tracks(self, active_track_ids: set[int]) -> None:
        """
        Remove dwell state for tracks that the tracker has dropped entirely.
        Call this periodically (e.g. every N frames) to avoid unbounded memory growth.
        """
        stale = [key for key in self._dwell if key[0] not in active_track_ids]
        for key in stale:
            del self._dwell[key]
            self._fired.discard(key)
        if stale:
            logger.debug("Purged %d stale track states on camera %s", len(stale), self.camera_id)
