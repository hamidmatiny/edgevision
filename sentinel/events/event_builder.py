"""
event_builder.py — Structured Incident Record Builder

Converts a CandidateEvent (from zone_engine.py) into a fully-structured
IncidentRecord dict and saves it as a JSON file in the evidence/ folder.

The IncidentRecord schema is the stable contract between the edge pipeline
and the future cloud/dashboard layer. Keep it versioned.

Schema (v1):
  {
    "schema_version": "1.0",
    "event_id": str,                   # UUID
    "camera_id": str,
    "zone_name": str,
    "track_id": int,
    "detection_class": str,
    "confidence": float,
    "centroid": [float, float],
    "trigger_wall_time": float,        # Unix timestamp
    "trigger_wall_time_iso": str,      # ISO 8601
    "dwell_elapsed_seconds": float,
    "clip_path": str | null,           # set after clip is written; null initially
    "vlm_verification": null,          # placeholder for Stage 3 (Phase 3)
    "audit_chain": {                   # explainability / audit trail
      "stage1_confidence": float,
      "stage2_zone_rule": str,
      "stage3_vlm": null,              # Phase 3
      "final_decision": "candidate"    # will become "confirmed"/"rejected" after VLM
    }
  }
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sentinel.rules.zone_engine import CandidateEvent

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"


def build_event(
    candidate: CandidateEvent,
    wall_time: float,
    clip_path: Optional[str] = None,
) -> dict:
    """
    Build a structured IncidentRecord dict from a CandidateEvent.

    Args:
        candidate: The CandidateEvent emitted by ZoneEngine.
        wall_time: Unix wall-clock time of the trigger (time.time()).
        clip_path: Path to the evidence clip, if already written. Can be updated later.

    Returns:
        IncidentRecord dict (not yet persisted — call save_event() to write to disk).
    """
    event_id = str(uuid.uuid4())
    iso_time = datetime.fromtimestamp(wall_time, tz=timezone.utc).isoformat()

    record = {
        "schema_version": SCHEMA_VERSION,
        "event_id": event_id,
        "camera_id": candidate.camera_id,
        "zone_name": candidate.zone_name,
        "track_id": candidate.track_id,
        "detection_class": candidate.detection_class,
        "confidence": round(candidate.confidence, 4),
        "centroid": [round(candidate.centroid[0], 2), round(candidate.centroid[1], 2)],
        "trigger_wall_time": wall_time,
        "trigger_wall_time_iso": iso_time,
        "dwell_elapsed_seconds": round(candidate.dwell_elapsed, 3),
        "clip_path": clip_path,
        # Placeholder for Phase 3 VLM verification — do not remove this key;
        # the cloud layer depends on its presence (null signals "not yet verified").
        "vlm_verification": None,
        "audit_chain": {
            "stage1_confidence": round(candidate.confidence, 4),
            "stage2_zone_rule": (
                f"zone='{candidate.zone_name}' "
                f"dwell={round(candidate.dwell_elapsed, 2)}s"
            ),
            # Stage 3 VLM slot — populated in Phase 3
            "stage3_vlm": None,
            # Decision before VLM: "candidate" means Stage 1+2 passed, VLM pending
            "final_decision": "candidate",
        },
    }

    return record


def save_event(record: dict, output_dir: str | Path = "evidence") -> str:
    """
    Write an IncidentRecord dict to a JSON file in output_dir.

    File is named: <camera_id>_<event_id>.json

    Returns the absolute path of the written file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{record['camera_id']}_{record['event_id']}.json"
    out_path = output_dir / filename

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    logger.info(
        "Event record saved: %s (camera=%s zone=%s track=%d class=%s conf=%.2f)",
        out_path,
        record["camera_id"],
        record["zone_name"],
        record["track_id"],
        record["detection_class"],
        record["confidence"],
    )

    return str(out_path)


def update_clip_path(record: dict, clip_path: str) -> dict:
    """
    Return a copy of the record with the clip_path field filled in.
    Use this when the clip is written asynchronously after the event record.
    """
    updated = dict(record)
    updated["clip_path"] = clip_path
    return updated


def rewrite_event(record: dict, path: str | Path) -> None:
    """Overwrite an existing IncidentRecord JSON file on disk."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
