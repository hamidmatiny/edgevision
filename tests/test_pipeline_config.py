"""Tests for pipeline configuration helpers."""

from pathlib import Path

from sentinel.pipeline import load_zones_for_camera

FIXTURE_ZONES = Path(__file__).parent / "fixtures" / "zones.yaml"


class TestLoadZonesForCamera:
    def test_reads_top_level_zones_wrapper(self):
        zones = load_zones_for_camera(str(FIXTURE_ZONES), "cam1")
        assert len(zones) == 1
        assert zones[0].name == "perimeter_east"
        assert zones[0].dwell_seconds == 3.0

    def test_loads_multiple_cameras_from_same_file(self):
        cam2_zones = load_zones_for_camera(str(FIXTURE_ZONES), "cam2")
        assert len(cam2_zones) == 1
        assert cam2_zones[0].name == "gate_entrance"

    def test_unknown_camera_returns_empty(self):
        zones = load_zones_for_camera(str(FIXTURE_ZONES), "cam_missing")
        assert zones == []

    def test_loads_committed_repo_zones_yaml(self):
        repo_zones = Path(__file__).resolve().parents[1] / "config" / "zones.yaml"
        zones = load_zones_for_camera(str(repo_zones), "cam1")
        assert len(zones) == 2
        zone_names = {z.name for z in zones}
        assert zone_names == {"perimeter_east", "restricted_storage"}
