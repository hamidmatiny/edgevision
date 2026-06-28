"""Tests for pipeline clip-path wiring."""

from sentinel.pipeline import CameraPipeline


class TestEventIdFromClipPath:
    def test_parses_uuid_from_standard_clip_filename(self):
        event_id = CameraPipeline._event_id_from_clip_path(
            "evidence/cam1_42594ca3-2eec-4778-83c6-e0a68b42ebdf_1782609319.mp4"
        )
        assert event_id == "42594ca3-2eec-4778-83c6-e0a68b42ebdf"

    def test_returns_none_for_unexpected_filename(self):
        assert CameraPipeline._event_id_from_clip_path("evidence/badname.mp4") is None
