"""Tests for stream_reader timeline helpers."""

import pytest

from sentinel.ingestion.stream_reader import StreamReader


class TestStreamTimeline:
    def test_stream_time_from_raw_frame_index(self):
        reader = StreamReader(source="dummy.mp4", frame_skip=1)
        reader._source_fps = 12.0

        assert reader._stream_time_seconds(1) == pytest.approx(0.0)
        assert reader._stream_time_seconds(13) == pytest.approx(1.0)
        assert reader._stream_time_seconds(25) == pytest.approx(2.0)

    def test_stream_time_defaults_fps_when_missing(self):
        reader = StreamReader(source="dummy.mp4", frame_skip=1)
        reader._source_fps = 30.0

        assert reader._stream_time_seconds(31) == pytest.approx(1.0)
