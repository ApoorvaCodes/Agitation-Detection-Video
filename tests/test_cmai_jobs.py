"""No external calls or model downloads; corrupt recordings fail preflight."""
import time
from unittest.mock import patch, MagicMock

from cmai.bundle import load_bundle
from cmai.jobs import start_analysis
from person1.config import Person1Config
import cv2
import pytest


def test_corrupt_recording_reports_precise_error_before_models(tmp_path):
    path = tmp_path / "source.mp4"
    path.write_bytes(b"not-a-video")
    job = start_analysis(path, load_bundle(), Person1Config())
    try:
        deadline = time.monotonic() + 10
        status = job.status()
        while status["phase"] == "running" and time.monotonic() < deadline:
            time.sleep(.05)
            status = job.status()
        assert status["phase"] == "error"
        assert "Unable to open video decoder" in status["message"]
        assert not (tmp_path / "perception.json").exists()
    finally:
        job.cancel()


def test_cancellation_terminates_worker_and_discards_partial_outputs(tmp_path):
    path = tmp_path / "source.mp4"
    path.write_bytes(b"fixture")
    (tmp_path / "progress.json").write_text('{"phase":"complete"}')
    fake = MagicMock()
    fake.poll.return_value = None
    with patch("cmai.jobs.subprocess.Popen", return_value=fake):
        job = start_analysis(path, load_bundle(), Person1Config())
        assert not (tmp_path / "progress.json").exists()
        assert job.status()["phase"] == "running"
        job.cancel()
        fake.terminate.assert_called_once()
        assert job.status()["phase"] == "cancelled"


@pytest.mark.parametrize("fps", [29.97, float("nan"), float("inf")])
def test_fractional_fps_is_preserved_and_invalid_metadata_releases_decoder(tmp_path, fps):
    from person1.ingestion import VideoLoader
    from person1.errors import VideoMetadataError
    path = tmp_path / "fixture.mp4"
    path.write_bytes(b"mock-decoder-input")
    cap = MagicMock()
    cap.isOpened.return_value = True
    properties = {cv2.CAP_PROP_FPS:fps, cv2.CAP_PROP_FRAME_COUNT:30,
                  cv2.CAP_PROP_FRAME_WIDTH:64, cv2.CAP_PROP_FRAME_HEIGHT:64, cv2.CAP_PROP_FOURCC:0}
    cap.get.side_effect = lambda name: properties[name]
    with patch("person1.ingestion.cv2.VideoCapture", return_value=cap):
        if fps == 29.97:
            with VideoLoader(path) as loader:
                assert loader.metadata.fps == 29.97
                assert loader.metadata.duration_seconds == pytest.approx(30/29.97)
        else:
            with pytest.raises(VideoMetadataError, match="Invalid FPS"):
                VideoLoader(path)
        cap.release.assert_called_once()
