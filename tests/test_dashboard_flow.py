import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from person2.pipeline import process_perception
from person3.dashboard_flow import analyze_video, read_dashboard_results, resolve_groq_key
from person3_fixtures import make_source
from dashboard import automatic_verification


def test_key_entry_secret_and_environment_precedence(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "environment-key")
    assert resolve_groq_key(" entered-key ", "secret-key") == "entered-key"
    assert resolve_groq_key("", "secret-key") == "secret-key"
    assert resolve_groq_key() == "environment-key"
    monkeypatch.delenv("GROQ_API_KEY")
    assert resolve_groq_key() == ""


def test_video_orchestration_calls_p1_then_p2_without_groq(tmp_path):
    path = tmp_path / "source.mp4"
    path.write_bytes(b"test-only-video-placeholder")
    source = make_source()
    calls = []

    def perception_runner(path, config):
        calls.append(("p1", path))
        return source

    def candidate_runner(p1, config, prototypes):
        assert p1 is source
        assert prototypes is None
        calls.append(("p2", p1.video.video_id))
        return process_perception(p1, config)

    p1, p2 = analyze_video(path, perception_runner=perception_runner, candidate_runner=candidate_runner)
    assert [name for name, _ in calls] == ["p1", "p2"]
    assert p1 is source
    assert p2.prototype_version is None
    assert all(not p.events for p in p2.persons)


def test_dashboard_results_must_match_video():
    source = make_source()
    raw = process_perception(source).model_dump(mode="json")
    p1, p2 = read_dashboard_results(source.model_dump_json(), json.dumps(raw))
    assert p1.video.video_id == source.video.video_id
    assert p2.schema_version == "1.0"
    raw["video_id"] = "different-video"
    with pytest.raises(ValueError, match="same video"):
        read_dashboard_results(source.model_dump_json(), json.dumps(raw))


def test_automatic_verification_skips_qwen_without_candidates():
    p2 = SimpleNamespace(persons=[SimpleNamespace(events=[])])
    with patch("dashboard.GroqQwenValidator") as verifier:
        assert automatic_verification(make_source(), p2, "key", "qwen")[0] == "no_candidates"
        verifier.assert_not_called()


def test_automatic_verification_reports_missing_key_for_candidates():
    p2 = SimpleNamespace(persons=[SimpleNamespace(events=[object()])])
    status, reviews, error = automatic_verification(make_source(), p2, "", "qwen")
    assert (status, reviews) == ("unavailable", [])
    assert "GROQ_API_KEY" in error


def test_automatic_verification_calls_p3_for_candidates():
    p2 = SimpleNamespace(persons=[SimpleNamespace(events=[object()])])
    with patch("dashboard.GroqQwenValidator") as verifier, patch("dashboard.validate_p2_result", return_value=[]) as validate:
        status, reviews, error = automatic_verification(make_source(), p2, "key", "qwen")
    assert (status, reviews, error) == ("complete", [], None)
    verifier.assert_called_once_with(api_key="key", model="qwen")
    validate.assert_called_once()


def test_automatic_verification_reports_qwen_failure():
    p2 = SimpleNamespace(persons=[SimpleNamespace(events=[object()])])
    with patch("dashboard.GroqQwenValidator", side_effect=TimeoutError("offline")):
        status, reviews, error = automatic_verification(make_source(), p2, "key", "qwen")
    assert status == "failed" and not reviews and "offline" in error
