"""UI regression checks use fake inputs and mocked stages, never Groq calls."""
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from person2.pipeline import process_perception
from person2.config import Person2Config
from person2.prototypes import build_prototypes
from person3_fixtures import make_source

APP = Path(__file__).resolve().parents[1] / "dashboard.py"


def make_app(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    return AppTest.from_file(str(APP), default_timeout=15)


def test_initial_screen_is_video_first_and_requests_masked_key(monkeypatch):
    app = make_app(monkeypatch).run()
    assert not app.exception
    assert app.radio[0].value == "Video"
    assert app.text_input[0].label == "Groq API key"
    assert app.text_input[0].proto.type == app.text_input[0].proto.PASSWORD
    assert app.button[0].label == "Analyse video" and app.button[0].disabled
    assert any("Upload a video" in item.value for item in app.info)


def test_entered_key_is_configured_and_json_mode_is_optional(monkeypatch):
    app = make_app(monkeypatch).run()
    app.text_input[0].set_value("test-key-not-real").run()
    assert not app.exception
    assert any("API key configured" in item.value for item in app.caption)
    app.radio[0].set_value("Existing results (advanced)").run()
    assert not app.exception
    assert any("matching P1 and P2 results" in item.value for item in app.info)


def test_video_button_runs_both_stages_and_does_not_verify_without_candidates(monkeypatch):
    uploaded = BytesIO(b"unit-fixture-only")
    uploaded.name = "fixture.mp4"
    p1 = make_source()
    p2 = process_perception(p1)
    with patch("streamlit.file_uploader", side_effect=lambda label, **kwargs: uploaded if label == "Upload video" else None), \
         patch("person3.dashboard_flow.analyze_video", return_value=(p1, p2)) as analyze:
        app = make_app(monkeypatch).run()
        assert not app.exception
        app.button[0].click().run()
        assert not app.exception
        assert not app.error, [e.value for e in app.error]
        analyze.assert_called_once()
        assert any("Video analysis completed" in x.value for x in app.success)
        assert any("no labels were generated" in x.value for x in app.info)
        verify = next(b for b in app.button if b.label == "Verify candidates with Groq")
        assert verify.disabled


def test_changing_video_clears_previous_results(monkeypatch):
    first = BytesIO(b"first-unit-fixture")
    first.name = "fixture.mp4"
    second = BytesIO(b"second-unit-fixture")
    second.name = "fixture.mp4"
    p1 = make_source()
    current = [first]
    with patch("streamlit.file_uploader", side_effect=lambda label, **kwargs: current[0] if label == "Upload video" else None), \
         patch("person3.dashboard_flow.analyze_video", return_value=(p1, process_perception(p1))):
        app = make_app(monkeypatch).run()
        app.button[0].click().run()
        previous_path = Path(app.session_state["source_path"])
        assert "p1" in app.session_state
        current[0] = second
        app.run()
        assert not app.exception
        assert "p1" not in app.session_state
        assert not previous_path.exists()


def test_groq_uses_entered_key_only_after_verify_click(monkeypatch):
    p1 = make_source()
    config = Person2Config(min_frames=2, overlap=0)
    descriptor = process_perception(p1, config).persons[0].chunks[0].fused_embedding
    bank = build_prototypes([("general_restlessness", descriptor)], "unit-test-only")
    p2 = process_perception(p1, config, bank)
    assert p2.persons[0].events
    files = {"Person 1 result JSON": BytesIO(p1.model_dump_json().encode()),
             "Person 2 candidate result JSON": BytesIO(p2.model_dump_json().encode())}
    with patch("streamlit.file_uploader", side_effect=lambda label, **kwargs: files.get(label)), \
         patch("person3.qwen_validator.GroqQwenValidator") as verifier, \
         patch("person3.pipeline.validate_p2_result", return_value=[]) as validate:
        app = make_app(monkeypatch).run()
        app.radio[0].set_value("Existing results (advanced)").run()
        button = next(b for b in app.button if b.label == "Verify candidates with Groq")
        assert button.disabled
        app.text_input[0].set_value("test-key-not-real").run()
        assert not app.exception
        verifier.assert_not_called()
        button = next(b for b in app.button if b.label == "Verify candidates with Groq")
        assert not button.disabled
        button.click().run()
        assert not app.exception
        verifier.assert_called_once_with(api_key="test-key-not-real", model=app.text_input[1].value)
        assert validate.call_args.kwargs["verifier"] is verifier.return_value
