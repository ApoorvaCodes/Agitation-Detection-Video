"""Validation safeguards run without native model initialization or downloads."""
import json
from pathlib import Path
import pytest
from cmai.runtime_validation import validate_video

ROOT=Path(__file__).resolve().parents[1]


def test_requires_source_and_existing_local_weights(tmp_path):
    with pytest.raises(ValueError,match='source video'):validate_video(tmp_path/'missing.mp4',tmp_path/'out',ROOT/'configs/default.yaml')
    video=tmp_path/'source.mp4';video.write_bytes(b'corrupt-test-only')
    config=tmp_path/'config.yaml';config.write_text('yolo_model: /missing/checkpoint.pt\n')
    with pytest.raises(ValueError,match='existing local YOLO'):validate_video(video,tmp_path/'out',config)
    assert not (tmp_path/'out').exists()


def test_preserves_prior_validation_evidence(tmp_path):
    video=tmp_path/'source.mp4';video.write_bytes(b'test-only')
    output=tmp_path/'out';output.mkdir();(output/'earlier-report.json').write_text('{}')
    with pytest.raises(ValueError,match='empty output'):validate_video(video,output,ROOT/'configs/default.yaml')
    assert (output/'earlier-report.json').read_text()=='{}'


def test_worker_error_is_recorded_without_false_success(tmp_path):
    video=tmp_path/'source.mp4';video.write_bytes(b'not-a-video')
    model=tmp_path/'test.pt';model.write_bytes(b'not-loaded-preflight-first')
    config=tmp_path/'config.yaml';config.write_text('yolo_model: '+str(model)+'\n')
    report=validate_video(video,tmp_path/'out',config,input_kind='synthetic_mechanics_only')
    assert report['status']=='failed'
    assert 'Unable to open video decoder' in report['worker']['message']
    assert 'candidates' not in report
    assert json.loads((tmp_path/'out/runtime-report.json').read_text())==report
    assert not (tmp_path/'out/camera-result.json').exists()
