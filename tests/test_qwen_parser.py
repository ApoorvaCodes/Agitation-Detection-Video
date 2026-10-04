import pytest
from person3.qwen_validator import canonical_verification_labels, parse_verification


@pytest.mark.parametrize("text", [
    '{"decision":"supported","reason":"Motion supports the candidate.","evidence_segment_ids":["p:frame:1"]}',
    'Result follows: ```json\n{"decision":"unsupported","reason":"No support.","evidence_segment_ids":[]}\n```',
])
def test_qwen_json_variants(text):
    result = parse_verification(text)
    assert result.decision in {"supported", "unsupported"}


def test_malformed_or_missing_fields_rejected():
    with pytest.raises(Exception): parse_verification("not json")
    with pytest.raises(Exception): parse_verification('{"decision":"maybe"}')


def test_remote_verifier_vocabulary_is_canonical_camera_ids():
    labels = canonical_verification_labels()
    assert labels[0] == "cmai_01_pacing_aimless_wandering"
    assert len(labels) == 29
    assert "Pushing" not in labels
