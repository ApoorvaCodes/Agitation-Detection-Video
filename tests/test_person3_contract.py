import pytest
from person3.contracts import CandidateBehaviour, ValidationResult
from person3.taxonomy import normalize_behaviour


def test_candidate_strict_validation_and_version():
    with pytest.raises(ValueError):
        CandidateBehaviour(candidate_id="x", person_id="p", behaviour="pushing", candidate_score=2,
                           start_timestamp=1, end_timestamp=2)
    with pytest.raises(ValueError):
        CandidateBehaviour(candidate_id="x", person_id="p", behaviour="pushing", candidate_score=.5,
                           start_timestamp=2, end_timestamp=2)
    with pytest.raises(ValueError):
        CandidateBehaviour(candidate_id="x", person_id="p", behaviour="pushing", candidate_score=.5,
                           start_timestamp=1, end_timestamp=2, surprise=True)


def test_supported_requires_source_interval_and_taxonomy_lookup():
    assert normalize_behaviour("pushing").key == "pushing"
    with pytest.raises(ValueError):
        ValidationResult(event_id="e", person_id="p", candidate_id="c", behaviour="pushing", candidate_score=.5,
                         validation_status="supported", reason="ok")
