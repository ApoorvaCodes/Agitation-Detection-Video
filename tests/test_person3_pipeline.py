from person3.contracts import Verification
from person3.pipeline import validate_candidates
from person3_fixtures import candidate, make_source


class FakeVerifier:
    def __init__(self, result): self.result = result; self.packet = None
    def validate(self, packet): self.packet = packet; return self.result


def test_supported_event_and_fail_closed_verifier_error():
    verifier = FakeVerifier(Verification(decision="supported", reason="Evidence supports review.", evidence_segment_ids=["person_0001:frame:1", "person_0001:frame:2"]))
    result = validate_candidates(make_source(), [candidate()], verifier)[0]
    assert result.validation_status == "supported" and (result.start_timestamp, result.end_timestamp) == (1, 2)
    assert verifier.packet.candidate_id == "cand_001"
    class Broken:
        def validate(self, packet): raise TimeoutError("offline")
    failed = validate_candidates(make_source(), [candidate()], Broken())[0]
    assert failed.validation_status == "insufficient_evidence" and failed.verifier_error == "offline"


def test_taxonomy_failure_skips_verifier():
    class Never:
        def validate(self, packet): raise AssertionError("should not call verifier")
    result = validate_candidates(make_source(), [candidate(behaviour="unknown")], Never())[0]
    assert result.validation_status == "insufficient_evidence"


def test_unsupported_and_insufficient_verifier_decisions_are_preserved():
    for decision in ("unsupported", "insufficient_evidence"):
        verifier = FakeVerifier(Verification(decision=decision, reason="Evidence does not support the candidate.", evidence_segment_ids=[]))
        result = validate_candidates(make_source(), [candidate()], verifier)[0]
        assert result.validation_status == decision
        assert result.start_timestamp is None and result.end_timestamp is None
