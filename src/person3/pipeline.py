"""Person 3 candidate validation pipeline."""
from person3.contracts import Verification
from person3.evidence import build_evidence_packet, candidates_from_p2
from person3.postprocess import deduplicate_events, result_from_verification
from person3.qwen_validator import GroqQwenValidator
from person3.validator import validate_packet


def validate_candidates(p1_result, candidates, verifier=None, deduplicate=True):
    verifier = verifier or GroqQwenValidator()
    people = {p.person_id: p for p in p1_result.persons}
    results = []
    for candidate in candidates:
        person = people.get(candidate.person_id)
        if person is None:
            continue
        packet = build_evidence_packet(candidate, person)
        valid, flags = validate_packet(packet)
        if not valid:
            response = Verification(decision="insufficient_evidence", reason="CMAI evidence quality checks did not pass.", evidence_segment_ids=[])
            results.append(result_from_verification(candidate, packet, response, flags))
            continue
        try:
            response = verifier.validate(packet)
            if not isinstance(response, Verification):
                response = Verification.model_validate(response)
            results.append(result_from_verification(candidate, packet, response, flags))
        except Exception as exc:
            response = Verification(decision="insufficient_evidence", reason="Qwen verification unavailable; review evidence locally.", evidence_segment_ids=[])
            results.append(result_from_verification(candidate, packet, response, flags, verifier_error=str(exc)[:500]))
    return deduplicate_events(results) if deduplicate else results


def validate_p2_result(p1_result, p2_result, verifier=None):
    candidates = candidates_from_p2(p2_result, p1_result)
    return validate_candidates(p1_result, candidates, verifier)
