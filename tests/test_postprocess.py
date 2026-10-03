from person3.contracts import Verification
from person3.evidence import build_evidence_packet
from person3.postprocess import deduplicate_events, result_from_verification
from person3_fixtures import candidate, make_source


def test_selected_ids_ground_timestamps_and_unknown_ids_abstain():
    c = candidate(); packet = build_evidence_packet(c, make_source().persons[0])
    result = result_from_verification(c, packet, Verification(decision="supported", reason="Observed movement.", evidence_segment_ids=["person_0001:frame:1", "person_0001:frame:3"]))
    assert (result.start_timestamp, result.end_timestamp) == (1, 3)
    bad = result_from_verification(c, packet, Verification(decision="supported", reason="X", evidence_segment_ids=["invented"]))
    assert bad.validation_status == "insufficient_evidence" and bad.start_timestamp is None


def test_dedup_same_behaviour_but_retain_overlapping_other_behaviour():
    p1 = build_evidence_packet(candidate(), make_source().persons[0])
    v = Verification(decision="supported", reason="ok", evidence_segment_ids=["person_0001:frame:1", "person_0001:frame:3"])
    one = result_from_verification(candidate(), p1, v)
    duplicate = result_from_verification(candidate(candidate_id="cand_002", candidate_score=.7), p1, v)
    other_c = candidate(candidate_id="cand_003", behaviour="hitting")
    other = result_from_verification(other_c, build_evidence_packet(other_c, make_source().persons[0]), v)
    output = deduplicate_events([one, duplicate, other])
    assert len(output) == 2 and {e.behaviour for e in output} == {"Pushing", "Hitting"}


def test_overlapping_same_behaviour_events_are_deduplicated():
    source = make_source(6)
    first = candidate(source_observation_ids=[f"person_0001:frame:{i}" for i in range(4)])
    second = candidate(candidate_id="cand_002", start_timestamp=1, end_timestamp=4,
                       source_observation_ids=[f"person_0001:frame:{i}" for i in range(1, 5)])
    one = result_from_verification(first, build_evidence_packet(first, source.persons[0]),
        Verification(decision="supported", reason="ok", evidence_segment_ids=["person_0001:frame:0", "person_0001:frame:3"]))
    two = result_from_verification(second, build_evidence_packet(second, source.persons[0]),
        Verification(decision="supported", reason="ok", evidence_segment_ids=["person_0001:frame:1", "person_0001:frame:4"]))
    assert len(deduplicate_events([one, two])) == 1
