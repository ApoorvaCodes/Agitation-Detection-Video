from person3.evidence import build_evidence_packet
from person3_fixtures import candidate, make_source


def test_packet_is_compact_and_traceable():
    packet = build_evidence_packet(candidate(), make_source().persons[0])
    assert packet.candidate_id == "cand_001"
    assert packet.source_window_ids == ["person_0001:0"]
    assert [x.timestamp for x in packet.segments] == [0, 1, 2, 3]
    assert "left_wrist.velocity.speed" in packet.segments[0].motion_features
