from person3.evidence import candidates_from_p2
from person3.p2_adapter import read_p2_handoff
from person3_fixtures import make_source


def test_existing_p2_schema_event_maps_chunk_frames_to_p1_ids():
    raw = {"schema_version": "1.0", "source_schema_version": "1.0", "persons": [{
        "person_id": "person_0001", "chunks": [{"chunk_id": "person_0001:0", "frame_indices": [0, 1, 2, 3],
            "scores": [{"behaviour": "pushing", "smoothed_similarity": .82}]}],
        "events": [{"behaviour": "pushing", "start_timestamp": 0, "end_timestamp": 3,
                    "peak_similarity": .8, "chunk_ids": ["person_0001:0"]}]}]}
    p2 = read_p2_handoff(raw)
    candidates = candidates_from_p2(p2, make_source())
    assert len(candidates) == 1
    assert candidates[0].candidate_score == .82
    assert candidates[0].source_observation_ids == [f"person_0001:frame:{i}" for i in range(4)]
