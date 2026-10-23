"""Reader for the existing Person 2 schema 1.0 candidate handoff.

Only fields used by Person 3 are projected; source scores are preserved without
reinterpreting their uncalibrated cosine semantics.
"""
from types import SimpleNamespace as NS


def read_p2_handoff(raw):
    if raw.get("schema_version") != "1.0" or raw.get("source_schema_version") != "1.0":
        raise ValueError("Person 3 currently accepts Person 2 / Person 1 schema 1.0")
    people = []
    for person in raw.get("persons", []):
        chunks = []
        for chunk in person.get("chunks", []):
            scores = [NS(behaviour=s.get("behaviour"), smoothed_similarity=s.get("smoothed_similarity"))
                      for s in chunk.get("scores", [])]
            chunks.append(NS(chunk_id=chunk["chunk_id"], frame_indices=chunk.get("frame_indices", []), scores=scores))
        events = [NS(behaviour=e["behaviour"], start_timestamp=e["start_timestamp"],
                     end_timestamp=e["end_timestamp"], peak_similarity=e["peak_similarity"],
                     chunk_ids=e["chunk_ids"],
                     candidate_source=e.get("candidate_source", "prototype"),
                     arm_side=e.get("arm_side"), candidate_score=e.get("candidate_score"),
                     candidate_status=e.get("candidate_status", "MODEL_CANDIDATE"),
                     detector_name=e.get("detector_name"),
                     score_semantics=e.get("score_semantics", "uncalibrated_cosine_similarity"),
                     evidence=e.get("evidence", {})) for e in person.get("events", [])]
        people.append(NS(person_id=person["person_id"], chunks=chunks, events=events))
    return NS(schema_version=raw["schema_version"], source_schema_version=raw["source_schema_version"], persons=people)
