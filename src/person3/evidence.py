"""Build small, traceable evidence packets from P1 observations and P2 events."""
from person3.contracts import CandidateBehaviour, EvidencePacket, EvidenceSegment
from person3.taxonomy import normalize_behaviour
from cmai.taxonomy import canonical_item


def observation_id(person_id: str, frame_index: int) -> str:
    return f"{person_id}:frame:{frame_index}"


def build_evidence_packet(candidate: CandidateBehaviour, person) -> EvidencePacket:
    observations = {observation_id(person.person_id, o.frame_index): o for o in person.observations}
    selected = set(candidate.source_observation_ids)
    if not selected and candidate.source_window_ids:
        # P2 chunk frame_indices are carried in the candidate evidence adapter.
        selected = set(candidate.evidence.get("source_observation_ids", []))
    rows = []
    for oid in candidate.source_observation_ids or sorted(selected):
        o = observations.get(oid)
        if o is None or not candidate.start_timestamp <= o.timestamp <= candidate.end_timestamp:
            continue
        features = o.motion.feature_values if o.motion else {}
        relevant = {k: v for k, v in features.items() if k.endswith(("velocity.speed", "acceleration.magnitude", "displacement.magnitude", ".jerk"))}
        pose = {}
        if o.normalized_pose:
            for name, point in o.normalized_pose.landmarks.items():
                if name in {"left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"}:
                    pose[name] = {"x": point.x, "y": point.y, "z": point.z}
        quality = o.quality.model_dump(mode="json", exclude_none=True)
        quality["contact_evidence"] = [e for e in candidate.evidence.get("contacts", []) if e["frame_index"] == o.frame_index]
        quality["candidate_source"] = ("motion_baseline" if candidate.evidence.get("motion_baseline")
                                        else candidate.evidence.get("candidate_source", "prototype"))
        movement = candidate.evidence.get("motion_baseline", {})
        if isinstance(movement, dict) and movement.get("detector") in {"pacing_trajectory_v1", "restlessness_pose_motion_v1"}:
            quality["movement_pattern"] = movement
        if candidate.evidence.get("arm_side"):
            quality["arm_side"] = candidate.evidence["arm_side"]
        hitting_row = next((item for item in candidate.evidence.get("motion_baseline", {}).get("observation_features", [])
                            if item.get("frame_index") == o.frame_index), None)
        if hitting_row:
            quality["hitting_features"] = hitting_row
        rows.append(EvidenceSegment(evidence_id=oid, timestamp=o.timestamp, frame_index=o.frame_index,
                                    motion_features=relevant, pose=pose, quality_flags=quality))
    # Remote verification receives the strict camera taxonomy ID.  Legacy P2
    # labels remain readable locally, but are migrated at this boundary.
    try:
        label = canonical_item(candidate.behaviour, allow_legacy=True)
    except ValueError:
        # Preserve the established Person 3 display-label contract for
        # legacy/non-camera candidates; CMAI IDs take the strict branch above.
        taxonomy_item = normalize_behaviour(candidate.behaviour)
        label = taxonomy_item.label if taxonomy_item else candidate.behaviour
    return EvidencePacket(person_id=candidate.person_id, candidate_id=candidate.candidate_id,
                          behaviour=label, candidate_score=candidate.candidate_score,
                          candidate_start=candidate.start_timestamp, candidate_end=candidate.end_timestamp,
                          source_window_ids=candidate.source_window_ids, segments=rows)


def candidates_from_p2(p2_result, p1_result, action_assessments=None) -> list[CandidateBehaviour]:
    """Adapt the existing Person2VideoResult schema 1.0 without mutating it."""
    if p2_result.schema_version != "1.0" or p2_result.source_schema_version != p1_result.schema_version:
        raise ValueError("unsupported P1/P2 schema version")
    source_people = {p.person_id: p for p in p1_result.persons}
    output = []
    for p2_person in p2_result.persons:
        if p2_person.person_id not in source_people:
            continue
        chunk_map = {chunk.chunk_id: chunk for chunk in p2_person.chunks}
        observations = source_people[p2_person.person_id].observations
        for index, event in enumerate(p2_person.events):
            chunks = [chunk_map[cid] for cid in event.chunk_ids if cid in chunk_map]
            frame_ids = {frame for chunk in chunks for frame in chunk.frame_indices}
            rows = [o for o in observations if o.frame_index in frame_ids
                    and event.start_timestamp <= o.timestamp <= event.end_timestamp]
            score = max((s.smoothed_similarity for chunk in chunks for s in chunk.scores
                         if s.behaviour == event.behaviour and s.smoothed_similarity is not None), default=event.peak_similarity)
            candidate = CandidateBehaviour(
                candidate_id=f"{p2_person.person_id}:{event.behaviour}:{index:04d}", person_id=p2_person.person_id,
                behaviour=event.behaviour, candidate_score=event.candidate_score if event.candidate_score is not None else score, start_timestamp=event.start_timestamp,
                end_timestamp=event.end_timestamp, source_window_ids=list(event.chunk_ids),
                source_observation_ids=[observation_id(p2_person.person_id, o.frame_index) for o in rows],
                evidence={"p2_peak_similarity": event.peak_similarity,
                          "candidate_source": event.candidate_source,"arm_side": event.arm_side,
                          "motion_baseline": event.evidence.get("motion_baseline",event.evidence)},
            )
            if action_assessments is not None:
                contacts = {e.evidence_id:e.model_dump(mode="json") for a in action_assessments.assessments
                            if a.person_id == p2_person.person_id and a.item_id == event.behaviour
                            and a.chunk_id in event.chunk_ids and a.status == "scored"
                            for e in a.contacts if e.contact == "observed" and event.start_timestamp <= e.timestamp < event.end_timestamp}
                candidate.evidence["contacts"] = list(contacts.values())
                candidate.evidence["score_semantics"] = "uncalibrated_cosine_similarity"
            output.append(candidate)
    return output
