"""Ground verifier selections in source observations and deduplicate events."""
from hashlib import sha1
from person3.contracts import ValidationResult


def result_from_verification(candidate, packet, verification, flags=(), verifier_error=None):
    by_id = {s.evidence_id: s for s in packet.segments}
    selected = list(dict.fromkeys(x for x in verification.evidence_segment_ids if x in by_id))
    invalid = set(verification.evidence_segment_ids) - set(by_id)
    status = verification.decision
    reason = verification.reason
    if invalid:
        status, reason = "insufficient_evidence", "Verifier returned unknown evidence IDs; result abstained."
    if status == "supported" and not selected:
        status, reason = "insufficient_evidence", "No valid source evidence IDs were selected."
    start = min((by_id[x].timestamp for x in selected), default=None) if status == "supported" else None
    end = max((by_id[x].timestamp for x in selected), default=None) if status == "supported" else None
    if start is not None and end is not None and end <= start:
        status, start, end = "insufficient_evidence", None, None
        reason = "Selected source evidence does not define a positive interval."
    if status == "supported":
        identity = f"{candidate.person_id}|{packet.behaviour}|{','.join(sorted(selected))}"
    else:
        identity = f"{candidate.person_id}|{packet.behaviour}|{candidate.candidate_id}|{status}"
    event_id = "evt_" + sha1(identity.encode()).hexdigest()[:12]
    return ValidationResult(event_id=event_id, person_id=candidate.person_id, candidate_id=candidate.candidate_id,
                           behaviour=packet.behaviour, candidate_score=candidate.candidate_score,
                           validation_status=status, reason=reason, start_timestamp=start, end_timestamp=end,
                           source_window_ids=candidate.source_window_ids,
                           source_observation_ids=candidate.source_observation_ids, selected_evidence_ids=selected,
                           quality_flags=list(flags), verifier_error=verifier_error)


def deduplicate_events(events, overlap_threshold=0.5):
    """Deduplicate same-person/same-behaviour events by evidence identity or IoU.

    Overlapping different labels are always retained. For same-label overlaps,
    keep the supported result with most selected IDs, then best candidate score.
    """
    ordered = sorted(events, key=lambda e: (e.person_id, e.behaviour, e.start_timestamp or -1, e.event_id))
    kept = []
    for event in ordered:
        if event.validation_status != "supported":
            duplicate_at = next((i for i, old in enumerate(kept)
                                 if old.validation_status == event.validation_status
                                 and old.person_id == event.person_id and old.behaviour == event.behaviour
                                 and old.candidate_id == event.candidate_id
                                 and old.selected_evidence_ids == event.selected_evidence_ids), None)
            if duplicate_at is None:
                kept.append(event)
            continue
        duplicate_at = None
        for i, old in enumerate(kept):
            if old.validation_status != "supported" or old.person_id != event.person_id or old.behaviour != event.behaviour:
                continue
            a, b = set(old.selected_evidence_ids), set(event.selected_evidence_ids)
            if a and a == b:
                duplicate_at = i; break
            intersection = max(0.0, min(old.end_timestamp, event.end_timestamp) - max(old.start_timestamp, event.start_timestamp))
            union = max(old.end_timestamp, event.end_timestamp) - min(old.start_timestamp, event.start_timestamp)
            if union and intersection / union >= overlap_threshold:
                duplicate_at = i; break
        if duplicate_at is None:
            kept.append(event)
        else:
            old = kept[duplicate_at]
            if (len(event.selected_evidence_ids), event.candidate_score) > (len(old.selected_evidence_ids), old.candidate_score):
                kept[duplicate_at] = event
    return sorted(kept, key=lambda e: (e.start_timestamp if e.start_timestamp is not None else float("inf"), e.person_id, e.behaviour, e.event_id))
