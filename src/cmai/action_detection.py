"""Action evidence gates and bounded candidate support, reusing P2 embeddings."""
from dataclasses import replace
from typing import Literal
from pydantic import Field
import math

from cmai.action_model import score_action
from cmai.interactions import InteractionEvidence, ContactObservation
from person2.contracts import Contract, BehaviourScore
from person2.embeddings import usable_pose
from person2.pipeline import aggregate_events

JOINTS = {"cmai_07_hitting": ("shoulder", "elbow", "wrist"), "cmai_08_kicking": ("hip", "knee", "ankle")}
LIMBS = {"cmai_07_hitting": {"left_hand", "right_hand"}, "cmai_08_kicking": {"left_foot", "right_foot"}}


class ActionAssessment(Contract):
    person_id: str
    chunk_id: str
    item_id: str
    status: Literal["scored", "low_quality", "contact_unavailable", "contact_unclear", "incompatible_evidence"]
    reason: str
    contact_present: bool = False
    positive_similarity: float | None = Field(default=None, ge=-1, le=1)
    negative_similarity: float | None = Field(default=None, ge=-1, le=1)
    contrast_margin: float | None = Field(default=None, ge=-2, le=2)
    contacts: list[ContactObservation] = Field(default_factory=list)


class ActionEvidenceResult(Contract):
    schema_version: Literal["cmai-action-assessments-1.0"] = "cmai-action-assessments-1.0"
    video_id: str
    recording_sha256: str | None = None
    provider_identity: dict | None = None
    assessments: list[ActionAssessment]


def evidence_gate(source, person, chunk, rule, quality, interactions):
    frames = set(chunk.frame_indices)
    observations = [o for o in person.observations if o.frame_index in frames]
    points = JOINTS[rule.item_id]

    def visible(o, side):
        joints = [f"{side}_{joint}" for joint in points]
        raw = o.pose.landmarks if o.pose else {}
        return (o.detection_confidence >= quality.min_detection_confidence
                and all(usable_pose(o, j) and (j in raw and raw[j].visibility is not None
                                             and raw[j].visibility >= quality.min_landmark_visibility) for j in joints)
                and o.motion and any(o.quality.feature_validity.get(f"{j}.velocity.speed", False)
                                     and o.motion.feature_values.get(f"{j}.velocity.speed") is not None
                                     and math.isfinite(o.motion.feature_values[f"{j}.velocity.speed"]) for j in joints))

    usable = {o.frame_index:{side for side in ("left", "right") if visible(o, side)} for o in observations}
    if chunk.status == "low_quality" or not observations or sum(bool(s) for s in usable.values())/len(observations) < rule.min_evidence_fraction:
        return "low_quality", "Insufficient visible limb/motion or track quality.", False, []
    if interactions is None:
        if rule.item_id == "cmai_07_hitting":
            return "scored", "Independent contact is optional for the Hitting arm-motion candidate.", False, []
        return "contact_unavailable", "No independent target/contact evidence provider is configured.", False, []
    rows = [e for e in interactions.observations if e.person_id == person.person_id and e.frame_index in frames
            and e.limb in LIMBS[rule.item_id]]
    if rule.item_id == "cmai_07_hitting":
        people = {p.person_id: {o.frame_index:o for o in p.observations} for p in source.persons}
        contacts = [e for e in rows if e.contact == "observed" and e.target_visible and e.actor_limb_visible
                    and e.limb.split("_")[0] in usable.get(e.frame_index,set())
                    and (e.target_kind == "object" or e.target_id not in people
                         or (e.frame_index in people[e.target_id]
                             and not people[e.target_id][e.frame_index].quality.bbox_interpolated
                             and people[e.target_id][e.frame_index].detection_confidence >= quality.min_detection_confidence))]
        if contacts and len({(e.target_kind,e.target_id) for e in contacts}) == 1:
            return "scored", "Arm motion scored; independently reviewed contact is optional supporting evidence.", True, contacts
        return "scored", "Arm motion scored without usable contact annotations; contact is optional.", False, []
    if not rows:
        return "contact_unavailable", "No target/contact evidence for this action window.", False, []
    people = {p.person_id: {o.frame_index:o for o in p.observations} for p in source.persons}
    for e in rows:
        target = people.get(e.target_id, {}).get(e.frame_index) if e.target_kind in {"person", "self"} else None
        if (e.contact == "unclear" or not e.target_visible or not e.actor_limb_visible
                or e.limb.split("_")[0] not in usable.get(e.frame_index, set())
                or target is not None and (target.quality.bbox_interpolated or target.detection_confidence < quality.min_detection_confidence)):
            return "contact_unclear", "Contact/target is unclear, occluded or on a low-quality track.", False, rows
    contacts = [e for e in rows if e.contact == "observed"]
    if contacts:
        if len({(e.target_kind,e.target_id) for e in contacts}) != 1:
            return "contact_unclear", "Multiple conflicting targets in one window; review locally.", False, rows
        return "scored", "Visible actor limb and independently observed contact on a traceable target.", True, contacts
    absent_frames = {e.frame_index for e in rows if e.contact == "absent"}
    if len(absent_frames)/len(observations) < quality.min_absent_contact_fraction:
        return "contact_unavailable", "Contact absence is not observed across enough of the window.", False, rows
    return "scored", "Explicit target/contact review found no contact in this window.", False, rows


def apply_action_rules(result, source, bundle, interactions=None, recording_sha256=None):
    if interactions is not None:
        interactions.validate_source(source, recording_sha256)
    assessments = []
    people = {p.person_id:p for p in source.persons}
    classes = {c.item_id:c for c in bundle.action_model.classes} if bundle.action_model else {}
    for person in result.persons:
        events = [event for event in person.events if event.candidate_source == "motion_baseline"]
        for rule in bundle.metadata.rules:
            support, history, previous_segment, previous_end, target = [], None, None, None, None

            def finish(barrier=None):
                if len(support) >= rule.min_consecutive_chunks:
                    merged = aggregate_events(support, replace(result.configuration, min_event_seconds=0))
                    for e in merged:
                        if barrier is not None:
                            e.end_timestamp = min(e.end_timestamp, barrier)
                        if e.end_timestamp-e.start_timestamp >= rule.min_event_seconds:
                            events.append(e)
                support.clear()

            for chunk in person.chunks:
                score = next((s for s in chunk.scores if s.behaviour == rule.item_id), None)
                if score is None:
                    score = BehaviourScore(behaviour=rule.item_id, shared_fraction=0)
                    chunk.scores.append(score)
                status, reason, contact, rows = evidence_gate(source, people[person.person_id], chunk, rule,
                                                              bundle.metadata.action_quality, interactions)
                positive, margin, shared = score_action(chunk.fused_embedding, classes[rule.item_id], result.configuration.min_shared_fraction)
                video = chunk.embeddings.get("video")
                if bundle.metadata.video_encoder and (video is None or not all(video.valid)):
                    status, reason = "incompatible_evidence", "Required video frames/features are unavailable; no silent modality fallback."
                if status == "scored" and positive is None:
                    status, reason = "incompatible_evidence", "Insufficient shared valid embedding coordinates for both centroids."
                # Hitting's arm-motion baseline/action score does not require
                # target contact. Optional contact annotations therefore must
                # not split its temporal support when they change or go unclear.
                if rule.item_id == "cmai_07_hitting":
                    # Keep the last reviewed target through missing/unclear
                    # optional contact rows, but respect an observed target change.
                    current_target = (rows[0].target_kind, rows[0].target_id) if contact else target
                else:
                    current_target = (rows[0].target_kind, rows[0].target_id) if contact else None
                if chunk.segment_id != previous_segment or current_target != target or (previous_end is not None and chunk.start_timestamp > previous_end) or status != "scored":
                    history = None
                score.similarity = positive if status == "scored" else None
                score.shared_fraction = shared
                if score.similarity is not None:
                    chunk.status = "scored"
                    history = positive if history is None else result.configuration.smoothing_alpha*positive+(1-result.configuration.smoothing_alpha)*history
                score.smoothed_similarity = history if status == "scored" else None
                contact_ok = contact or rule.item_id == "cmai_07_hitting"
                score.candidate = bool(status == "scored" and contact_ok and min(positive, history) >= rule.similarity_threshold
                                       and margin >= rule.contrast_margin)
                if not score.candidate or previous_segment != chunk.segment_id or current_target != target or (previous_end is not None and chunk.start_timestamp > previous_end):
                    finish(chunk.start_timestamp)
                if score.candidate:
                    isolated = chunk.model_copy(deep=True)
                    isolated.scores = [s for s in isolated.scores if s.behaviour == rule.item_id]
                    support.append(isolated)
                assessments.append(ActionAssessment(person_id=person.person_id, chunk_id=chunk.chunk_id, item_id=rule.item_id,
                                    status=status, reason=reason, contact_present=contact,
                                    positive_similarity=positive if status == "scored" else None,
                                    negative_similarity=max(-1,min(1,positive-margin)) if status == "scored" else None,
                                    contrast_margin=margin if status == "scored" else None, contacts=rows))
                previous_segment, previous_end, target = chunk.segment_id, chunk.end_timestamp, current_target
            finish()
        for chunk in person.chunks:
            if chunk.status == "scored" and not any(s.similarity is not None for s in chunk.scores):
                chunk.status = "insufficient_evidence"
        baseline_events = [e for e in events if e.candidate_source == "motion_baseline"]
        rule_events = [e for e in events if e.candidate_source != "motion_baseline"
                       and not (e.behaviour == "cmai_07_hitting" and any(
                           b.start_timestamp < e.end_timestamp and e.start_timestamp < b.end_timestamp
                           for b in baseline_events))]
        person.events = sorted([*baseline_events,*rule_events],key=lambda e:(e.start_timestamp,e.behaviour,e.arm_side or ""))
    return result, ActionEvidenceResult(video_id=source.video.video_id, recording_sha256=recording_sha256,
                        provider_identity={"provider_id":interactions.provider_id,"provider_version":interactions.provider_version,
                                           "method":interactions.method,"annotation_protocol":interactions.annotation_protocol} if interactions else None,
                        assessments=assessments)
