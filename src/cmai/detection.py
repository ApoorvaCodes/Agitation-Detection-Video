"""Behaviour-specific prototype rules on the existing Person 2 baseline."""
from dataclasses import asdict, replace

from person2.pipeline import aggregate_events, process_perception
from person2.embeddings import usable_pose
from cmai.bundle import EVIDENCE_REQUIREMENTS
import math


def has_evidence(observations, chunk, rule):
    joints, features = EVIDENCE_REQUIREMENTS[rule.item_id]
    frames = set(chunk.frame_indices)
    rows = [o for o in observations if o.frame_index in frames]
    good = [o for o in rows if all(usable_pose(o, j) for j in joints)
            and o.motion and any(o.quality.feature_validity.get(n, False)
                                 and o.motion.feature_values.get(n) is not None
                                 and math.isfinite(o.motion.feature_values[n]) for n in features)]
    return bool(rows) and len(good)/len(rows) >= rule.min_evidence_fraction


def apply_rules(result, rules, source=None):
    """Require consecutive measured support; quality/track barriers break runs."""
    for person in result.persons:
        events = [event for event in person.events if event.candidate_source == "motion_baseline"]
        for rule in rules:
            support, prior_segment, prior_end = [], None, None
            observations = {o.frame_index: o for p in source.persons if p.person_id == person.person_id
                            for o in p.observations} if source is not None else None
            history = None

            def finish(barrier=None):
                if len(support) >= rule.min_consecutive_chunks:
                    merged = aggregate_events(support, replace(result.configuration, min_event_seconds=0))
                    for event in merged:
                        if barrier is not None:
                            event.end_timestamp = min(event.end_timestamp, barrier)
                        if event.end_timestamp - event.start_timestamp >= rule.min_event_seconds:
                            events.append(event)
                support.clear()

            for chunk in person.chunks:
                score = next((s for s in chunk.scores if s.behaviour == rule.item_id), None)
                if observations is not None and score is not None:
                    if not has_evidence(list(observations.values()), chunk, rule):
                        score.similarity = score.smoothed_similarity = None
                        score.candidate = False
                if chunk.status != "scored" or prior_segment != chunk.segment_id or score is None or score.similarity is None:
                    history = None
                if score is not None and score.similarity is not None:
                    history = score.similarity if history is None else (result.configuration.smoothing_alpha * score.similarity
                              + (1-result.configuration.smoothing_alpha) * history)
                    score.smoothed_similarity = history
                # Raw and smoothed evidence must both meet the threshold. EMA
                # memory alone cannot count as a new supporting observation.
                valid = (chunk.status == "scored" and score is not None and score.similarity is not None
                         and score.smoothed_similarity is not None
                         and min(score.similarity, score.smoothed_similarity) >= rule.similarity_threshold)
                if score is not None:
                    score.candidate = valid
                if (not valid or prior_segment != chunk.segment_id
                        or (prior_end is not None and chunk.start_timestamp > prior_end)):
                    finish(chunk.start_timestamp)
                isolated = chunk.model_copy(deep=True)
                isolated.scores = [s for s in isolated.scores if s.behaviour == rule.item_id]
                for s in isolated.scores:
                    s.candidate = valid
                if valid:
                    support.append(isolated)
                prior_segment, prior_end = chunk.segment_id, chunk.end_timestamp
            finish()
        for chunk in person.chunks:
            if chunk.status == "scored" and not any(s.similarity is not None for s in chunk.scores):
                chunk.status = "insufficient_evidence"
        person.events = sorted(events, key=lambda e: (e.start_timestamp, e.behaviour, e.arm_side or ""))
    return result


def detect_with_assessments(source, bundle, interactions=None, video_path=None, recording_sha256=None, extracted_result=None):
    if bundle.metadata.mode == "legacy_review":
        raise ValueError("legacy review cannot run a detector without original model assets")
    video_encoder = None
    if interactions is not None:
        interactions.validate_source(source, recording_sha256)
    if extracted_result is not None:
        if extracted_result.video_id != source.video.video_id:
            raise ValueError("cached embeddings belong to another recording")
        cached_cfg = asdict(extracted_result.configuration)
        active_cfg = asdict(bundle.metadata.configuration)
        cached_cfg.pop("similarity_threshold")
        active_cfg.pop("similarity_threshold")
        if cached_cfg != active_cfg:
            raise ValueError("cached embeddings use incompatible extraction configuration")
        result = extracted_result.model_copy(deep=True)
        result.configuration = bundle.metadata.configuration
        for person in result.persons:
            for chunk in person.chunks:
                if chunk.status == "no_prototypes" and bundle.action_model:
                    chunk.status = "scored"
    elif bundle.metadata.video_encoder is not None:
        if video_path is None:
            raise ValueError("this action bundle requires the original source video")
        from cmai.video_actions import LocalR3DEncoder
        video_encoder = LocalR3DEncoder(video_path, bundle.checkpoint_path, bundle.metadata.video_encoder, interactions, source)
    if extracted_result is None:
        prototypes = None if bundle.metadata.detector_mode == "interaction_actions" else bundle.bank
        result = process_perception(source, bundle.metadata.configuration, prototypes,
                                pose_encoder=bundle.pose_encoder(), video_encoder=video_encoder)
    if bundle.metadata.detector_mode == "interaction_actions":
        from cmai.action_detection import apply_action_rules
        return apply_action_rules(result, source, bundle, interactions, recording_sha256)
    result = apply_rules(result, bundle.metadata.rules, source)
    from cmai.demo_physical_behaviour import DemoPhysicalBehaviourDetector
    demo_detector = DemoPhysicalBehaviourDetector()
    result = demo_detector.detect(result, source)
    print("DEMO DETECTOR STATUS", flush=True)
    print({"enabled": demo_detector.config.enabled,
           "config_path": "configs/demo_physical_behaviour.json",
           "config_loaded": True, "detector_instantiated": True,
           "tracks_received": len(source.persons),
           "observations_received": sum(len(p.observations) for p in source.persons),
           "windows_evaluated": sum(len(p.windows) for p in source.persons),
           "candidates_generated": sum(len(p.events) for p in result.persons)}, flush=True)
    return result, None


def detect(source, bundle, **kwargs):
    return detect_with_assessments(source, bundle, **kwargs)[0]
