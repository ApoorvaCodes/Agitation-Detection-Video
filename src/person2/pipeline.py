from dataclasses import asdict
import math

import numpy as np

from person1.contracts import Person1VideoResult
from person2.chunking import temporal_chunks
from person2.config import Person2Config
from person2.contracts import (BehaviourEvent, BehaviourScore, ChunkResult, Person2VideoResult,
                               PersonResult, PrototypeBank)
from person2.embeddings import (VideoEncoder, StatsPoseEncoder, TemporalPoseEncoder, fuse,
                                motion_embedding, space_id, usable_pose)
from person1.features import KEY_JOINTS
from person2.prototypes import cosine_similarity
from person2.hitting import HittingConfig, analyze_hitting
from person2.movement_patterns import MovementPatternConfig, analyze_movement_patterns


def aggregate_events(chunks, config):
    """Merge consecutive candidate support intervals; barriers close events."""
    events, active = [], {}

    def finish(label, barrier=None):
        event = active.pop(label)
        if barrier is not None:
            event.end_timestamp = min(event.end_timestamp, barrier)
        if event.end_timestamp > event.start_timestamp and event.end_timestamp - event.start_timestamp >= config.min_event_seconds:
            events.append(event)

    previous_segment = None
    for chunk in chunks:
        if previous_segment != chunk.segment_id or chunk.status != "scored":
            for label in list(active):
                finish(label, chunk.start_timestamp)
        candidates = {s.behaviour: s for s in chunk.scores if s.candidate}
        for label in list(active):
            if label not in candidates or chunk.start_timestamp > active[label].end_timestamp:
                finish(label, chunk.start_timestamp)
        for label, score in candidates.items():
            if label not in active:
                active[label] = BehaviourEvent(behaviour=label, start_timestamp=chunk.start_timestamp,
                                               end_timestamp=chunk.end_timestamp,
                                               peak_similarity=score.smoothed_similarity,
                                               chunk_ids=[chunk.chunk_id])
            else:
                event = active[label]
                event.end_timestamp = max(event.end_timestamp, chunk.end_timestamp)
                event.peak_similarity = max(event.peak_similarity, score.smoothed_similarity)
                event.chunk_ids.append(chunk.chunk_id)
        previous_segment = chunk.segment_id
    for label in list(active):
        finish(label)
    return sorted(events, key=lambda e: (e.start_timestamp, e.behaviour))


def process_perception(source: Person1VideoResult, config: Person2Config | None = None,
                       prototypes: PrototypeBank | None = None,
                       video_encoder: VideoEncoder | None = None,
                       pose_encoder: VideoEncoder | None = None,
                       hitting_config: HittingConfig | None = None,
                       movement_config: MovementPatternConfig | None = None) -> Person2VideoResult:
    config = config or Person2Config()
    hitting_config = hitting_config or HittingConfig.load()
    movement_config = movement_config or MovementPatternConfig.load()
    pose_encoder = pose_encoder or StatsPoseEncoder()
    if isinstance(pose_encoder, TemporalPoseEncoder) and pose_encoder.window_seconds != config.window_seconds:
        raise ValueError("temporal encoder duration must match pipeline window_seconds")
    if config.pose_weight == config.motion_weight == 0:
        raise ValueError("this baseline requires an enabled pose or motion modality")
    if source.schema_version != "1.0":
        raise ValueError("unsupported Person 1 schema version")
    names = source.video.feature_names
    if len(names) != len(set(names)):
        raise ValueError("motion feature names must be unique")
    ids = [p.person_id for p in source.persons]
    if len(ids) != len(set(ids)):
        raise ValueError("person IDs must be unique")
    fps = source.video.processed_fps if source.video.processed_fps is not None else source.video.fps
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("sampling fps must be finite and positive")
    spaces = {"pose": list(pose_encoder.feature_names), "motion": [f"{n}.{s}" for n in names for s in ("mean", "std")] or ["unavailable"]}
    if video_encoder:
        spaces["video"] = list(video_encoder.feature_names)
    for coordinates in spaces.values():
        if (not coordinates or any(not isinstance(n, str) or not n.strip() for n in coordinates)
                or len(coordinates) != len(set(coordinates))):
            raise ValueError("encoder feature names must be nonempty and unique")
    known_spaces = {}
    persons = []
    weights = {"pose": config.pose_weight, "motion": config.motion_weight, "video": config.video_weight}
    for person in source.persons:
        chunks, history = [], {}
        previous_segment = None
        for index, chunk in enumerate(temporal_chunks(person.observations, config, fps)):
            observations = chunk.observations
            embeddings = {"pose": pose_encoder.encode(observations)}
            motion, _, summary = motion_embedding(observations, names)
            embeddings["motion"] = motion
            if video_encoder:
                embeddings["video"] = video_encoder.encode(observations)
            for name, embedding in embeddings.items():
                if not embedding.space.strip():
                    raise ValueError(f"{name} encoder requires a nonempty versioned embedding space")
                if len(embedding.values) != len(spaces[name]):
                    raise ValueError(f"{name} encoder feature ordering/dimension mismatch")
                if name in known_spaces and known_spaces[name] != embedding.space:
                    raise ValueError(f"{name} encoder changed embedding space within a run")
                known_spaces[name] = embedding.space
            fused = fuse(embeddings, weights)
            fused.space = space_id("chunk_fusion_v1", [fused.space], {"window_seconds": config.window_seconds})
            for prototype in prototypes.prototypes if prototypes else []:
                if prototype.embedding.space != fused.space or len(prototype.embedding.values) != len(fused.values):
                    raise ValueError("incompatible embedding spaces; rebuild prototypes with the same encoder/configuration")
            valid_frames = 0
            for o in observations:
                pose_ok = config.pose_weight > 0 and any(usable_pose(o, n) for n in KEY_JOINTS)
                motion_ok = bool(config.motion_weight > 0 and o.motion and any(o.quality.feature_validity.get(n, False)
                                                 and o.motion.feature_values.get(n) is not None
                                                 and np.isfinite(o.motion.feature_values[n]) for n in names))
                valid_frames += int(not o.quality.bbox_interpolated and (pose_ok or motion_ok))
            expected = max(len(observations), math.ceil((chunk.end - chunk.start) * fps - 1e-9))
            fraction = valid_frames / expected
            low_quality = len(observations) < config.min_frames or fraction < config.min_valid_fraction
            status = "low_quality" if low_quality else "no_prototypes" if not prototypes or not prototypes.prototypes else "scored"
            if previous_segment != chunk.segment_id or status != "scored":
                history.clear()
            scores = []
            if status == "scored":
                for prototype in prototypes.prototypes:
                    similarity, shared = cosine_similarity(fused, prototype.embedding, config.min_shared_fraction)
                    smooth = None
                    if similarity is not None:
                        old = history.get(prototype.behaviour, similarity)
                        smooth = config.smoothing_alpha * similarity + (1 - config.smoothing_alpha) * old
                        history[prototype.behaviour] = smooth
                    else:
                        history.pop(prototype.behaviour, None)
                    scores.append(BehaviourScore(behaviour=prototype.behaviour, similarity=similarity,
                                                 smoothed_similarity=smooth, shared_fraction=shared,
                                                 candidate=smooth is not None and smooth >= config.similarity_threshold))
                if not any(s.similarity is not None for s in scores):
                    status = "insufficient_evidence"
                    history.clear()
            chunks.append(ChunkResult(chunk_id=f"{person.person_id}:{index}", segment_id=chunk.segment_id,
                                      start_timestamp=chunk.start, end_timestamp=chunk.end,
                                      frame_indices=[o.frame_index for o in observations], valid_fraction=fraction,
                                      status=status, embeddings=embeddings, fused_embedding=fused,
                                      motion_features=summary, scores=scores))
            previous_segment = chunk.segment_id
        events = aggregate_events(chunks, config)
        if hitting_config.enabled:
            hitting_events, diagnostics = analyze_hitting(person, hitting_config)
            diagnostics["total_frames_processed"] = source.diagnostics.get("frames_processed", source.video.frame_count)
            diagnostics["total_source_video_frames"] = source.video.frame_count
            for key,value in source.diagnostics.items():
                diagnostics[f"p1_{key}"] = value
            for chunk in chunks:
                for key, value in diagnostics.items():
                    chunk.motion_features[f"hitting.{key}"] = value
                chunk.motion_features["hitting.baseline_enabled"] = 1.0
            for candidate in hitting_events:
                supporting = [chunk for chunk in chunks if chunk.end_timestamp > candidate["start_timestamp"]
                              and chunk.start_timestamp < candidate["end_timestamp"]]
                if not supporting:
                    continue
                matching = next((e for e in events if e.behaviour == candidate["behaviour"]
                                 and e.start_timestamp < candidate["end_timestamp"]
                                 and candidate["start_timestamp"] < e.end_timestamp), None)
                if matching:
                    matching.evidence["motion_baseline"] = candidate["evidence"]
                    matching.arm_side = candidate["arm_side"]
                    continue
                events.append(BehaviourEvent(behaviour=candidate["behaviour"],
                    start_timestamp=candidate["start_timestamp"],end_timestamp=candidate["end_timestamp"],
                    peak_similarity=candidate["candidate_score"],chunk_ids=[c.chunk_id for c in supporting],
                    candidate_source="motion_baseline",arm_side=candidate["arm_side"],
                    candidate_score=candidate["candidate_score"],evidence=candidate["evidence"]))
        movement_events, movement_diagnostics = analyze_movement_patterns(person, movement_config, sampling_fps=fps)
        for detector, diagnostic in movement_diagnostics.items():
            for key, value in diagnostic.items():
                if isinstance(value, (int, float, bool)):
                    for chunk in chunks:
                        chunk.motion_features[f"{detector}.{key}"] = float(value)
        for behaviour, start, end, score, evidence in movement_events:
            supporting = [c for c in chunks if c.end_timestamp > start and c.start_timestamp < end]
            if supporting:
                events.append(BehaviourEvent(behaviour=behaviour, start_timestamp=start, end_timestamp=end,
                    peak_similarity=score, chunk_ids=[c.chunk_id for c in supporting], candidate_source="motion_baseline",
                    candidate_score=score, evidence=evidence))
        persons.append(PersonResult(person_id=person.person_id, chunks=chunks,
                                    movement_diagnostics=movement_diagnostics,
                                    events=sorted(events,key=lambda e:(e.start_timestamp,e.behaviour,e.arm_side or ""))))
    return Person2VideoResult(video_id=source.video.video_id, configuration=asdict(config),
                             embedding_spaces=spaces, prototype_version=prototypes.version if prototypes else None,
                             persons=persons)
