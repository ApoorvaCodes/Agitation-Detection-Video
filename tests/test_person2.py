import json
from dataclasses import replace

import numpy as np
import pytest

from person1.contracts import (BoundingBox, Landmark, MotionFeatures, ObservationQuality,
                               Person1VideoResult, PersonObservation, PoseData, TrackedPerson, VideoMetadata)
from person2.chunking import temporal_chunks
from person2.config import Person2Config
from person2.contracts import Embedding, Person2VideoResult
from person2.embeddings import RGBHistogramEncoder, motion_embedding, pose_embedding
from person2.io import load_result, save_result
from person2.pipeline import process_perception
from person2.prototypes import build_prototypes, cosine_similarity


def observation(t, index=None, good=True, x=1.0):
    return PersonObservation(timestamp=t, frame_index=round(t * 5) if index is None else index,
                             bbox=BoundingBox(x_min=0, y_min=0, x_max=1, y_max=1), detection_confidence=1,
                             normalized_pose=PoseData(landmarks={"left_wrist": Landmark(x=x, y=.5)}),
                             motion=MotionFeatures(feature_values={"left_wrist.velocity.speed": x}),
                             quality=ObservationQuality(detection_confidence=1, valid_landmarks=int(good),
                                                        pose_detected=good,
                                                        bbox_interpolated=not good,
                                                        landmark_validity={"left_wrist": good},
                                                        feature_validity={"left_wrist.velocity.speed": good}))


def source(observations=None):
    return Person1VideoResult(video=VideoMetadata(video_id="fixture", source_path="unused.avi", fps=5,
                                                   processed_fps=5, width=32, height=32,
                                                   feature_names=["left_wrist.velocity.speed"]),
                              persons=[TrackedPerson(person_id="p1", observations=observations if observations is not None
                                                     else [observation(i / 5) for i in range(10)])])


def test_half_open_windows_and_gap_segments():
    config = Person2Config(window_seconds=1, overlap=0)
    chunks = list(temporal_chunks([observation(t) for t in (0, .2, 1, 3, 3.2)], config, 5))
    assert [c.segment_id for c in chunks] == [0, 0, 1]
    assert [o.timestamp for o in chunks[0].observations] == [0, .2]
    assert chunks[-1].end == pytest.approx(3.4)


@pytest.mark.parametrize("times", [[0, 0], [.2, 0]])
def test_nonmonotonic_timestamps_rejected(times):
    with pytest.raises(ValueError, match="strictly increasing"):
        list(temporal_chunks([observation(t, i) for i, t in enumerate(times)], Person2Config(), 5))


@pytest.mark.parametrize("kwargs", [{"overlap": 1}, {"window_seconds": 0}, {"pose_weight": -1},
                                     {"smoothing_alpha": 0}, {"max_gap_seconds": float("nan")}])
def test_bad_config(kwargs):
    with pytest.raises(ValueError):
        Person2Config(**kwargs)


def test_missing_masks_are_not_zero_evidence():
    observations = [observation(0, good=False), observation(.2, good=False)]
    assert not any(pose_embedding(observations).valid)
    motion, _, summary = motion_embedding(observations, ["left_wrist.velocity.speed"])
    assert not any(motion.valid)
    assert all(v is None for v in summary.values())
    result = process_perception(source(observations))
    assert result.persons[0].chunks[0].status == "low_quality"
    assert result.persons[0].events == []


def test_pose_derivative_respects_invalid_sample():
    obs = [observation(0, x=1), observation(.2, good=False, x=2), observation(.4, x=3)]
    embedding = pose_embedding(obs)
    from person2.embeddings import POSE_NAMES
    assert not embedding.valid[POSE_NAMES.index("left_wrist.x.mean_velocity")]


def test_sparse_windows_quality_accounts_for_expected_samples():
    result = process_perception(source([observation(0), observation(.8), observation(1.6)]))
    assert result.persons[0].chunks[0].valid_fraction == pytest.approx(3 / 9)
    assert result.persons[0].chunks[0].status == "low_quality"


def test_exemplars_detection_events_and_json(tmp_path):
    perception = source()
    unlabelled = process_perception(perception)
    assert all(c.status == "no_prototypes" for c in unlabelled.persons[0].chunks)
    examples = [("general_restlessness", unlabelled.persons[0].chunks[0].fused_embedding)]
    bank = build_prototypes(examples, "test-v1")
    result = process_perception(perception, prototypes=bank)
    chunk = result.persons[0].chunks[0]
    assert chunk.scores[0].similarity == pytest.approx(1)
    assert chunk.scores[0].candidate
    assert len(result.persons[0].events) == 1
    assert result.persons[0].events[0].end_timestamp == pytest.approx(2)
    path = tmp_path / "result.json"
    save_result(result, path)
    assert load_result(path) == result
    assert json.loads(path.read_text())["score_semantics"] == "uncalibrated_cosine_similarity"
    with pytest.raises(ValueError):
        Person2VideoResult.model_validate({**result.model_dump(), "schema_version": "2.0"})


def test_space_mismatch_rejected():
    initial = process_perception(source())
    bank = build_prototypes([("pacing_aimless_wandering", initial.persons[0].chunks[0].fused_embedding)], "v1")
    with pytest.raises(ValueError, match="incompatible"):
        process_perception(source(), replace(Person2Config(), pose_weight=2), bank)
    with pytest.raises(ValueError, match="incompatible"):
        process_perception(source(), replace(Person2Config(), window_seconds=3), bank)


def test_cosine_shared_mask_zero_norm_and_centroid():
    a = Embedding(space="s", values=[1, 0], valid=[True, False])
    b = Embedding(space="s", values=[1, 2], valid=[True, True])
    assert cosine_similarity(a, b, .6) == (None, .5)
    assert cosine_similarity(a, b, .5) == (1, .5)
    zero = Embedding(space="s", values=[0, 0], valid=[True, True])
    assert cosine_similarity(zero, b)[0] is None
    bank = build_prototypes([("b", a), ("b", b)], "v1")
    assert bank.prototypes[0].example_count == 2
    with pytest.raises(ValueError):
        build_prototypes([("b", zero)], "v1")
    with pytest.raises(ValueError):
        Embedding(space="s", values=[float("nan")], valid=[True])


def test_temporal_smoothing_reset_after_gap_and_quality():
    config = Person2Config(window_seconds=.6, overlap=0, min_event_seconds=0)
    obs = [observation(t) for t in (0, .2, .4, .6, .8, 1, 3, 3.2, 3.4)]
    for o in obs[3:6]:
        o.quality.bbox_interpolated = True
    initial = process_perception(source(obs), config)
    bank = build_prototypes([("candidate", initial.persons[0].chunks[0].fused_embedding)], "v1")
    result = process_perception(source(obs), config, bank)
    chunks = result.persons[0].chunks
    assert chunks[1].status == "low_quality"
    assert chunks[-1].scores[0].smoothed_similarity == chunks[-1].scores[0].similarity
    assert len(result.persons[0].events) == 2
    assert result.persons[0].events[0].end_timestamp <= .6


def test_persons_empty_and_independent():
    assert process_perception(source([])).persons[0].chunks == []
    perception = source()
    perception.persons.append(TrackedPerson(person_id="p2", observations=[observation(5, x=9)]))
    result = process_perception(perception)
    assert result.persons[0].chunks[0].chunk_id == "p1:0"
    assert result.persons[1].chunks[0].chunk_id == "p2:0"
    assert result.persons[1].chunks[0].start_timestamp == 5


def test_local_video_histogram_and_provider(tmp_path):
    import cv2
    path = tmp_path / "fixture.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (32, 32))
    for _ in range(10):
        writer.write(np.full((32, 32, 3), [0, 0, 255], dtype=np.uint8))
    writer.release()
    encoder = RGBHistogramEncoder(path)
    result = process_perception(source(), video_encoder=encoder)
    embedding = result.persons[0].chunks[0].embeddings["video"]
    assert len(embedding.values) == 24 and all(embedding.valid)
    assert embedding.values[7] > .9
    with pytest.raises(ValueError, match="cannot open"):
        process_perception(source(), video_encoder=RGBHistogramEncoder(tmp_path / "missing.avi"))


def test_cli_prototype_workflow(tmp_path, monkeypatch):
    from person1.io import save_perception
    from person2.cli import main
    perception_path = tmp_path / "perception.json"
    output = tmp_path / "result.json"
    save_perception(source(), perception_path)
    monkeypatch.setattr("sys.argv", ["person2", "run", "--input", str(perception_path), "--output", str(output)])
    main()
    manifest = tmp_path / "labels.json"
    manifest.write_text(json.dumps([{"result": "result.json", "chunk_id": "p1:0", "behaviour": "cmai_29_general_restlessness", "split": "train"}]))
    bank = tmp_path / "bank.json"
    monkeypatch.setattr("sys.argv", ["person2", "build-prototypes", "--manifest", str(manifest),
                                     "--output", str(bank), "--version", "v1"])
    main()
    monkeypatch.setattr("sys.argv", ["person2", "run", "--input", str(perception_path), "--output", str(output),
                                     "--prototypes", str(bank)])
    main()
    assert load_result(output).persons[0].events


def test_smoothing_uses_previous_score():
    config = Person2Config(window_seconds=1, overlap=0, smoothing_alpha=.25, min_event_seconds=0)
    obs = [observation(i / 5, x=1 if i < 5 else -3) for i in range(10)]
    initial = process_perception(source(obs), config)
    bank = build_prototypes([("candidate", initial.persons[0].chunks[0].fused_embedding)], "v1")
    result = process_perception(source(obs), config, bank)
    first, second = [c.scores[0] for c in result.persons[0].chunks]
    assert second.similarity != pytest.approx(first.similarity)
    assert second.smoothed_similarity == pytest.approx(.25 * second.similarity + .75 * first.similarity)


def test_incompatible_prototypes_across_behaviours_rejected():
    with pytest.raises(ValueError, match="same embedding space"):
        build_prototypes([("a", Embedding(space="s1", values=[1], valid=[True])),
                          ("b", Embedding(space="s2", values=[1], valid=[True]))], "v1")


def test_schema_artifacts_match_models():
    from pathlib import Path
    from person2.contracts import PrototypeBank
    root = Path(__file__).resolve().parents[1]
    for filename, model in (("person2.schema.json", Person2VideoResult),
                            ("person2.prototypes.schema.json", PrototypeBank)):
        assert json.loads((root / "docs" / filename).read_text()) == model.model_json_schema()
