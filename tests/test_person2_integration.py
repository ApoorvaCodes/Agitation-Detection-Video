import json

import pytest

from person2.contracts import Embedding
from person2.embeddings import TemporalPoseEncoder, fuse, load_video_encoder
from person2.pipeline import process_perception
from person2.prototypes import build_prototypes
from person2.config import Person2Config
from test_person2 import observation, source


def test_disabled_modalities_cannot_pass_quality_gate():
    obs = [observation(i / 5) for i in range(5)]
    for o in obs:
        o.normalized_pose = None
        o.quality.pose_detected = False
    result = process_perception(source(obs), Person2Config(pose_weight=1, motion_weight=0))
    assert result.persons[0].chunks[0].status == "low_quality"
    assert result.persons[0].chunks[0].valid_fraction == 0
    for o in obs:
        o.quality.feature_validity["left_wrist.velocity.speed"] = False
        o.normalized_pose = observation(0).normalized_pose
        o.quality.pose_detected = True
    result = process_perception(source(obs), Person2Config(pose_weight=0, motion_weight=1))
    assert result.persons[0].chunks[0].status == "low_quality"


def test_nonfinite_and_absent_observations_preserve_abstention():
    obs = [observation(i / 5) for i in range(5)]
    for o in obs:
        o.normalized_pose.landmarks["left_wrist"].x = float("nan")
        o.motion.feature_values["left_wrist.velocity.speed"] = float("inf")
    chunk = process_perception(source(obs)).persons[0].chunks[0]
    assert chunk.status == "low_quality" and chunk.scores == []
    assert not any(chunk.fused_embedding.valid)
    assert all(v == 0 for v in chunk.fused_embedding.values)
    for o in obs:
        o.normalized_pose, o.motion = None, None
    assert process_perception(source(obs)).persons[0].events == []


def test_incompatible_bank_rejected_even_for_low_quality_chunk():
    bank = build_prototypes([("candidate", Embedding(space="incompatible", values=[1], valid=[True]))], "v1")
    with pytest.raises(ValueError, match="incompatible"):
        process_perception(source([observation(0, good=False)]), prototypes=bank)


def test_overlapping_candidate_is_clipped_at_abstention():
    config = Person2Config(window_seconds=2, overlap=.5, min_event_seconds=0)
    obs = [observation(i / 5, good=i < 7) for i in range(15)]
    initial = process_perception(source(obs), config)
    assert initial.persons[0].chunks[0].status == "no_prototypes"
    assert initial.persons[0].chunks[1].status == "low_quality"
    bank = build_prototypes([("candidate", initial.persons[0].chunks[0].fused_embedding)], "v1")
    result = process_perception(source(obs), config, bank)
    assert len(result.persons[0].events) == 1
    assert result.persons[0].events[0].end_timestamp == 1
    assert result.persons[0].events[0].chunk_ids == ["p1:0"]


def test_time_bins_use_actual_times_and_never_fill_missing_bins():
    encoder = TemporalPoseEncoder(window_seconds=1, bins=2)
    obs = [observation(0, index=0, x=1), observation(.1, index=1, x=3),
           observation(.6, index=2, x=9)]
    embedding = encoder.encode(obs)
    i = encoder.feature_names.index("left_wrist.x.time_bin_0")
    assert embedding.values[i:i + 2] == [2, 9]
    obs[-1].quality.landmark_validity["left_wrist"] = False
    embedding = encoder.encode(obs)
    assert embedding.valid[i:i + 2] == [True, False]
    assert embedding.values[i + 1] == 0
    with pytest.raises(ValueError, match="duration"):
        process_perception(source(), pose_encoder=encoder)


def test_temporal_encoder_space_changes_with_settings():
    assert TemporalPoseEncoder(1, 2).encode([observation(0)]).space != TemporalPoseEncoder(2, 2).encode([observation(0)]).space
    with pytest.raises(ValueError, match="increasing"):
        TemporalPoseEncoder().encode([observation(.2, index=1), observation(0, index=2)])


class DriftingEncoder:
    feature_names = ["feature"]
    def __init__(self):
        self.calls = 0

    def encode(self, observations):
        self.calls += 1
        return Embedding(space=f"model:{self.calls}", values=[1], valid=[True])


def test_video_encoder_cannot_drift_spaces_between_chunks():
    with pytest.raises(ValueError, match="changed embedding space"):
        process_perception(source(), video_encoder=DriftingEncoder())


def test_configured_video_identity_and_fusion_masks(tmp_path):
    spec = tmp_path / "encoder.json"
    spec.write_text(json.dumps({"factory": "person2.embeddings:RGBHistogramEncoder", "kwargs": {"video_path": "unused.avi"},
                                "identity": {"model": "histogram", "version": "1", "preprocessing": "rgb-crop"}}))
    encoder = load_video_encoder(spec)
    assert encoder.identity["model"] == "histogram"
    assert len(encoder.feature_names) == 24
    embedding = Embedding(space="s", values=[1, 99], valid=[True, False])
    assert fuse({"pose": embedding}, {"pose": 1}).values == [1, 0]
    assert fuse({"pose": embedding}, {"pose": 0}).values == [0, 0]


def test_legacy_contract_serialization_shape_is_preserved():
    result = process_perception(source())
    assert set(result.model_dump()) == {"schema_version", "source_schema_version", "video_id", "score_semantics",
                                        "configuration", "embedding_spaces", "prototype_version", "persons"}
    assert result.schema_version == "1.0"
    assert result.persons[0].chunks[0].embeddings["pose"].space.startswith("pose_stats_v1:")


def test_insufficient_shared_evidence_does_not_create_candidates():
    config = Person2Config(min_shared_fraction=1)
    initial = process_perception(source(), config)
    embedding = initial.persons[0].chunks[0].fused_embedding.model_copy(deep=True)
    # Prototype has an additional coordinate that this track cannot observe.
    coordinate = embedding.valid.index(False)
    embedding.valid[coordinate] = True
    embedding.values[coordinate] = 1
    bank = build_prototypes([("candidate", embedding)], "v1")
    result = process_perception(source(), config, bank)
    assert all(c.status == "insufficient_evidence" for c in result.persons[0].chunks)
    assert result.persons[0].events == []
    assert all(not s.candidate and s.similarity is None for c in result.persons[0].chunks for s in c.scores)


def test_cli_prototype_builder_rejects_heldout_labels(tmp_path, monkeypatch):
    from person2.cli import main
    manifest = tmp_path / "labels.json"
    manifest.write_text(json.dumps([{"split": "test", "behaviour": "candidate", "result": "not-read.json"}]))
    monkeypatch.setattr("sys.argv", ["person2", "build-prototypes", "--manifest", str(manifest),
                                     "--output", str(tmp_path / "bank.json"), "--version", "v1"])
    with pytest.raises(ValueError, match="training examples only"):
        main()


def test_cli_encoder_sidecar_and_temporal_configuration(tmp_path, monkeypatch):
    from person1.io import save_perception
    from person2.cli import main
    path = tmp_path / "source.json"
    save_perception(source(), path)
    output = tmp_path / "output.json"
    monkeypatch.setattr("sys.argv", ["person2", "run", "--input", str(path), "--output", str(output),
                                     "--pose-representation", "time-bins", "--temporal-bins", "3"])
    main()
    metadata = json.loads((tmp_path / "output.json.encoders.json").read_text())
    assert metadata["encoders"]["pose"]["identity"]["bins"] == 3
    assert metadata["encoders"]["pose"]["spaces"][0].startswith("pose_time_bins_v1:")
    assert json.loads(output.read_text())["schema_version"] == "1.0"
