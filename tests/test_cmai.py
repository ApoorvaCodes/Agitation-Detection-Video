"""Synthetic contract/rule fixtures only; not trained detector evidence."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from cmai.bundle import (DetectorBundle, DetectorRule, load_bundle, research_override,
                         legacy_result_bundle, migrate_bank)
from cmai.contracts import CameraResult
from cmai.detection import detect, apply_rules
from cmai.results import build_camera_result, review_event, create_evidence, export_archive
from cmai.taxonomy import load_taxonomy, canonical_item
from person2.config import Person2Config
from person2.pipeline import process_perception
from person2.prototypes import build_prototypes
from test_person2 import observation, source

LABEL = "cmai_29_general_restlessness"


def candidates():
    data = source([observation(i / 5, x=1) for i in range(20)])
    for o in data.persons[0].observations:
        from person1.contracts import Landmark
        for joint in ("left_hip", "right_hip", "left_knee", "right_knee"):
            o.normalized_pose.landmarks[joint] = Landmark(x=.3, y=.5)
            o.quality.landmark_validity[joint] = True
        o.motion.feature_values["body_centroid.velocity.speed"] = 1
        o.quality.feature_validity["body_centroid.velocity.speed"] = True
    config = Person2Config(window_seconds=1, overlap=0, min_frames=2, video_weight=0)
    chunk = process_perception(data, config).persons[0].chunks[0]
    bank = build_prototypes([(LABEL, chunk.fused_embedding)], "synthetic-test-only")
    bundle = research_override(bank, config)
    return data, detect(data, bundle), bundle


def test_taxonomy_complete_strict_and_alias_migration():
    taxonomy = load_taxonomy()
    assert len(taxonomy.items) == 29
    assert taxonomy.item("cmai_04_cursing_verbal_aggression").camera_status == "not_camera_only"
    assert canonical_item("general_restlessness", allow_legacy=True) == LABEL
    with pytest.raises(ValueError, match="unknown"):
        canonical_item("general_restlessness")
    raw = taxonomy.model_dump()
    raw["items"][1] = raw["items"][0]
    with pytest.raises(ValueError):
        type(taxonomy).model_validate(raw)
    raw = taxonomy.model_dump()
    raw["taxonomy_version"] = "community-form"
    with pytest.raises(ValueError):
        type(taxonomy).model_validate(raw)


def test_default_bundle_abstains_and_reports_all_items():
    bundle = load_bundle()
    data = source()
    result = build_camera_result(data, detect(data, bundle), bundle)
    assert not result.events and len(result.availability) == 29
    assert all(a.status in {"unavailable", "not_assessed_by_camera"} for a in result.availability)
    assert all(i.status in {"no_prototypes", "low_quality"} for c in result.coverage for i in c.intervals)
    assert CameraResult.model_validate_json(result.model_dump_json()) == result


def test_release_cannot_be_enabled_without_real_evaluation():
    raw = load_bundle().metadata.model_dump()
    raw.update(mode="released", rules=[DetectorRule(item_id=LABEL, similarity_threshold=.8).model_dump()],
               prototype_file="bank.json", prototype_sha256="0"*64)
    with pytest.raises(ValueError, match="evaluation"):
        DetectorBundle.model_validate(raw)


def test_bundle_asset_hash_and_label_checks(tmp_path):
    _, _, bundle = candidates()
    raw = bundle.metadata.model_dump()
    bank_file = tmp_path / "bank.json"
    bank_file.write_text(bundle.bank.model_dump_json())
    raw["prototype_file"] = bank_file.name
    raw["prototype_sha256"] = sha256(bank_file.read_bytes()).hexdigest()
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(raw))
    assert load_bundle(path).bank == bundle.bank
    bank_file.write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(path)
    raw["rules"][0]["item_id"] = "cmai_07_hitting"
    with pytest.raises(ValueError, match="not enabled"):
        DetectorBundle.model_validate(raw)


def test_legacy_handoff_migration_keeps_source_and_reviewer_separate():
    data, p2, _ = candidates()
    for p in p2.persons:
        for c in p.chunks:
            for s in c.scores:
                s.behaviour = "general_restlessness"
        for e in p.events:
            e.behaviour = "general_restlessness"
    result = build_camera_result(data, p2, legacy_result_bundle(p2))
    assert result.events[0].cmai_item_id == LABEL
    assert result.events[0].source_label == "general_restlessness"
    assert result.events[0].quality.occlusion_fraction is None
    before = result.model_dump()
    updated = review_event(result, result.events[0].event_id, "confirmed", "tester", "Synthetic UI check")
    event = updated.events[0]
    assert event.status == "reviewed_confirmed" and event.model_status == "candidate"
    assert event.score == result.events[0].score
    assert result.model_dump() == before
    rejected = review_event(updated, event.event_id, "rejected", "tester")
    assert len(rejected.events[0].review_history) == 2
    assert rejected.events[0].status == "reviewed_rejected"


@pytest.mark.parametrize("field,value", [("cmai_item_id", "unknown"), ("taxonomy_version", "short-form"),
                                       ("status", "reviewed_confirmed"), ("score", 1.1)])
def test_invalid_event_contract_rejected(field, value):
    data, p2, bundle = candidates()
    raw = build_camera_result(data, p2, bundle).model_dump()
    raw["events"][0][field] = value
    with pytest.raises(ValueError):
        CameraResult.model_validate(raw)


def test_result_rejects_wrong_video_track_chunks_and_false_availability():
    data, p2, bundle = candidates()
    result = build_camera_result(data, p2, bundle)
    for key, value in [("video_id", "other-video"), ("person_id", "other-person"), ("chunk_ids", ["unknown"])]:
        raw = result.model_dump()
        raw["events"][0]["evidence"][key] = value
        with pytest.raises(ValueError):
            CameraResult.model_validate(raw)
    raw = result.model_dump()
    raw["availability"][3]["status"] = "available"
    with pytest.raises(ValueError, match="availability"):
        CameraResult.model_validate(raw)


def test_consecutive_evidence_stops_at_quality_gap_and_raw_score_failure():
    _, p2, bundle = candidates()
    assert len(p2.persons[0].events) == 1
    p2.persons[0].chunks[2].status = "low_quality"
    p2.persons[0].chunks[3].scores[0].similarity = .1
    p2.persons[0].chunks[3].scores[0].smoothed_similarity = .99
    apply_rules(p2, bundle.metadata.rules)
    assert len(p2.persons[0].events) == 1
    assert p2.persons[0].events[0].end_timestamp <= p2.persons[0].chunks[2].start_timestamp
    p2.persons[0].chunks[1].segment_id = 1
    apply_rules(p2, bundle.metadata.rules)
    assert not p2.persons[0].events


def test_missing_invalid_and_incompatible_evidence_abstains():
    data, p2, bundle = candidates()
    for o in data.persons[0].observations:
        o.quality.bbox_interpolated = True
    result = detect(data, bundle)
    assert not result.persons[0].events
    assert all(c.status == "low_quality" for c in result.persons[0].chunks)
    bundle.bank.prototypes[0].embedding.space = "incompatible-model-version"
    with pytest.raises(ValueError, match="incompatible embedding"):
        detect(data, bundle)


def test_clip_failures_remain_unknown_and_export_references_are_relative(tmp_path):
    data, p2, bundle = candidates()
    result = build_camera_result(data, p2, bundle)
    with patch("person3.clips.extract_candidate_clip", return_value=None):
        updated = create_evidence(result, "missing.mp4", tmp_path)
    assert updated.events[0].evidence.clip_status == "unavailable"
    assert updated.events[0].status == "candidate"
    assert export_archive(updated, tmp_path)
    raw = result.model_dump()
    raw["events"][0]["evidence"].update(clip_status="available", clip_path="../private.mp4", clip_start_timestamp=0)
    with pytest.raises(ValueError, match="relative"):
        CameraResult.model_validate(raw)


def test_schema_files_match():
    from cmai.experiments import EventManifest
    from cmai.taxonomy import CameraTaxonomy
    root = Path(__file__).resolve().parents[1]
    for name, cls in [("cmai.camera-result", CameraResult), ("cmai.detector-bundle", DetectorBundle),
                      ("cmai.taxonomy", CameraTaxonomy), ("cmai.annotations", EventManifest)]:
        assert json.loads((root / "docs" / f"{name}.schema.json").read_text()) == cls.model_json_schema()
