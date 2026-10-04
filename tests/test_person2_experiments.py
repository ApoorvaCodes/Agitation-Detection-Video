"""Synthetic algorithm fixtures only; not empirical model-selection results."""
import json
from pathlib import Path

import pytest

from person1.io import save_perception
from person2.config import Person2Config
from person2.contracts import ChunkResult, Embedding
from person2.experiments import (ExperimentManifest, LabelledChunk, evaluate, run_experiments,
                                validate_split_integrity)
from test_person2 import observation, source


def manifest_fixture(tmp_path):
    records = []
    for split in ("train", "validation", "test"):
        data = source([observation(i / 5, x=1 if i < 5 else -3 if i < 10 else 0) for i in range(15)])
        from person1.contracts import Landmark
        for o in data.persons[0].observations:
            for joint in ("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"):
                o.normalized_pose.landmarks[joint] = Landmark(x=.3, y=.5)
                o.quality.landmark_validity[joint] = True
            for name in ("bbox_center.displacement", "body_centroid.velocity.speed"):
                o.motion.feature_values[name] = .2
                o.quality.feature_validity[name] = True
        data.video.video_id = f"fixture-{split}"
        path = tmp_path / f"{split}.json"
        save_perception(data, path)
        for index, labels in enumerate((["cmai_01_pacing_aimless_wandering"], ["cmai_29_general_restlessness"], [])):
            records.append({"perception": path.name, "person_id": "p1", "chunk_id": f"p1:{index}",
                            "subject_id": f"subject-{split}", "session_id": f"session-{split}",
                            "annotation_id": f"annotation-{split}-{index}", "split": split, "labels": labels})
    manifest = {"schema_version": "2.0", "provenance": {"name": "synthetic unit fixtures only",
                 "source": "tests, not research data", "annotation_protocol": "explicit algorithm fixture labels",
                 "annotation_version": "test-1", "license_or_permission": "repository tests", "label_agreement": "Synthetic unit labels; no human agreement measured"},
                "behaviours": ["cmai_01_pacing_aimless_wandering", "cmai_29_general_restlessness"], "records": records}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def test_no_dataset_produces_pending_report_not_metrics(tmp_path):
    report = run_experiments(None, tmp_path / "pending")
    assert report["status"] == "pending_labelled_data"
    assert report["selection"] is None
    assert len(report["variants"]) == 4
    assert all(v["metrics"] is None for v in report["variants"].values())
    assert not list((tmp_path / "pending").glob("*.prototypes.json"))


def test_four_ablations_reproducible_and_train_only(tmp_path):
    path, manifest = manifest_fixture(tmp_path)
    config = Person2Config(window_seconds=1, overlap=0)
    report = run_experiments(path, tmp_path / "first", config, thresholds=[.5, .8])
    repeat = run_experiments(path, tmp_path / "second", config, thresholds=[.5, .8])
    assert report == repeat
    assert report["status"] == "evaluated"
    for name, variant in report["variants"].items():
        assert variant["threshold_selection_split"] == "validation"
        assert variant["metrics"]["test"]["annotated_chunks"] == 3
        bank = json.loads((tmp_path / "first" / variant["prototype_file"]).read_text())
        assert [p["example_count"] for p in bank["prototypes"]] == [1, 1]
        assert variant["encoder_metadata"]["encoders"]["pose"]["spaces"]
    # Test labels can change reported test metrics but cannot influence tuning or prototypes.
    for r in manifest["records"]:
        if r["split"] == "test":
            r["labels"] = []
    path.write_text(json.dumps(manifest))
    changed = run_experiments(path, tmp_path / "changed", config, thresholds=[.5, .8])
    for name in report["variants"]:
        assert report["variants"][name]["selected_threshold"] == changed["variants"][name]["selected_threshold"]
        assert report["variants"][name]["prototype_sha256"] == changed["variants"][name]["prototype_sha256"]


@pytest.mark.parametrize("field", ["subject_id", "session_id", "perception"])
def test_cross_split_leakage_rejected(tmp_path, field):
    _, manifest = manifest_fixture(tmp_path)
    manifest["records"][3][field] = manifest["records"][0][field]
    parsed = ExperimentManifest.model_validate(manifest)
    with pytest.raises(ValueError, match="leakage|duplicate|conflicting"):
        validate_split_integrity(parsed, tmp_path)


def test_copied_files_and_duplicate_annotations_rejected(tmp_path):
    _, manifest = manifest_fixture(tmp_path)
    (tmp_path / "copy.json").write_bytes((tmp_path / "train.json").read_bytes())
    manifest["records"][3]["perception"] = "copy.json"
    with pytest.raises(ValueError, match="leakage|duplicate|conflicting"):
        validate_split_integrity(ExperimentManifest.model_validate(manifest), tmp_path)
    manifest["records"][3] = dict(manifest["records"][0])
    with pytest.raises(ValueError, match="duplicate"):
        validate_split_integrity(ExperimentManifest.model_validate(manifest), tmp_path)


def test_labels_and_all_splits_are_required(tmp_path):
    _, manifest = manifest_fixture(tmp_path)
    del manifest["records"][0]["labels"]
    with pytest.raises(ValueError):
        ExperimentManifest.model_validate(manifest)


def test_missing_training_class_stays_pending(tmp_path):
    path, manifest = manifest_fixture(tmp_path)
    manifest["records"][1]["labels"] = []
    path.write_text(json.dumps(manifest))
    report = run_experiments(path, tmp_path / "report", Person2Config(window_seconds=1, overlap=0))
    assert report["status"] == "incomplete_evaluation"
    assert all(v["status"] == "pending_usable_training_examples" and v["metrics"] is None
               for v in report["variants"].values())


def test_abstention_metrics_count_missed_positives_without_false_normality():
    record = LabelledChunk(perception="unused", person_id="p", chunk_id="p:0", subject_id="s", session_id="t",
                          annotation_id="a", split="test", labels=["a"])
    embedding = Embedding(space="empty", values=[0], valid=[False])
    chunk = ChunkResult(chunk_id="p:0", segment_id=0, start_timestamp=0, end_timestamp=1, frame_indices=[],
                        valid_fraction=0, status="low_quality", embeddings={}, fused_embedding=embedding,
                        motion_features={}, scores=[])
    metrics, predictions = evaluate([(record, chunk)], ["a", "b"])
    assert metrics["micro"]["coverage"] == 0
    assert metrics["micro"]["fn"] == 1
    assert metrics["micro"]["tn"] == 0
    assert metrics["per_label"]["b"]["abstained_negative"] == 1
    assert predictions[0]["abstained"] == ["a", "b"]


def test_invalid_thresholds_and_cli_pending(tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        run_experiments(None, tmp_path, thresholds=[float("nan")])
    from person2.cli import main
    monkeypatch.setattr("sys.argv", ["person2", "experiment", "--output-dir", str(tmp_path / "cli")])
    main()
    assert json.loads((tmp_path / "cli" / "report.json").read_text())["status"] == "pending_labelled_data"


def test_manifest_schema_stays_current():
    root = Path(__file__).resolve().parents[1]
    assert json.loads((root / "docs" / "person2.experiments.schema.json").read_text()) == ExperimentManifest.model_json_schema()
