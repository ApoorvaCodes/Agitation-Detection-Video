"""Synthetic annotations exercise evaluation plumbing, not model performance."""
import json
from pathlib import Path
import pytest

from cmai.experiments import prepare_manifest, evaluate_events
from person1.io import save_perception
from person2.config import Person2Config
from person2.experiments import ExperimentManifest, run_experiments
from test_cmai import candidates, LABEL


def annotations(tmp_path):
    recordings = []
    for split in ("train", "validation", "test"):
        data, _, _ = candidates()
        data.video.video_id = split
        data.video.duration_seconds = 4
        path = tmp_path / f"{split}.json"
        save_perception(data, path)
        recordings.append(dict(perception=path.name, person_id="p1", subject_id=f"subject-{split}",
                               session_id=f"session-{split}", split=split, annotation_id=f"record-{split}",
                               reviewed_start=0, reviewed_end=4,
                               events=[dict(annotation_id=f"event-{split}", perception=path.name, person_id="p1",
                                            split=split, start_timestamp=0, end_timestamp=3, labels=[LABEL], certainty="observable")]))
    raw = dict(schema_version="cmai-event-annotations-1.0", taxonomy_version="cmai-long-form-camera-v1",
               behaviours=[LABEL], recordings=recordings,
               provenance=dict(name="Synthetic fixtures only", source="Unit tests", annotation_protocol="Synthetic event intervals",
                               annotation_version="unit-1", license_or_permission="Repository tests", label_agreement="No human agreement measured"))
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(raw))
    return path, raw


def test_event_annotations_to_disjoint_manifest_and_four_ablations(tmp_path):
    path, _ = annotations(tmp_path)
    config = Person2Config(window_seconds=1, overlap=0, video_weight=0)
    output = tmp_path / "prepared.json"
    converted = prepare_manifest(path, output, config)
    assert converted.schema_version == "2.0"
    assert converted.records[3].labels == []  # Explicitly reviewed final second.
    report = run_experiments(output, tmp_path / "experiment", config, thresholds=[.5,.8])
    assert report["status"] == "evaluated"
    assert all("events" in v["metrics"]["test"] for v in report["variants"].values())
    # Held-out event labels cannot influence threshold selection or prototypes.
    raw = converted.model_dump()
    for e in raw["event_annotations"]:
        if e["split"] == "test":
            e["end_timestamp"] = 2
    output.write_text(json.dumps(raw))
    changed = run_experiments(output, tmp_path / "changed", config, thresholds=[.5,.8])
    for name, v in report["variants"].items():
        assert v["selected_threshold"] == changed["variants"][name]["selected_threshold"]
        assert v["prototype_sha256"] == changed["variants"][name]["prototype_sha256"]


def test_uncertain_annotations_are_excluded_never_negative(tmp_path):
    path, raw = annotations(tmp_path)
    for recording in raw["recordings"]:
        recording["events"].append(dict(annotation_id="uncertain-"+recording["split"], perception=recording["perception"],
                                       person_id="p1", split=recording["split"], start_timestamp=3, end_timestamp=4,
                                       labels=[LABEL], certainty="uncertain"))
    path.write_text(json.dumps(raw))
    manifest = prepare_manifest(path, tmp_path / "prepared.json", Person2Config(window_seconds=1, overlap=0))
    assert len(manifest.records) == 9
    assert all(r.labels == [LABEL] for r in manifest.records)


@pytest.mark.parametrize("field", ["subject_id", "session_id"])
def test_event_dataset_leakage_rejected(tmp_path, field):
    path, raw = annotations(tmp_path)
    raw["recordings"][1][field] = raw["recordings"][0][field]
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="leakage"):
        prepare_manifest(path, tmp_path / "prepared.json", Person2Config(window_seconds=1, overlap=0))


def test_canonical_labels_camera_mode_and_schema_required(tmp_path):
    path, _ = annotations(tmp_path)
    manifest = prepare_manifest(path, tmp_path / "prepared.json", Person2Config(window_seconds=1, overlap=0)).model_dump()
    for value in ("general_restlessness", "cmai_07_hitting", "cmai_04_cursing_verbal_aggression"):
        modified = dict(manifest, behaviours=[value])
        with pytest.raises(ValueError):
            ExperimentManifest.model_validate(modified)
    manifest["schema_version"] = "1.0"
    with pytest.raises(ValueError):
        ExperimentManifest.model_validate(manifest)


def test_event_matching_counts_one_match_and_reports_timing(tmp_path):
    path, _ = annotations(tmp_path)
    manifest = prepare_manifest(path, tmp_path / "prepared.json", Person2Config(window_seconds=1, overlap=0))
    _, result, _ = candidates()
    result.persons[0].events[0].end_timestamp = 3
    metrics = evaluate_events({(tmp_path / "test.json").resolve():result}, manifest, tmp_path, "test")
    assert metrics["per_item"][LABEL]["tp"] == 1
    assert metrics["mean_absolute_onset_error_seconds"] == 0
    result.persons[0].events.append(result.persons[0].events[0].model_copy(deep=True))
    metrics = evaluate_events({(tmp_path / "test.json").resolve():result}, manifest, tmp_path, "test")
    assert metrics["per_item"][LABEL]["tp"] == 1 and metrics["per_item"][LABEL]["fp"] == 1


def test_annotation_chunk_configuration_cannot_silently_change(tmp_path):
    path, _ = annotations(tmp_path)
    output = tmp_path / "prepared.json"
    prepare_manifest(path, output, Person2Config(window_seconds=1, overlap=0))
    with pytest.raises(ValueError, match="configuration differs"):
        run_experiments(output, tmp_path / "wrong", Person2Config(window_seconds=2))


def test_research_bundle_packaging_and_release_criteria_fail_closed(tmp_path):
    from hashlib import sha256
    from cmai.bundle import write_research_bundle, load_bundle
    path, _ = annotations(tmp_path)
    config = Person2Config(window_seconds=1, overlap=0, video_weight=0)
    prepared = tmp_path / "prepared.json"
    prepare_manifest(path, prepared, config)
    run_experiments(prepared, tmp_path / "experiment", config)
    report_path = tmp_path / "experiment/report.json"
    output = tmp_path / "bundle.json"
    bundle = write_research_bundle(report_path, "pose_plus_motion", output)
    assert bundle.metadata.mode == "research"
    criteria = tmp_path / "criteria.json"
    criteria.write_text(json.dumps([dict(item_id=LABEL, min_precision=.1, min_recall=.1, min_event_f1=.1,
                                        min_coverage=.1, min_test_positive_events=1)]))
    raw = bundle.metadata.model_dump()
    raw.update(mode="released", release_evidence=dict(annotation_protocol="Synthetic test only", dataset_provenance="Synthetic test only",
               permissions="Unit fixtures", evaluation_report="experiment/report.json", evaluation_report_sha256=sha256(report_path.read_bytes()).hexdigest(),
               acceptance_criteria="criteria.json", acceptance_criteria_sha256=sha256(criteria.read_bytes()).hexdigest(),
               evidence_path_test="Unit fixture only", approved_by="Unit fixture only", approved_at="2026-01-01T00:00:00Z"))
    output.write_text(json.dumps(raw))
    assert load_bundle(output).metadata.mode == "released"  # Gate exercised; no shipped asset created.
    altered = json.loads(criteria.read_text())
    altered[0]["min_test_positive_events"] = 100
    criteria.write_text(json.dumps(altered))
    raw["release_evidence"]["acceptance_criteria_sha256"] = sha256(criteria.read_bytes()).hexdigest()
    output.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="insufficient held-out"):
        load_bundle(output)
