"""Reproducible, explicitly labelled, subject/session-disjoint ablations.

No dataset is downloaded or labelled here. Synthetic unit fixtures are never
used as empirical evidence. Threshold selection uses validation only.
"""
from dataclasses import asdict, replace
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
from typing import Literal

from pydantic import Field, model_validator

from person1.io import load_perception
from person2.config import Person2Config
from person2.contracts import Contract
from person2.embeddings import TemporalPoseEncoder, encoder_metadata
from person2.io import save_prototypes
from person2.pipeline import process_perception
from person2.prototypes import build_prototypes
from cmai.taxonomy import VERSION, require_initial_items
from cmai.bundle import DetectorRule
from cmai.detection import apply_rules, has_evidence


class DatasetProvenance(Contract):
    name: str = Field(min_length=1)
    source: str = Field(min_length=1)
    annotation_protocol: str = Field(min_length=1)
    annotation_version: str = Field(min_length=1)
    license_or_permission: str = Field(min_length=1)
    label_agreement: str = Field(min_length=1)


class LabelledChunk(Contract):
    perception: str = Field(min_length=1)
    person_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    subject_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    annotation_id: str = Field(min_length=1)
    split: Literal["train", "validation", "test"]
    # Required even when empty: [] explicitly denotes an annotated negative.
    labels: list[str]


class ExperimentManifest(Contract):
    schema_version: Literal["2.0"] = "2.0"
    taxonomy_version: Literal["cmai-long-form-camera-v1"] = VERSION
    detector_mode: Literal["pose_motion_camera"] = "pose_motion_camera"
    provenance: DatasetProvenance
    behaviours: list[str] = Field(min_length=1)
    records: list[LabelledChunk] = Field(min_length=1)
    event_annotations: list[dict] = Field(default_factory=list)
    reviewed_intervals: list[dict] = Field(default_factory=list)
    chunk_configuration: Person2Config | None = None

    @model_validator(mode="after")
    def validate_labels(self):
        require_initial_items(self.behaviours)
        if len(self.behaviours) != len(set(self.behaviours)) or any(not b.strip() for b in self.behaviours):
            raise ValueError("behaviours must be unique nonempty labels")
        for record in self.records:
            if len(record.labels) != len(set(record.labels)) or not set(record.labels) <= set(self.behaviours):
                raise ValueError("record labels must be unique and belong to behaviours")
        if {r.split for r in self.records} != {"train", "validation", "test"}:
            raise ValueError("manifest needs train, validation, and test records")
        if self.event_annotations:
            from cmai.experiments import EventAnnotation
            self.event_annotations = [EventAnnotation.model_validate(e).model_dump() for e in self.event_annotations]
            videos = {(r.perception, r.person_id, r.split) for r in self.records}
            for e in self.event_annotations:
                if (e["perception"], e["person_id"], e["split"]) not in videos:
                    raise ValueError("event annotation has no matching chunk records in its split")
                if not set(e["labels"]) <= set(self.behaviours):
                    raise ValueError("event label not declared in behaviours")
            if not self.reviewed_intervals:
                raise ValueError("event evaluation requires explicit reviewed recording intervals")
        if self.reviewed_intervals:
            from cmai.experiments import ReviewedInterval
            self.reviewed_intervals = [ReviewedInterval.model_validate(r).model_dump() for r in self.reviewed_intervals]
            videos = {(r.perception, r.person_id, r.split) for r in self.records}
            if any((r["perception"], r["person_id"], r["split"]) not in videos for r in self.reviewed_intervals):
                raise ValueError("reviewed interval has no matching recording records")
            if len({e["annotation_id"] for e in self.event_annotations}) != len(self.event_annotations):
                raise ValueError("event annotation IDs must be unique")
            for e in self.event_annotations:
                if not any((r["perception"], r["person_id"], r["split"]) == (e["perception"], e["person_id"], e["split"])
                           and r["start_timestamp"] <= e["start_timestamp"] and r["end_timestamp"] >= e["end_timestamp"]
                           for r in self.reviewed_intervals):
                    raise ValueError("event outside reviewed recording intervals")
        return self


def content_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def runtime_metadata():
    package = Path(__file__).parent
    return {"python": platform.python_version(), "platform": platform.platform(),
            "dependencies": {name: version(name) for name in ("numpy", "pydantic", "opencv-python", "pydantic-core",
                                                               "annotated-types", "typing-extensions", "typing-inspection")},
            "code_sha256": {f"{p.parent.name}/{p.name}": content_hash(p)
                            for p in sorted([*package.parent.glob("person[12]/*.py"), *package.parent.glob("cmai/*.py")])}}


def validate_split_integrity(manifest, base):
    """Global subjects, sessions, source videos and identical files cannot leak."""
    partition = {name: {} for name in ("subject", "session", "file", "video")}
    track_subjects, seen, hashes, sources, sessions = {}, set(), {}, {}, {}
    audit = []
    for record in manifest.records:
        path = (base / record.perception).resolve()
        if path not in sources:
            hashes[path] = content_hash(path)
            sources[path] = load_perception(path)
        video = sources[path].video.video_id
        if video in sessions and sessions[video] != record.session_id:
            raise ValueError("one source video cannot have conflicting session IDs")
        sessions[video] = record.session_id
        key = (hashes[path], record.person_id, record.chunk_id)
        if key in seen:
            raise ValueError("duplicate annotated chunk")
        seen.add(key)
        track = (hashes[path], record.person_id)
        if track in track_subjects and track_subjects[track] != record.subject_id:
            raise ValueError("one source track cannot have conflicting subject IDs")
        track_subjects[track] = record.subject_id
        for kind, value in (("subject", record.subject_id), ("session", record.session_id),
                            ("file", hashes[path]), ("video", video)):
            old = partition[kind].get(value)
            if old is not None and old != record.split:
                raise ValueError(f"{kind} leakage across splits: {value}")
            partition[kind][value] = record.split
        audit.append({**record.model_dump(), "perception_sha256": hashes[path], "video_id": video})
    return sources, audit


def chunk_for(result, record):
    matches = [c for p in result.persons if p.person_id == record.person_id
               for c in p.chunks if c.chunk_id == record.chunk_id]
    if len(matches) != 1:
        raise ValueError(f"annotated chunk does not exist: {record.perception} {record.chunk_id}")
    return matches[0]


def evaluate(rows, behaviours):
    """Chunk multilabel metrics; abstained positives count as missed evidence.

    Abstained negatives are not true negatives. Coverage makes missing evidence
    visible rather than inflating conditional accuracy. No probabilities/AUROC.
    """
    counts = {b: {k: 0 for k in ("tp", "fp", "fn", "tn", "abstained_positive", "abstained_negative")}
              for b in behaviours}
    predictions = []
    for record, chunk in rows:
        available = {s.behaviour: s for s in chunk.scores} if chunk.status == "scored" else {}
        predicted, abstained = [], []
        for b in behaviours:
            positive = b in record.labels
            score = available.get(b)
            c = counts[b]
            if score is None or score.similarity is None:
                c["abstained_positive" if positive else "abstained_negative"] += 1
                if positive:
                    c["fn"] += 1
                abstained.append(b)
            elif score.candidate:
                c["tp" if positive else "fp"] += 1
                predicted.append(b)
            else:
                c["fn" if positive else "tn"] += 1
        predictions.append({"perception": record.perception, "chunk_id": record.chunk_id,
                            "annotation_id": record.annotation_id, "truth": record.labels,
                            "candidates": predicted, "abstained": abstained, "status": chunk.status,
                            "scores": [s.model_dump() for s in chunk.scores]})

    def derived(c, total):
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        return {**c, "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else None,
                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                "coverage": 1 - (c["abstained_positive"] + c["abstained_negative"]) / total if total else None}

    per_label = {b: derived(c, len(rows)) for b, c in counts.items()}
    micro = derived({k: sum(c[k] for c in counts.values()) for k in next(iter(counts.values()))},
                    len(rows) * len(behaviours))
    f1s = [m["f1"] for m in per_label.values() if m["f1"] is not None]
    return {"annotated_chunks": len(rows), "per_label": per_label, "micro": micro,
            "macro_f1": sum(f1s) / len(f1s) if f1s else None,
            "macro_f1_labels": len(f1s)}, predictions


def experiment_variants(config, temporal_bins):
    return [
        ("pose_only", replace(config, pose_weight=1, motion_weight=0, video_weight=0), None),
        ("motion_only", replace(config, pose_weight=0, motion_weight=1, video_weight=0), None),
        ("pose_plus_motion", replace(config, pose_weight=1, motion_weight=1, video_weight=0), None),
        ("temporal_pose_plus_motion", replace(config, pose_weight=1, motion_weight=1, video_weight=0),
         TemporalPoseEncoder(config.window_seconds, temporal_bins)),
    ]


LIMITATIONS = [
    "No clinical validation, calibrated probability, or claim of a best model.",
    "Chunk and optional event timing metrics assess visible candidate support, not clinically validated behaviour.",
    "Overlapping chunks are correlated; bootstrap confidence intervals are not estimated.",
    "Subject/session IDs and annotation provenance are supplied by the dataset owner and must be audited.",
    "Prototypes and temporal bins are engineering baselines, not pretrained or trained deep action models.",
    "Model selection remains provisional until representative data and independent replication are available.",
]


def run_experiments(manifest_path, output_dir, config=None, thresholds=(.6, .7, .8, .9), temporal_bins=4):
    config = config or Person2Config()
    variants = experiment_variants(config, temporal_bins)
    thresholds = sorted(set(thresholds))
    if not thresholds or any(not -1 <= t <= 1 for t in thresholds):
        raise ValueError("threshold grid must contain finite values in [-1,1]")
    report = {"schema_version": "2.0", "taxonomy_version": VERSION,
              "status": "pending_labelled_data", "runtime": runtime_metadata(),
              "base_configuration": asdict(config), "threshold_grid": thresholds,
              "variants": {name: {"configuration": asdict(cfg), "pose_identity":
                                   (encoder.identity if encoder else {"model": "pose_stats", "version": "1"}),
                                   "metrics": None} for name, cfg, encoder in variants},
              "limitations": LIMITATIONS, "selection": None}
    output_dir = Path(output_dir)
    if manifest_path is None:
        report["reason"] = "No labelled manifest supplied. No training, evaluation, or model ranking performed."
    else:
        manifest_path = Path(manifest_path)
        manifest = ExperimentManifest.model_validate_json(manifest_path.read_text())
        if manifest.chunk_configuration is not None and asdict(manifest.chunk_configuration) != asdict(config):
            raise ValueError("configuration differs from annotation chunk preparation; prepare the manifest again")
        sources, audit = validate_split_integrity(manifest, manifest_path.parent)
        report.update({"status": "evaluated", "dataset_provenance": manifest.provenance.model_dump(),
                       "manifest_sha256": content_hash(manifest_path), "split_audit": audit,
                       "behaviours": manifest.behaviours, "event_annotations": manifest.event_annotations,
                       "reviewed_intervals": manifest.reviewed_intervals,
                       "annotation_chunk_configuration": asdict(manifest.chunk_configuration) if manifest.chunk_configuration else None})
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, cfg, encoder in variants:
            extracted = {p: process_perception(source, cfg, pose_encoder=encoder) for p, source in sources.items()}
            examples = []
            unusable = []
            # Labels in validation/test never contribute to prototype construction.
            for record in manifest.records:
                chunk = chunk_for(extracted[(manifest_path.parent / record.perception).resolve()], record)
                if record.split == "train":
                    if chunk.status == "low_quality" or not any(chunk.fused_embedding.valid):
                        unusable.append(record.annotation_id)
                    else:
                        source = sources[(manifest_path.parent / record.perception).resolve()]
                        person = next(p for p in source.persons if p.person_id == record.person_id)
                        for label in record.labels:
                            if has_evidence(person.observations, chunk, DetectorRule(item_id=label, similarity_threshold=cfg.similarity_threshold)):
                                examples.append((label, chunk.fused_embedding))
                            else:
                                unusable.append(f"{record.annotation_id}:{label}:missing_required_camera_evidence")
            if {label for label, _ in examples} != set(manifest.behaviours):
                report["variants"][name].update({"status": "pending_usable_training_examples",
                                                "unusable_training_annotations": unusable})
                continue
            bank = build_prototypes(examples, f"{manifest.provenance.annotation_version}:{name}")
            save_prototypes(bank, output_dir / f"{name}.prototypes.json")

            def infer(split, threshold):
                paths = {(manifest_path.parent / r.perception).resolve() for r in manifest.records if r.split == split}
                rules = [DetectorRule(item_id=b, similarity_threshold=threshold) for b in manifest.behaviours]
                results = {p: apply_rules(process_perception(sources[p], replace(cfg, similarity_threshold=threshold),
                                                 prototypes=bank, pose_encoder=encoder), rules, sources[p]) for p in sorted(paths)}
                rows = [(r, chunk_for(results[(manifest_path.parent / r.perception).resolve()], r))
                        for r in manifest.records if r.split == split]
                metrics, predictions = evaluate(rows, manifest.behaviours)
                if manifest.event_annotations:
                    from cmai.experiments import evaluate_events
                    metrics["events"] = evaluate_events(results, manifest, manifest_path.parent, split)
                return metrics, predictions

            validation = [(t, *infer("validation", t)) for t in thresholds]
            eligible = [(t, m, predictions) for t, m, predictions in validation
                        if m["micro"]["f1"] is not None and m["micro"]["coverage"] > 0]
            if not eligible:
                report["variants"][name].update({"status": "pending_usable_validation_evidence",
                                                "validation_grid": [{"threshold": t, "metrics": m} for t, m, _ in validation]})
                continue
            # Deterministic tie break favors the higher threshold; never inspect test to tune.
            metric_key = lambda m: m.get("events", {}).get("micro_f1", m["micro"]["f1"])
            eligible = [row for row in eligible if metric_key(row[1]) is not None]
            if not eligible:
                report["variants"][name].update(status="pending_usable_validation_events")
                continue
            chosen, val_metrics, val_predictions = max(eligible, key=lambda item: (metric_key(item[1]), item[0]))
            test_metrics, test_predictions = infer("test", chosen)
            first = next(iter(extracted.values()))
            report["variants"][name] = {"status": "evaluated", "configuration": asdict(replace(cfg, similarity_threshold=chosen)),
                                         "encoder_metadata": encoder_metadata(first, encoder),
                                         "selected_threshold": chosen, "threshold_selection_split": "validation",
                                         "detector_rules": [DetectorRule(item_id=b, similarity_threshold=chosen).model_dump() for b in manifest.behaviours],
                                         "validation_grid": [{"threshold": t, "metrics": m} for t, m, _ in validation],
                                         "metrics": {"validation": val_metrics, "test": test_metrics},
                                         "predictions": {"validation": val_predictions, "test": test_predictions},
                                         "prototype_file": f"{name}.prototypes.json",
                                         "prototype_sha256": content_hash(output_dir / f"{name}.prototypes.json"),
                                         "unusable_training_annotations": unusable}
        if any(v.get("status") != "evaluated" for v in report["variants"].values()):
            report["status"] = "incomplete_evaluation"
        report["selection"] = "Comparison only; no automatic claim that any representation is best."
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
