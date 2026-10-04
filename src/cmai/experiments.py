"""Event annotations → reproducible chunk manifests and event timing metrics."""
import argparse
import json
from pathlib import Path
from typing import Literal
from pydantic import Field, model_validator

from cmai.taxonomy import VERSION, require_initial_items
from person1.io import load_perception
from person2.config import Person2Config
from person2.contracts import Contract
from person2.pipeline import process_perception


class EventAnnotation(Contract):
    annotation_id: str = Field(min_length=1)
    perception: str = Field(min_length=1)
    person_id: str = Field(min_length=1)
    split: Literal["train", "validation", "test"]
    start_timestamp: float = Field(ge=0)
    end_timestamp: float = Field(gt=0)
    labels: list[str] = Field(min_length=1)
    certainty: Literal["observable", "uncertain"]

    @model_validator(mode="after")
    def check_annotation(self):
        require_initial_items(self.labels)
        if len(set(self.labels)) != len(self.labels) or self.end_timestamp <= self.start_timestamp:
            raise ValueError("event labels must be unique and interval positive")
        return self


class ReviewedInterval(Contract):
    perception: str = Field(min_length=1)
    person_id: str = Field(min_length=1)
    split: Literal["train", "validation", "test"]
    start_timestamp: float = Field(ge=0)
    end_timestamp: float = Field(gt=0)

    @model_validator(mode="after")
    def check_interval(self):
        if self.end_timestamp <= self.start_timestamp:
            raise ValueError("reviewed interval must be positive")
        return self


class AnnotatedRecording(Contract):
    perception: str
    person_id: str
    subject_id: str
    session_id: str
    split: Literal["train", "validation", "test"]
    annotation_id: str
    # Explicitly reviewed coverage; outside it there is no negative label.
    reviewed_start: float = Field(ge=0)
    reviewed_end: float = Field(gt=0)
    events: list[EventAnnotation]

    @model_validator(mode="after")
    def check_scope(self):
        if self.reviewed_end <= self.reviewed_start:
            raise ValueError("reviewed interval must be positive")
        for e in self.events:
            if (e.perception, e.person_id, e.split) != (self.perception, self.person_id, self.split):
                raise ValueError("event must belong to annotated recording/person/split")
            if e.start_timestamp < self.reviewed_start or e.end_timestamp > self.reviewed_end:
                raise ValueError("event outside reviewed coverage")
        return self


class EventManifest(Contract):
    schema_version: Literal["cmai-event-annotations-1.0"] = "cmai-event-annotations-1.0"
    taxonomy_version: Literal["cmai-long-form-camera-v1"] = VERSION
    provenance: dict
    behaviours: list[str] = Field(min_length=1)
    recordings: list[AnnotatedRecording] = Field(min_length=1)

    @model_validator(mode="after")
    def check_labels(self):
        require_initial_items(self.behaviours)
        if len(set(self.behaviours)) != len(self.behaviours):
            raise ValueError("duplicate behaviours")
        for r in self.recordings:
            for e in r.events:
                if not set(e.labels) <= set(self.behaviours):
                    raise ValueError("event labels must belong to declared behaviours")
        return self


def overlap(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


def prepare_manifest(path, output, config=None):
    from person2.experiments import ExperimentManifest, validate_split_integrity
    path, output = Path(path), Path(output)
    config = config or Person2Config()
    manifest = EventManifest.model_validate_json(path.read_text())
    require_initial_items(manifest.behaviours)
    records, events = [], []
    for recording in manifest.recordings:
        source_path = (path.parent / recording.perception).resolve()
        source = load_perception(source_path)
        if source.video.duration_seconds is not None and recording.reviewed_end > source.video.duration_seconds:
            raise ValueError("reviewed interval exceeds recording duration")
        chunks = [c for p in process_perception(source, config).persons if p.person_id == recording.person_id for c in p.chunks]
        if not chunks:
            raise ValueError("annotated person has no chunks")
        for c in chunks:
            if c.start_timestamp < recording.reviewed_start or c.end_timestamp > recording.reviewed_end:
                continue
            touching = [e for e in recording.events if overlap(c.start_timestamp, c.end_timestamp, e.start_timestamp, e.end_timestamp) > 0]
            if any(e.certainty == "uncertain" for e in touching):
                continue  # Never turn uncertain source intervals into negative labels.
            labels = sorted({b for e in touching for b in e.labels})
            # Event-edge chunks have ambiguous onset support; retain only fully
            # contained positives or explicitly reviewed negatives for training.
            if touching and not all(e.start_timestamp <= c.start_timestamp and e.end_timestamp >= c.end_timestamp for e in touching):
                continue
            records.append(dict(perception=str(source_path), person_id=recording.person_id, chunk_id=c.chunk_id,
                                subject_id=recording.subject_id, session_id=recording.session_id,
                                annotation_id=f"{recording.annotation_id}:{c.chunk_id}", split=recording.split, labels=labels))
        events.extend({**e.model_dump(), "perception": str(source_path)} for e in recording.events)
    converted = ExperimentManifest(provenance=manifest.provenance, behaviours=manifest.behaviours,
                                   records=records, event_annotations=events, chunk_configuration=config,
                                   reviewed_intervals=[dict(perception=str((path.parent/r.perception).resolve()),
                                                           person_id=r.person_id, split=r.split,
                                                           start_timestamp=r.reviewed_start, end_timestamp=r.reviewed_end)
                                                       for r in manifest.recordings])
    validate_split_integrity(converted, output.parent)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(converted.model_dump_json(indent=2) + "\n")
    return converted


def evaluate_events(results, manifest, base, split, iou_threshold=.5):
    """One-to-one event matching; report timing only for matched candidates.

    Scoring is limited to the explicitly annotated chunk coverage. Prediction
    intervals touching uncertainty are excluded and counted, never negatives.
    """
    truth = [EventAnnotation.model_validate(e) for e in manifest.event_annotations if e["split"] == split]
    counts, timing = {}, []
    for label in manifest.behaviours:
        positives = [e for e in truth if e.certainty == "observable" and label in e.labels]
        used, tp, fp, excluded = set(), 0, 0, 0
        for path, result in results.items():
            for person in result.persons:
                records = [r for r in manifest.records if r.split == split and (base / r.perception).resolve() == path and r.person_id == person.person_id]
                chunk_ids = {r.chunk_id for r in records}
                spans = [c for c in person.chunks if c.chunk_id in chunk_ids]
                for event in person.events:
                    if event.behaviour != label:
                        continue
                    coverage = [r for r in manifest.reviewed_intervals if r["split"] == split
                                and (base / r["perception"]).resolve() == path and r["person_id"] == person.person_id]
                    fully_reviewed = any(r["start_timestamp"] <= event.start_timestamp and r["end_timestamp"] >= event.end_timestamp for r in coverage)
                    if not fully_reviewed:
                        excluded += 1
                        continue
                    if any(e.certainty == "uncertain" and (base / e.perception).resolve() == path and e.person_id == person.person_id
                           and overlap(event.start_timestamp, event.end_timestamp, e.start_timestamp, e.end_timestamp) for e in truth):
                        excluded += 1
                        continue
                    matches = []
                    for index, e in enumerate(positives):
                        if index in used or (base / e.perception).resolve() != path or e.person_id != person.person_id:
                            continue
                        inter = overlap(event.start_timestamp, event.end_timestamp, e.start_timestamp, e.end_timestamp)
                        union = event.end_timestamp-event.start_timestamp + e.end_timestamp-e.start_timestamp-inter
                        if inter / union >= iou_threshold:
                            matches.append((inter / union, index, e))
                    if matches:
                        iou, index, e = max(matches, key=lambda m: (m[0], -m[1]))
                        used.add(index); tp += 1
                        timing.append(dict(item_id=label, iou=iou, onset_error_seconds=event.start_timestamp-e.start_timestamp,
                                           offset_error_seconds=event.end_timestamp-e.end_timestamp))
                    else:
                        fp += 1
        fn = len(positives) - tp
        counts[label] = dict(tp=tp, fp=fp, fn=fn, excluded_uncertain_or_unreviewed=excluded,
                             precision=tp/(tp+fp) if tp+fp else None,
                             recall=tp/(tp+fn) if tp+fn else None,
                             f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)
    tp, fp, fn = (sum(c[k] for c in counts.values()) for k in ("tp", "fp", "fn"))
    return dict(iou_threshold=iou_threshold, per_item=counts, micro_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                matched_timing=timing, mean_absolute_onset_error_seconds=sum(abs(t["onset_error_seconds"]) for t in timing)/len(timing) if timing else None,
                mean_absolute_offset_error_seconds=sum(abs(t["offset_error_seconds"]) for t in timing)/len(timing) if timing else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    args = parser.parse_args()
    config = Person2Config(**json.loads(Path(args.config).read_text())) if args.config else None
    prepare_manifest(args.annotations, args.output, config)


if __name__ == "__main__":
    main()
