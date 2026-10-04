"""Versioned detector asset loading and explicit release gates.

Research banks can produce candidates but never become shipped detectors merely
because their vectors parse. No trained assets are bundled with this project.
"""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal
from pydantic import Field, model_validator

from cmai.taxonomy import VERSION, ROOT, require_initial_items, require_action_items, canonical_item
from person2.config import Person2Config
from person2.contracts import Contract, PrototypeBank
from person2.embeddings import TemporalPoseEncoder


class DetectorRule(Contract):
    item_id: str
    similarity_threshold: float = Field(ge=-1, le=1)
    min_consecutive_chunks: int = Field(default=2, ge=2)
    min_event_seconds: float = Field(default=2, gt=0)
    min_evidence_fraction: float = Field(default=.6, gt=0, le=1)
    contrast_margin: float = Field(default=0, ge=0, le=2)


class ActionQuality(Contract):
    min_detection_confidence: float = Field(default=.6, ge=0, le=1)
    min_landmark_visibility: float = Field(default=.7, ge=0, le=1)
    min_absent_contact_fraction: float = Field(default=.8, gt=0, le=1)


class LocalVideoEncoder(Contract):
    model: Literal["torchvision_r3d_18_kinetics400_v1"]
    checkpoint_file: str = Field(min_length=1)
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source: Literal["https://download.pytorch.org/models/r3d_18-b3b3357e.pth"]
    clip_frames: Literal[16] = 16
    sample_fps: Literal[15] = 15


class ActionAsset(Contract):
    model: Literal["masked_two_centroid_classifier_v1"] = "masked_two_centroid_classifier_v1"
    file: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


EVIDENCE_REQUIREMENTS = {
    "cmai_01_pacing_aimless_wandering": (("left_hip", "right_hip", "left_ankle", "right_ankle"), ("bbox_center.displacement",)),
    "cmai_26_repetitious_mannerisms": (("left_shoulder", "right_shoulder", "left_wrist", "right_wrist"),
                                      ("left_wrist.velocity.speed", "right_wrist.velocity.speed")),
    "cmai_29_general_restlessness": (("left_hip", "right_hip", "left_knee", "right_knee"),
                                    ("body_centroid.velocity.speed", "left_knee.velocity.speed", "right_knee.velocity.speed")),
}


class ReleaseEvidence(Contract):
    annotation_protocol: str = Field(min_length=1)
    dataset_provenance: str = Field(min_length=1)
    permissions: str = Field(min_length=1)
    evaluation_report: str = Field(min_length=1)
    evaluation_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    acceptance_criteria: str = Field(min_length=1)
    acceptance_criteria_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_path_test: str = Field(min_length=1)
    approved_by: str = Field(min_length=1)
    approved_at: str = Field(min_length=1)


class AcceptanceCriteria(Contract):
    """Thresholds supplied by project reviewers, never invented by the code."""
    item_id: str
    min_precision: float = Field(ge=0, le=1)
    min_recall: float = Field(ge=0, le=1)
    min_event_f1: float = Field(ge=0, le=1)
    min_coverage: float = Field(ge=0, le=1)
    min_test_positive_events: int = Field(ge=1)
    min_test_negative_intervals: int = Field(default=1, ge=1)
    max_mean_absolute_timing_error_seconds: float | None = Field(default=None, ge=0)


class DetectorBundle(Contract):
    schema_version: Literal["1.0", "1.1"] = "1.0"
    taxonomy_version: Literal["cmai-long-form-camera-v1"] = VERSION
    detector_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    mode: Literal["unavailable", "research", "released", "legacy_review"]
    configuration: Person2Config
    pose_representation: Literal["stats", "time-bins"] = "stats"
    temporal_bins: int = Field(default=4, ge=2)
    prototype_file: str | None = None
    prototype_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    rules: list[DetectorRule] = Field(default_factory=list)
    release_evidence: ReleaseEvidence | None = None
    detector_mode: Literal["pose_motion_camera", "interaction_actions"] = "pose_motion_camera"
    declared_labels: list[str] = Field(default_factory=list)
    action_asset: ActionAsset | None = None
    action_quality: ActionQuality = Field(default_factory=ActionQuality)
    video_encoder: LocalVideoEncoder | None = None

    @model_validator(mode="after")
    def check_gate(self):
        labels = [r.item_id for r in self.rules]
        validate_labels = require_action_items if self.detector_mode == "interaction_actions" else require_initial_items
        validate_labels(labels)
        validate_labels(self.declared_labels)
        if self.detector_mode == "interaction_actions":
            if self.schema_version != "1.1":
                raise ValueError("interaction action bundles require schema 1.1")
            if len(set(self.declared_labels)) != len(self.declared_labels) or not set(labels) <= set(self.declared_labels):
                raise ValueError("action rules must use unique declared canonical labels")
            if self.prototype_file or self.prototype_sha256:
                raise ValueError("interaction bundles require positive and negative action assets, not positive-only prototypes")
            if self.mode != "unavailable" and (not self.action_asset or not labels):
                raise ValueError("enabled interaction bundle needs supervised action asset and rules")
        elif self.action_asset or self.video_encoder:
            raise ValueError("action/video assets require interaction_actions mode")
        if self.video_encoder and (self.configuration.video_weight <= 0
                or abs(self.configuration.window_seconds-16/15) > 1e-6):
            raise ValueError("R3D-18 requires an enabled video modality and 16/15-second windows")
        if len(labels) != len(set(labels)):
            raise ValueError("detector rules must have unique item IDs")
        if self.mode == "unavailable":
            if self.rules or self.prototype_file or self.prototype_sha256 or self.action_asset:
                raise ValueError("unavailable bundle cannot enable detectors")
        elif self.detector_mode == "pose_motion_camera" and self.mode != "legacy_review" and (not self.rules or not self.prototype_file or not self.prototype_sha256):
            raise ValueError("enabled bundle needs hashed prototypes and detector rules")
        if self.mode == "released" and self.release_evidence is None:
            raise ValueError("released detectors require evaluation, acceptance and approval evidence")
        return self


@dataclass(frozen=True)
class LoadedBundle:
    metadata: DetectorBundle
    bank: PrototypeBank | None
    bundle_sha256: str
    action_model: object | None = None
    checkpoint_path: Path | None = None

    def pose_encoder(self):
        b = self.metadata
        return TemporalPoseEncoder(b.configuration.window_seconds, b.temporal_bins) if b.pose_representation == "time-bins" else None


def migrate_bank(bank):
    """Only three explicit starter aliases migrate; unknown labels fail closed."""
    raw = bank.model_dump()
    for p in raw["prototypes"]:
        p["behaviour"] = canonical_item(p["behaviour"], allow_legacy=True)
    require_initial_items([p["behaviour"] for p in raw["prototypes"]])
    return PrototypeBank.model_validate(raw)


def load_bundle(path=None):
    import json
    path = Path(path or ROOT / "configs/cmai_detector_bundle.json")
    data = path.read_bytes()
    b = DetectorBundle.model_validate_json(data)
    if b.mode == "legacy_review":
        raise ValueError("legacy_review is an import status, not a runnable detector bundle")
    bank = None
    action_model = None
    checkpoint_path = None
    if b.action_asset:
        from cmai.action_model import ActionModel
        raw = (path.parent / b.action_asset.file).read_bytes()
        if sha256(raw).hexdigest() != b.action_asset.sha256:
            raise ValueError("action model asset checksum mismatch")
        action_model = ActionModel.model_validate_json(raw)
        if {c.item_id for c in action_model.classes} != {r.item_id for r in b.rules}:
            raise ValueError("action model classes must match bundle rules")
        bank = action_model.positive_bank()
    if b.video_encoder:
        checkpoint_path = (path.parent / b.video_encoder.checkpoint_file).resolve()
        if not checkpoint_path.is_file():
            raise ValueError("R3D-18 checkpoint missing; provide an explicit local checkpoint, no download is automatic")
        if sha256(checkpoint_path.read_bytes()).hexdigest() != b.video_encoder.checkpoint_sha256:
            raise ValueError("video checkpoint checksum mismatch")
    if b.prototype_file:
        asset = (path.parent / b.prototype_file).resolve()
        raw = asset.read_bytes()
        if sha256(raw).hexdigest() != b.prototype_sha256:
            raise ValueError("prototype asset checksum mismatch")
        bank = PrototypeBank.model_validate_json(raw)
        require_initial_items([p.behaviour for p in bank.prototypes])
        if {p.behaviour for p in bank.prototypes} != {r.item_id for r in b.rules}:
            raise ValueError("bundle rules must match prototype labels exactly")
    if b.mode == "released":
        evidence = b.release_evidence
        raw = (path.parent / evidence.evaluation_report).read_bytes()
        if sha256(raw).hexdigest() != evidence.evaluation_report_sha256:
            raise ValueError("evaluation report checksum mismatch")
        report = json.loads(raw)
        if report.get("status") != "evaluated" or report.get("taxonomy_version") != VERSION:
            raise ValueError("release requires a completed evaluation for this taxonomy")
        if not report.get("split_audit") or not report.get("dataset_provenance"):
            raise ValueError("release report requires provenance and disjoint split audit")
        matches = [v for v in report.get("variants", {}).values()
                   if (v.get("action_model_sha256") == b.action_asset.sha256 if b.action_asset else v.get("prototype_sha256") == b.prototype_sha256)
                   and v.get("configuration") == b.configuration.__dict__]
        if not matches or not matches[0].get("metrics", {}).get("test"):
            raise ValueError("release report does not evaluate this prototype/configuration")
        evaluated = matches[0]
        pose = evaluated.get("encoder_metadata", {}).get("encoders", {}).get("pose", {}).get("identity", {})
        expected_pose = "pose_time_bins" if b.pose_representation == "time-bins" else "pose_stats"
        if pose.get("model") != expected_pose or (b.pose_representation == "time-bins" and pose.get("bins") != b.temporal_bins):
            raise ValueError("release report evaluates a different pose encoder")
        for relative, digest in report.get("runtime", {}).get("code_sha256", {}).items():
            code = ROOT / "src" / relative
            if not code.is_file() or sha256(code.read_bytes()).hexdigest() != digest:
                raise ValueError("detector implementation differs from evaluated code")
        if not report.get("runtime", {}).get("code_sha256"):
            raise ValueError("release report lacks implementation hashes")
        if [DetectorRule.model_validate(r).model_dump() for r in evaluated.get("detector_rules", [])] != [r.model_dump() for r in b.rules]:
            raise ValueError("release report does not evaluate these detector rules")
        criteria_raw = (path.parent / evidence.acceptance_criteria).read_bytes()
        if sha256(criteria_raw).hexdigest() != evidence.acceptance_criteria_sha256:
            raise ValueError("acceptance criteria checksum mismatch")
        criteria = [AcceptanceCriteria.model_validate(c) for c in json.loads(criteria_raw)]
        if {c.item_id for c in criteria} != {r.item_id for r in b.rules} or len(criteria) != len(b.rules):
            raise ValueError("acceptance criteria must cover enabled items exactly")
        test = evaluated["metrics"]["test"]
        for c in criteria:
            chunk = test.get("per_label", {}).get(c.item_id, {})
            event = test.get("events", {}).get("per_item", {}).get(c.item_id, {})
            if b.detector_mode == "interaction_actions":
                chunk = event
            checks = [(chunk.get("precision"), c.min_precision), (chunk.get("recall"), c.min_recall),
                      (chunk.get("coverage"), c.min_coverage), (event.get("f1"), c.min_event_f1)]
            if any(actual is None or actual < minimum for actual, minimum in checks):
                raise ValueError(f"held-out metrics do not meet acceptance criteria: {c.item_id}")
            if b.detector_mode == "interaction_actions":
                if event.get("negative_intervals", 0) < c.min_test_negative_intervals:
                    raise ValueError("insufficient held-out negative intervals for action release")
                timing = event.get("mean_absolute_timing_error_seconds")
                if c.max_mean_absolute_timing_error_seconds is None or timing is None or timing > c.max_mean_absolute_timing_error_seconds:
                    raise ValueError("action release requires reviewer-specified held-out timing acceptance")
            if event.get("tp", 0) + event.get("fn", 0) < c.min_test_positive_events:
                raise ValueError("insufficient held-out positive events for release")
        if b.detector_mode == "interaction_actions":
            if evaluated.get("action_quality") != b.action_quality.model_dump() or evaluated.get("video_encoder") != (b.video_encoder.model_dump() if b.video_encoder else None):
                raise ValueError("release report evaluates different interaction quality/video settings")
    return LoadedBundle(b, bank, sha256(data).hexdigest(), action_model, checkpoint_path)


def research_override(bank, config):
    bank = migrate_bank(bank)
    metadata = DetectorBundle(detector_id="custom-labelled-prototypes", version=bank.version,
                              mode="research", configuration=config, prototype_file="uploaded",
                              prototype_sha256=sha256(bank.model_dump_json().encode()).hexdigest(),
                              rules=[DetectorRule(item_id=p.behaviour, similarity_threshold=config.similarity_threshold)
                                     for p in bank.prototypes])
    return LoadedBundle(metadata, bank, sha256(metadata.model_dump_json().encode()).hexdigest())


def legacy_result_bundle(result):
    """Explicit migration: original assets and evaluation are unknown."""
    labels = {canonical_item(s.behaviour, allow_legacy=True)
              for p in result.persons for c in p.chunks for s in c.scores}
    labels |= {canonical_item(e.behaviour, allow_legacy=True) for p in result.persons for e in p.events}
    require_initial_items(labels)
    metadata = DetectorBundle(detector_id="legacy-person2-result-import", version=result.prototype_version or "unknown",
                              mode="legacy_review", configuration=result.configuration,
                              rules=[DetectorRule(item_id=b, similarity_threshold=result.configuration.similarity_threshold)
                                     for b in sorted(labels)])
    return LoadedBundle(metadata, None, sha256(metadata.model_dump_json().encode()).hexdigest())


def write_research_bundle(report_path, variant, output):
    """Package an evaluated variant for research, without claiming release."""
    import json
    import os
    report_path, output = Path(report_path), Path(output)
    report = json.loads(report_path.read_text())
    if report.get("taxonomy_version") != VERSION:
        raise ValueError("report taxonomy version mismatch")
    chosen = report["variants"][variant]
    if chosen.get("status") != "evaluated":
        raise ValueError("variant has no completed evaluation")
    asset = (report_path.parent / chosen["prototype_file"]).resolve()
    if sha256(asset.read_bytes()).hexdigest() != chosen["prototype_sha256"]:
        raise ValueError("prototype report checksum mismatch")
    pose = chosen["encoder_metadata"]["encoders"]["pose"]["identity"]
    metadata = DetectorBundle(detector_id=f"pose-motion-prototypes:{variant}",
                              version=report["manifest_sha256"][:16], mode="research",
                              configuration=chosen["configuration"],
                              pose_representation="time-bins" if pose["model"] == "pose_time_bins" else "stats",
                              temporal_bins=pose.get("bins", 4),
                              prototype_file=os.path.relpath(asset, output.parent.resolve()),
                              prototype_sha256=chosen["prototype_sha256"], rules=chosen["detector_rules"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(metadata.model_dump_json(indent=2)+"\n")
    return load_bundle(output)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Package an explicitly chosen evaluated variant as a research bundle")
    parser.add_argument("--report", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    write_research_bundle(args.report, args.variant, args.output)


if __name__ == "__main__":
    main()
