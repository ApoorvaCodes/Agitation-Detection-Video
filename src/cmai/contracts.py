"""CMAI camera companion 1.0; legacy P1/P2 JSON remains unchanged."""
from typing import Literal
from pydantic import Field, model_validator

from cmai.bundle import DetectorBundle
from cmai.taxonomy import VERSION, load_taxonomy
from person2.contracts import Contract, BehaviourScore
from cmai.interactions import ContactObservation
from cmai.action_detection import ActionEvidenceResult, LIMBS
from cmai.taxonomy import ACTION_ITEMS, MOVEMENT_BASELINE_ITEMS

ReviewStatus = Literal["pending", "confirmed", "rejected", "uncertain"]


class ReviewDecision(Contract):
    decision: Literal["confirmed", "rejected", "uncertain"]
    reviewed_at: str = Field(min_length=1)
    reviewer: str = Field(min_length=1)
    note: str = ""


class CandidateEvidence(Contract):
    video_id: str = Field(min_length=1)
    person_id: str = Field(min_length=1)
    chunk_ids: list[str] = Field(min_length=1)
    frame_indices: list[int] = Field(min_length=1)
    clip_path: str | None = None
    clip_status: Literal["pending", "available", "unavailable"] = "pending"
    clip_start_timestamp: float | None = Field(default=None, ge=0)
    contacts: list[ContactObservation] = Field(default_factory=list)

    @model_validator(mode="after")
    def safe_references(self):
        from pathlib import PurePosixPath
        if self.clip_path:
            path = PurePosixPath(self.clip_path)
            if path.is_absolute() or ".." in path.parts or "\\" in self.clip_path:
                raise ValueError("clip_path must be a relative evidence reference")
        if self.clip_status == "available" and (not self.clip_path or self.clip_start_timestamp is None):
            raise ValueError("available clip requires its relative path and source offset")
        if len(set(self.chunk_ids)) != len(self.chunk_ids) or len(set(self.frame_indices)) != len(self.frame_indices):
            raise ValueError("evidence references must be unique")
        return self


class CandidateQuality(Contract):
    visibility: Literal["adequate", "limited", "unknown"] = "unknown"
    occlusion_fraction: float | None = Field(default=None, ge=0, le=1)
    min_valid_fraction: float = Field(ge=0, le=1)
    review_status: ReviewStatus = "pending"


class CameraEvent(Contract):
    event_id: str = Field(min_length=1)
    taxonomy_version: Literal["cmai-long-form-camera-v1"] = VERSION
    cmai_item_id: str
    # Model result remains immutable when a human reviews the event.
    model_status: Literal["candidate"] = "candidate"
    status: Literal["candidate", "reviewed_confirmed", "reviewed_rejected", "uncertain"] = "candidate"
    start_timestamp: float = Field(ge=0)
    end_timestamp: float = Field(gt=0)
    score: float = Field(ge=-1, le=1)
    score_semantics: Literal["uncalibrated_cosine_similarity", "heuristic_motion_score"] = "uncalibrated_cosine_similarity"
    source_label: str
    source_candidate_id: str
    evidence: CandidateEvidence
    quality: CandidateQuality
    review_history: list[ReviewDecision] = Field(default_factory=list)
    evidence_check: dict
    # Qwen/Groq is machine evidence review, never a human confirmation.
    machine_validation: dict | None = None

    @model_validator(mode="after")
    def check_event(self):
        item = load_taxonomy().item(self.cmai_item_id)
        if item.camera_status in {"not_camera_only", "out_of_initial_scope"}:
            raise ValueError("camera-ineligible items cannot be emitted as events")
        if self.end_timestamp <= self.start_timestamp:
            raise ValueError("event must have positive duration")
        review = self.review_history[-1].decision if self.review_history else "pending"
        expected = {"pending": "candidate", "confirmed": "reviewed_confirmed",
                    "rejected": "reviewed_rejected", "uncertain": "uncertain"}[review]
        if self.status != expected or self.quality.review_status != review:
            raise ValueError("event status must agree with separate reviewer history")
        return self


class ItemAvailability(Contract):
    cmai_item_id: str
    status: Literal["available", "research_only", "unavailable", "not_assessed_by_camera"]
    reason: str = Field(min_length=1)


class CoverageInterval(Contract):
    chunk_id: str
    segment_id: int = Field(ge=0)
    start_timestamp: float = Field(ge=0)
    end_timestamp: float = Field(gt=0)
    valid_fraction: float = Field(ge=0, le=1)
    status: Literal["scored", "low_quality", "no_prototypes", "insufficient_evidence"]
    scores: list[BehaviourScore]

    @model_validator(mode="after")
    def check_interval(self):
        if self.end_timestamp <= self.start_timestamp:
            raise ValueError("coverage interval must have positive duration")
        return self


class TrackCoverage(Contract):
    person_id: str
    observed_frame_count: int = Field(ge=0)
    first_timestamp: float | None = Field(default=None, ge=0)
    last_timestamp: float | None = Field(default=None, ge=0)
    # Overlapping intervals intentionally remain explicit. Do not sum durations.
    intervals: list[CoverageInterval]


class CameraResult(Contract):
    contract_version: Literal["cmai-camera-result-1.0", "cmai-camera-result-1.1"] = "cmai-camera-result-1.0"
    taxonomy_version: Literal["cmai-long-form-camera-v1"] = VERSION
    taxonomy_form: Literal["CMAI 29-item long form"] = "CMAI 29-item long form"
    taxonomy_edition: str
    taxonomy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    video_id: str
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    recording_duration_seconds: float | None = Field(default=None, gt=0)
    source_metadata: dict
    detector: DetectorBundle
    detector_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    encoder_metadata: dict
    availability: list[ItemAvailability]
    coverage: list[TrackCoverage]
    events: list[CameraEvent]
    limitations: list[str]
    action_assessments: ActionEvidenceResult | None = None

    @model_validator(mode="after")
    def consistent_result(self):
        taxonomy = load_taxonomy()
        if self.detector.detector_mode == "interaction_actions":
            if self.contract_version != "cmai-camera-result-1.1":
                raise ValueError("action results require companion contract 1.1")
            if self.action_assessments is None or self.action_assessments.video_id != self.video_id:
                raise ValueError("action results require matching evidence assessments")
            if self.action_assessments.recording_sha256 != self.source_sha256:
                raise ValueError("action assessments must match source recording checksum")
        ids = [a.cmai_item_id for a in self.availability]
        if set(ids) != {i.item_id for i in taxonomy.items} or len(ids) != 29:
            raise ValueError("result must report availability for all 29 items")
        rules = {r.item_id for r in self.detector.rules}
        for a in self.availability:
            item = taxonomy.item(a.cmai_item_id)
            if a.cmai_item_id in MOVEMENT_BASELINE_ITEMS:
                if a.status != "research_only":
                    raise ValueError("experimental movement baseline availability must be research-only")
                continue
            if a.cmai_item_id == "cmai_07_hitting" and a.cmai_item_id not in rules:
                if a.status not in {"research_only", "unavailable"}:
                    raise ValueError("Hitting baseline availability must be research-only or unavailable")
                continue
            expected = ("not_assessed_by_camera" if item.camera_status in {"not_camera_only", "out_of_initial_scope"}
                        else "unavailable" if a.cmai_item_id not in rules else
                        "available" if self.detector.mode == "released" else "research_only")
            if a.status != expected:
                raise ValueError("availability must agree with camera scope and detector release state")
        people = {c.person_id: {i.chunk_id: i for i in c.intervals} for c in self.coverage}
        if len(people) != len(self.coverage) or len({e.event_id for e in self.events}) != len(self.events):
            raise ValueError("duplicate person or event IDs")
        for track in self.coverage:
            if len({c.chunk_id for c in track.intervals}) != len(track.intervals):
                raise ValueError("duplicate chunk IDs in track coverage")
        for e in self.events:
            motion_baseline = (e.cmai_item_id in MOVEMENT_BASELINE_ITEMS or e.cmai_item_id == "cmai_07_hitting"
                               and e.evidence_check.get("candidate_source") == "motion_baseline")
            if e.cmai_item_id in MOVEMENT_BASELINE_ITEMS:
                expected_detector = ("pacing_trajectory_v1" if e.cmai_item_id == "cmai_01_pacing_aimless_wandering"
                                     else "restlessness_pose_motion_v1")
                features = e.evidence_check.get("motion_features", {})
                source_ids = features.get("source_observation_ids", []) if isinstance(features, dict) else []
                timestamps = features.get("observation_timestamps", []) if isinstance(features, dict) else []
                source_refs = {f"{e.evidence.person_id}:frame:{frame}" for frame in e.evidence.frame_indices}
                if (e.evidence_check.get("candidate_source") != "motion_baseline"
                        or features.get("detector") != expected_detector
                        or not isinstance(source_ids, list) or len(source_ids) < 2
                        or not isinstance(timestamps, list) or len(timestamps) < 2
                        or not all(isinstance(oid, str) and oid for oid in source_ids)
                        or not set(source_ids) <= source_refs
                        or any(not isinstance(t, (int, float)) or not e.start_timestamp <= t <= e.end_timestamp for t in timestamps)
                        or not features.get("acceptance_reasons")):
                    raise ValueError("movement events require source-grounded experimental detector evidence")
            optional_contact_hitting = e.cmai_item_id == "cmai_07_hitting"
            if e.cmai_item_id in ACTION_ITEMS and not motion_baseline and not optional_contact_hitting:
                contacts = e.evidence.contacts
                supported = {c.evidence_id:c for a in self.action_assessments.assessments
                             if a.person_id == e.evidence.person_id and a.item_id == e.cmai_item_id
                             and a.chunk_id in e.evidence.chunk_ids and a.status == "scored"
                             for c in a.contacts} if self.action_assessments else {}
                if any(c.limb not in LIMBS[e.cmai_item_id] or supported.get(c.evidence_id) != c for c in contacts):
                    raise ValueError("action contact must match the limb and scored assessments")
                if not contacts or any(c.person_id != e.evidence.person_id or c.contact != "observed"
                                       or not c.target_visible or not c.actor_limb_visible
                                       or c.frame_index not in e.evidence.frame_indices
                                       or not e.start_timestamp <= c.timestamp < e.end_timestamp for c in contacts):
                    raise ValueError("action candidate requires visible, source-linked observed contact")
                if len({(c.target_kind,c.target_id) for c in contacts}) != 1:
                    raise ValueError("action event cannot merge different targets")
            if e.cmai_item_id not in rules and not motion_baseline:
                raise ValueError("event item has no enabled detector")
            if e.evidence.video_id != self.video_id or e.evidence.person_id not in people:
                raise ValueError("event evidence refers to another video/person")
            if not set(e.evidence.chunk_ids) <= set(people[e.evidence.person_id]):
                raise ValueError("event evidence refers to unknown chunks")
            chunks = [people[e.evidence.person_id][cid] for cid in e.evidence.chunk_ids]
            if any(c.status != "scored" for c in chunks) and not motion_baseline:
                raise ValueError("event cannot claim abstained chunks as supporting evidence")
            if len({c.segment_id for c in chunks}) != 1:
                raise ValueError("event cannot bridge a track gap")
            if e.start_timestamp < min(c.start_timestamp for c in chunks) or e.end_timestamp > max(c.end_timestamp for c in chunks):
                raise ValueError("event exceeds its supporting evidence intervals")
            if self.recording_duration_seconds is not None and e.end_timestamp > self.recording_duration_seconds + 1e-6:
                raise ValueError("event exceeds source recording")
        return self
