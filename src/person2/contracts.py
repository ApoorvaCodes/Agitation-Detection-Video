"""Versioned research evidence handoff; scores are not clinical probabilities."""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from person2.config import Person2Config


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Embedding(Contract):
    space: str
    values: list[float]
    valid: list[bool]

    @model_validator(mode="after")
    def check_shape(self):
        if not self.values or len(self.values) != len(self.valid):
            raise ValueError("embedding values and mask must have equal nonzero length")
        return self


class BehaviourScore(Contract):
    behaviour: str
    similarity: float | None = Field(default=None, ge=-1, le=1)
    smoothed_similarity: float | None = Field(default=None, ge=-1, le=1)
    shared_fraction: float = Field(ge=0, le=1)
    candidate: bool = False


class ChunkResult(Contract):
    chunk_id: str
    segment_id: int
    start_timestamp: float = Field(ge=0)
    end_timestamp: float = Field(ge=0)
    frame_indices: list[int]
    valid_fraction: float = Field(ge=0, le=1)
    status: Literal["scored", "low_quality", "no_prototypes", "insufficient_evidence"]
    embeddings: dict[str, Embedding]
    fused_embedding: Embedding
    motion_features: dict[str, float | None]
    scores: list[BehaviourScore]

    @model_validator(mode="after")
    def check_interval(self):
        if self.end_timestamp <= self.start_timestamp:
            raise ValueError("chunk interval must have positive duration")
        return self


class BehaviourEvent(Contract):
    behaviour: str
    start_timestamp: float = Field(ge=0)
    end_timestamp: float = Field(ge=0)
    peak_similarity: float = Field(ge=-1, le=1)
    chunk_ids: list[str]
    candidate_source: Literal["prototype", "motion_baseline"] = "prototype"
    arm_side: Literal["left", "right"] | None = None
    candidate_score: float | None = Field(default=None, ge=0, le=1)
    evidence: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_interval(self):
        if self.end_timestamp <= self.start_timestamp or not self.chunk_ids:
            raise ValueError("event must have positive duration and supporting chunk IDs")
        if self.candidate_source == "motion_baseline":
            allowed = {"cmai_07_hitting", "cmai_01_pacing_aimless_wandering", "cmai_29_general_restlessness"}
            if self.behaviour not in allowed or self.candidate_score is None:
                raise ValueError("motion-baseline event requires an enabled experimental behaviour and score")
            if self.behaviour == "cmai_07_hitting" and not self.arm_side:
                raise ValueError("Hitting motion-baseline events require an arm side")
            movement_detectors = {
                "cmai_01_pacing_aimless_wandering": "pacing_trajectory_v1",
                "cmai_29_general_restlessness": "restlessness_pose_motion_v1",
            }
            if self.behaviour in movement_detectors:
                evidence = self.evidence
                ids, timestamps = evidence.get("source_observation_ids"), evidence.get("observation_timestamps")
                if (evidence.get("detector") != movement_detectors[self.behaviour]
                        or not isinstance(ids, list) or len(ids) < 2
                        or not isinstance(timestamps, list) or len(timestamps) != len(ids)
                        or not evidence.get("acceptance_reasons")):
                    raise ValueError("movement baseline events require versioned, timestamped accepted source evidence")
                if (not all(isinstance(oid, str) and oid for oid in ids) or len(ids) != len(set(ids))
                        or any(not isinstance(t, (int, float)) or not self.start_timestamp <= t <= self.end_timestamp for t in timestamps)
                        or any(b <= a for a, b in zip(timestamps, timestamps[1:]))):
                    raise ValueError("movement source IDs and timestamps must be unique, ordered and inside the event interval")
        return self


class PersonResult(Contract):
    person_id: str
    chunks: list[ChunkResult]
    events: list[BehaviourEvent]
    movement_diagnostics: dict[str, Any] = Field(default_factory=dict)


class Person2VideoResult(Contract):
    schema_version: Literal["1.0"] = "1.0"
    source_schema_version: Literal["1.0"] = "1.0"
    video_id: str
    score_semantics: Literal["uncalibrated_cosine_similarity"] = "uncalibrated_cosine_similarity"
    configuration: Person2Config
    embedding_spaces: dict[str, list[str]]
    prototype_version: str | None
    persons: list[PersonResult]


class Prototype(Contract):
    behaviour: str = Field(min_length=1)
    embedding: Embedding
    example_count: int = Field(ge=1)


class PrototypeBank(Contract):
    schema_version: Literal["1.0"] = "1.0"
    version: str = Field(min_length=1)
    prototypes: list[Prototype]

    @model_validator(mode="after")
    def unique_labels(self):
        labels = [p.behaviour for p in self.prototypes]
        if len(labels) != len(set(labels)):
            raise ValueError("prototype behaviour labels must be unique")
        spaces = {(p.embedding.space, len(p.embedding.values)) for p in self.prototypes}
        if len(spaces) > 1:
            raise ValueError("all prototypes must use the same embedding space")
        return self
