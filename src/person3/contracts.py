"""Strict, versioned Person 2 to Person 3 and event contracts."""
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class CandidateBehaviour(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    candidate_id: str = Field(min_length=1)
    person_id: str = Field(min_length=1)
    behaviour: str = Field(min_length=1)
    candidate_score: float = Field(ge=-1, le=1)
    start_timestamp: float = Field(ge=0)
    end_timestamp: float = Field(ge=0)
    source_window_ids: list[str] = Field(default_factory=list)
    source_observation_ids: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def interval_is_valid(self):
        if self.end_timestamp <= self.start_timestamp:
            raise ValueError("candidate interval must have positive duration")
        if len(self.source_window_ids) != len(set(self.source_window_ids)):
            raise ValueError("source_window_ids must be unique")
        if len(self.source_observation_ids) != len(set(self.source_observation_ids)):
            raise ValueError("source_observation_ids must be unique")
        return self


class EvidenceSegment(StrictModel):
    evidence_id: str
    timestamp: float = Field(ge=0)
    frame_index: int = Field(ge=0)
    motion_features: dict[str, float | None] = Field(default_factory=dict)
    pose: dict[str, dict[str, float | None]] = Field(default_factory=dict)
    quality_flags: dict[str, Any] = Field(default_factory=dict)


class EvidencePacket(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    person_id: str
    candidate_id: str
    behaviour: str
    candidate_score: float = Field(ge=-1, le=1)
    candidate_start: float = Field(ge=0)
    candidate_end: float = Field(gt=0)
    source_window_ids: list[str]
    segments: list[EvidenceSegment]

    @model_validator(mode="after")
    def interval_is_valid(self):
        if self.candidate_end <= self.candidate_start:
            raise ValueError("evidence candidate interval must have positive duration")
        return self


class Verification(StrictModel):
    decision: Literal["supported", "unsupported", "insufficient_evidence"]
    reason: str = Field(min_length=1, max_length=2000)
    evidence_segment_ids: list[str] = Field(default_factory=list)


class ValidationResult(StrictModel):
    event_id: str
    person_id: str
    candidate_id: str
    behaviour: str
    candidate_score: float
    validation_status: Literal["supported", "unsupported", "insufficient_evidence"]
    reason: str
    start_timestamp: float | None = None
    end_timestamp: float | None = None
    source_window_ids: list[str] = Field(default_factory=list)
    source_observation_ids: list[str] = Field(default_factory=list)
    selected_evidence_ids: list[str] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)
    verifier_error: str | None = None

    @model_validator(mode="after")
    def supported_requires_interval(self):
        if self.validation_status == "supported":
            if self.start_timestamp is None or self.end_timestamp is None or self.end_timestamp <= self.start_timestamp:
                raise ValueError("supported event requires a positive source-derived interval")
            if not self.selected_evidence_ids:
                raise ValueError("supported event requires selected source evidence")
        return self
