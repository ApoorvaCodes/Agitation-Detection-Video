"""Explicit visual-contact evidence; geometry/proximity is never contact proof."""
from typing import Literal
from pydantic import Field, model_validator
from person1.contracts import BoundingBox
from person2.contracts import Contract


class ContactObservation(Contract):
    evidence_id: str = Field(min_length=1)
    person_id: str = Field(min_length=1)
    frame_index: int = Field(ge=0)
    timestamp: float = Field(ge=0)
    limb: Literal["left_hand", "right_hand", "left_foot", "right_foot"]
    target_kind: Literal["person", "self", "object"]
    target_id: str = Field(min_length=1)
    target_bbox: BoundingBox
    target_visible: bool
    actor_limb_visible: bool
    contact: Literal["observed", "absent", "unclear"]
    # Reference to inspected source frame / external detector evidence, not label.
    evidence_reference: str = Field(min_length=1)

    @model_validator(mode="after")
    def positive_target_box(self):
        if self.target_bbox.coordinate_system != "normalized":
            raise ValueError("contact target must use normalized source coordinates")
        if self.target_bbox.x_max <= self.target_bbox.x_min or self.target_bbox.y_max <= self.target_bbox.y_min:
            raise ValueError("contact target requires a positive source bounding box")
        return self


class InteractionEvidence(Contract):
    schema_version: Literal["cmai-interaction-evidence-1.0"] = "cmai-interaction-evidence-1.0"
    video_id: str = Field(min_length=1)
    recording_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_id: str = Field(min_length=1)
    provider_version: str = Field(min_length=1)
    method: Literal["manual_visual_review", "external_contact_detector"]
    annotation_protocol: str = Field(min_length=1)
    observations: list[ContactObservation]

    @model_validator(mode="after")
    def unique_evidence(self):
        if len({o.evidence_id for o in self.observations}) != len(self.observations):
            raise ValueError("duplicate contact evidence IDs")
        keys = [(o.person_id, o.frame_index, o.limb, o.target_kind, o.target_id) for o in self.observations]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate/conflicting contact observations")
        return self

    def validate_source(self, source, recording_sha256=None):
        if self.video_id != source.video.video_id:
            raise ValueError("interaction evidence belongs to another video")
        if recording_sha256 is not None and self.recording_sha256 != recording_sha256:
            raise ValueError("interaction evidence recording checksum mismatch")
        people = {p.person_id: {o.frame_index:o for o in p.observations} for p in source.persons}
        for e in self.observations:
            actor = people.get(e.person_id, {}).get(e.frame_index)
            if actor is None or abs(actor.timestamp-e.timestamp) > 1e-6:
                raise ValueError("contact evidence has no matching actor frame/timestamp")
            if e.target_kind == "person":
                target = people.get(e.target_id, {}).get(e.frame_index)
                if target is None or e.target_id == e.person_id or abs(target.timestamp-e.timestamp) > 1e-6:
                    raise ValueError("contact target has no matching distinct person frame")
                if e.target_bbox != target.bbox:
                    raise ValueError("target box does not match the referenced track")
            if e.target_kind == "self" and e.target_id != e.person_id:
                raise ValueError("self-contact must reference the actor track")
            if e.target_kind == "self" and e.target_bbox != actor.bbox:
                raise ValueError("self target box must match the actor track")
        return self
