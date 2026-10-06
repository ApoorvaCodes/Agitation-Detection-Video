"""Strict camera taxonomy, separate from the legacy Person 3 vocabulary."""
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from person2.contracts import Contract

VERSION = "cmai-long-form-camera-v1"
ROOT = Path(__file__).resolve().parents[2]
INITIAL_ITEMS = {"cmai_01_pacing_aimless_wandering", "cmai_26_repetitious_mannerisms",
                 "cmai_29_general_restlessness"}
ACTION_ITEMS = {"cmai_07_hitting", "cmai_08_kicking"}
CAMERA_OBSERVABLE_STATUSES = {"camera_candidate", "conditional", "fall_like_only"}
LEGACY_LABELS = {item.removeprefix("cmai_").split("_", 1)[1]: item for item in INITIAL_ITEMS}


class CameraItem(Contract):
    item_id: str
    item_number: int = Field(ge=1, le=29)
    display_name: str = Field(min_length=1)
    camera_status: Literal["camera_candidate", "conditional", "fall_like_only", "not_camera_only", "out_of_initial_scope"]
    positive_definition: str = Field(min_length=1)
    exclusions: list[str] = Field(min_length=1)
    evidence_needs: list[str] = Field(min_length=1)


class CameraTaxonomy(Contract):
    taxonomy_version: Literal["cmai-long-form-camera-v1"]
    form: Literal["CMAI 29-item long form"]
    edition: str = Field(min_length=1)
    source: str = Field(min_length=1)
    items: list[CameraItem]

    @model_validator(mode="after")
    def unique_complete_items(self):
        if sorted(i.item_number for i in self.items) != list(range(1, 30)):
            raise ValueError("taxonomy must contain exactly items 01 through 29")
        if len({i.item_id for i in self.items}) != 29:
            raise ValueError("taxonomy item IDs must be unique")
        for item in self.items:
            if not item.item_id.startswith(f"cmai_{item.item_number:02d}_"):
                raise ValueError("item ID and CMAI number disagree")
        return self

    def item(self, item_id):
        for item in self.items:
            if item.item_id == item_id:
                return item
        raise ValueError(f"unknown CMAI item ID: {item_id}")


def load_taxonomy(path=None):
    return CameraTaxonomy.model_validate_json(Path(path or ROOT / "configs/cmai_long_form_camera_v1.json").read_text())


def camera_observable_items(taxonomy=None):
    """Return canonical items whose definitions permit camera evidence.

    This is derived from the versioned taxonomy; it is deliberately broader
    than the currently enabled detector assets.  It must not be used to claim
    that an item has been evaluated when no matching detector rule exists.
    """
    taxonomy = taxonomy or load_taxonomy()
    return tuple(item for item in taxonomy.items if item.camera_status in CAMERA_OBSERVABLE_STATUSES)


def camera_observable_ids(taxonomy=None):
    return frozenset(item.item_id for item in camera_observable_items(taxonomy))


def validate_canonical_cmai_behaviour(cmai_id, cmai_name=None, *, taxonomy=None,
                                      allowed_ids=None):
    """Validate and canonicalize a physical-video behaviour.

    The returned name is always read from the versioned taxonomy.  Caller
    supplied names are checked only for consistency and are never trusted for
    display.  Unknown, non-camera, and disallowed IDs fail closed.
    """
    taxonomy = taxonomy or load_taxonomy()
    if not isinstance(cmai_id, str) or not cmai_id.strip():
        raise ValueError("canonical CMAI ID is required")
    item = taxonomy.item(cmai_id)
    if item.camera_status not in CAMERA_OBSERVABLE_STATUSES:
        raise ValueError("CMAI item is not observable from physical video")
    allowed = set(allowed_ids) if allowed_ids is not None else INITIAL_ITEMS | ACTION_ITEMS
    if item.item_id not in allowed:
        raise ValueError("CMAI item is not enabled for this physical-video pipeline")
    if cmai_name is not None and cmai_name != item.display_name:
        raise ValueError("canonical CMAI name does not match the taxonomy")
    return item


def canonical_item(label, *, allow_legacy=False):
    item_id = LEGACY_LABELS.get(label, label) if allow_legacy else label
    return load_taxonomy().item(item_id).item_id


def require_initial_items(labels):
    for label in labels:
        canonical_item(label)
        if label not in INITIAL_ITEMS:
            raise ValueError(f"item not enabled for pose/motion camera research: {label}")


def require_action_items(labels):
    for label in labels:
        canonical_item(label)
        if label not in ACTION_ITEMS:
            raise ValueError(f"item not enabled for interaction action research: {label}")
