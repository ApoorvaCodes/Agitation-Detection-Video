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
MOVEMENT_BASELINE_ITEMS = {"cmai_01_pacing_aimless_wandering", "cmai_29_general_restlessness"}
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
