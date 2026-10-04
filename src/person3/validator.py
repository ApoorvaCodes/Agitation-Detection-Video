"""Deterministic CMAI-aware evidence quality checks."""
from person3.taxonomy import normalize_behaviour
from cmai.taxonomy import CAMERA_OBSERVABLE_STATUSES, canonical_item, load_taxonomy


def validate_packet(packet):
    flags = []
    legacy = None
    try:
        item_id = canonical_item(packet.behaviour, allow_legacy=True)
    except ValueError:
        legacy = normalize_behaviour(packet.behaviour)
        if legacy is None:
            return False, ["behaviour_not_in_cmai_taxonomy"]
        item_id = {"hitting": "cmai_07_hitting", "kicking": "cmai_08_kicking"}.get(legacy.key, legacy.key)
    try:
        item = load_taxonomy().item(item_id)
    except ValueError:
        # Keep the legacy Person 3 handoff readable; its physical-domain
        # labels are not emitted by the canonical camera result contract.
        if legacy is None or legacy.domain != "physical":
            return False, ["behaviour_not_observable_from_camera"]
    else:
        if item.camera_status not in CAMERA_OBSERVABLE_STATUSES:
            return False, ["behaviour_not_observable_from_camera"]
    if not packet.segments:
        return False, ["no_source_observations"]
    if len(packet.segments) < 2:
        return False, ["fewer_than_two_source_observations"]
    usable = [s for s in packet.segments if s.quality_flags.get("pose_detected") and not s.quality_flags.get("bbox_interpolated")]
    if len(usable) < 2:
        return False, ["insufficient_pose_quality"]
    if item_id in {"cmai_07_hitting", "cmai_08_kicking"}:
        limbs = {"left_hand", "right_hand"} if item_id == "cmai_07_hitting" else {"left_foot", "right_foot"}
        contacts = [e for s in usable for e in s.quality_flags.get("contact_evidence", [])
                    if e.get("person_id") == packet.person_id and e.get("frame_index") == s.frame_index
                    and e.get("contact") == "observed" and e.get("limb") in limbs
                    and e.get("target_visible") and e.get("actor_limb_visible")]
        if not contacts:
            return False, ["target_or_contact_evidence_unavailable"]
    times = [s.timestamp for s in packet.segments]
    if any(b <= a for a, b in zip(times, times[1:])):
        return False, ["non_increasing_source_timestamps"]
    return True, flags
