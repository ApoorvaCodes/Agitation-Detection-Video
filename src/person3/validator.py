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
    if item_id != "cmai_01_pacing_aimless_wandering" and len(usable) < 2:
        return False, ["insufficient_pose_quality"]
    if item_id == "cmai_07_hitting":
        motion_rows = [s.quality_flags.get("hitting_features") for s in usable
                       if s.quality_flags.get("candidate_source") == "motion_baseline"]
        motion_rows = [row for row in motion_rows if row]
        if motion_rows:
            if len(motion_rows) < 2 or not any((row.get("wrist_velocity") or 0) > 0
                    and row.get("wrist_acceleration") is not None and row.get("arm_extension") is not None
                    and row.get("extension_change") is not None for row in motion_rows):
                return False, ["hitting_motion_features_incomplete"]
        else:
            measured = [s for s in usable if any(s.motion_features.get(f"{side}_wrist.velocity.speed") is not None
                        and s.motion_features.get(f"{side}_wrist.acceleration.magnitude") is not None
                        and f"{side}_wrist" in s.pose and f"{side}_elbow" in s.pose
                        for side in ("left", "right"))]
            if len(measured) < 2:
                return False, ["hitting_motion_features_unavailable"]
    elif item_id == "cmai_08_kicking":
        limbs = {"left_foot", "right_foot"}
        contacts = [e for s in usable for e in s.quality_flags.get("contact_evidence", [])
                    if e.get("person_id") == packet.person_id and e.get("frame_index") == s.frame_index
                    and e.get("contact") == "observed" and e.get("limb") in limbs
                    and e.get("target_visible") and e.get("actor_limb_visible")]
        if not contacts:
            return False, ["target_or_contact_evidence_unavailable"]
    elif item_id in {"cmai_01_pacing_aimless_wandering", "cmai_29_general_restlessness"}:
        rows = [s for s in packet.segments if not s.quality_flags.get("bbox_interpolated")]
        expected_detector = ("pacing_trajectory_v1" if item_id == "cmai_01_pacing_aimless_wandering"
                             else "restlessness_pose_motion_v1")
        pattern = next((s.quality_flags.get("movement_pattern") for s in rows
                        if s.quality_flags.get("movement_pattern", {}).get("detector") == expected_detector), None)
        if len(rows) < 2 or pattern is None:
            return False, ["movement_pattern_source_evidence_unavailable"]
        if item_id == "cmai_01_pacing_aimless_wandering":
            if pattern.get("direction_reversals", 0) < 1 or pattern.get("path_length_body_scales", 0) <= 0:
                return False, ["pacing_trajectory_evidence_incomplete"]
        else:
            pose_rows = [s for s in rows if s.quality_flags.get("pose_detected")]
            if len(pose_rows) < 2 or pattern.get("repeated_movement_burst_count", 0) < 2:
                return False, ["restlessness_pose_motion_evidence_incomplete"]
    times = [s.timestamp for s in packet.segments]
    if any(b <= a for a, b in zip(times, times[1:])):
        return False, ["non_increasing_source_timestamps"]
    return True, flags
