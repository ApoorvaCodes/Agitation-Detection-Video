"""Deterministic CMAI-aware evidence quality checks."""
from person3.taxonomy import normalize_behaviour


def validate_packet(packet):
    flags = []
    item = normalize_behaviour(packet.behaviour)
    if item is None:
        return False, ["behaviour_not_in_cmai_taxonomy"]
    if not packet.segments:
        return False, ["no_source_observations"]
    if len(packet.segments) < 2:
        return False, ["fewer_than_two_source_observations"]
    usable = [s for s in packet.segments if s.quality_flags.get("pose_detected") and not s.quality_flags.get("bbox_interpolated")]
    if len(usable) < 2:
        return False, ["insufficient_pose_quality"]
    times = [s.timestamp for s in packet.segments]
    if any(b <= a for a, b in zip(times, times[1:])):
        return False, ["non_increasing_source_timestamps"]
    return True, flags
