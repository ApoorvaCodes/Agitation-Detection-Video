"""Evidence-driven demo physical-behaviour rules.

This module is intentionally separate from the production detector bundle.  It
uses only serialized P1 pose/motion observations and emits candidates marked
``DEMO_ONLY``.  Its evidence scores are rule-support measurements, never
probabilities, clinical confidence, or validated CMAI predictions.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import median

from person2.contracts import BehaviourEvent
from cmai.taxonomy import validate_canonical_cmai_behaviour


ROOT = Path(__file__).resolve().parents[2]
DEMO_DETECTOR = "demo_physical_behaviour_rules_v1"
DEMO_STATUS = "DEMO_ONLY"
SCORE_SEMANTICS = "demo_rule_evidence_strength_not_probability"


@dataclass(frozen=True)
class DemoPhysicalBehaviourConfig:
    enabled: bool = True
    min_valid_fraction: float = .6
    min_event_duration: float = .25
    max_event_duration: float = 8.0
    baseline_mad_multiplier: float = 3.0
    hitting: dict = None
    kicking: dict = None
    pacing: dict = None
    repetitive: dict = None
    restlessness: dict = None

    @classmethod
    def from_path(cls, path=None):
        data = json.loads(Path(path or ROOT / "configs/demo_physical_behaviour.json").read_text())
        return cls(**data)


def _value(observation, name):
    value = observation.motion.feature_values.get(name) if observation.motion else None
    return float(value) if value is not None and math.isfinite(value) else None


def _point(observation, name):
    pose = observation.normalized_pose
    landmark = pose.landmarks.get(name) if pose else None
    if landmark is None or landmark.x is None or landmark.y is None:
        return None
    return (float(landmark.x), float(landmark.y))


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle(a, b, c):
    if not all((a, b, c)):
        return None
    ux, uy = a[0] - b[0], a[1] - b[1]
    vx, vy = c[0] - b[0], c[1] - b[1]
    denominator = math.hypot(ux, uy) * math.hypot(vx, vy)
    if denominator <= 1e-12:
        return None
    cosine = max(-1.0, min(1.0, (ux * vx + uy * vy) / denominator))
    return math.degrees(math.acos(cosine))


def _baseline(values, multiplier):
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    # Use the lower half as the track's normal-motion baseline so a short
    # action burst does not raise its own threshold to the burst magnitude.
    ordered = sorted(clean)
    center = median(ordered[:max(1, (len(ordered) + 1) // 2)])
    mad = median([abs(v - center) for v in ordered[:max(1, (len(ordered) + 1) // 2)]])
    return center + multiplier * max(mad, 1e-6)


def _valid_fraction(rows):
    return sum(not o.quality.bbox_interpolated and bool(o.quality.pose_detected)
               for o in rows) / len(rows) if rows else 0.0


def _groups(rows, flags):
    groups, current = [], []
    for row, flag in zip(rows, flags):
        if flag:
            current.append(row)
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def _strength(rows, required):
    observed = []
    for row in rows:
        observed.extend(v for v in (_value(row, name) for name in required) if v is not None)
    if not observed:
        return 0.0
    return min(1.0, len(observed) / (len(rows) * len(required)))


def _candidate(person_id, label, rows, chunks, evidence, strength):
    first, last = rows[0], rows[-1]
    item = validate_canonical_cmai_behaviour(label)
    chunk_ids = [getattr(c, "chunk_id", f"window:{c.start_timestamp:g}:{c.end_timestamp:g}")
                 for c in chunks if any(c.start_timestamp <= o.timestamp <= c.end_timestamp for o in rows)]
    # Direct unit tests can exercise the detector before P2 chunking.  The
    # worker replaces these temporary source-window IDs with real P2 chunk IDs.
    if not chunk_ids:
        chunk_ids = [f"demo:{person_id}:{first.frame_index}-{last.frame_index}"]
    return BehaviourEvent(
        behaviour=label, start_timestamp=first.timestamp,
        end_timestamp=max(last.timestamp, first.timestamp + 1e-6),
        peak_similarity=round(float(max(0.0, min(1.0, strength))), 6),
        chunk_ids=chunk_ids,
        detector_name=DEMO_DETECTOR, candidate_status=DEMO_STATUS,
        score_semantics=SCORE_SEMANTICS, evidence=evidence,
        canonical_cmai_id=item.item_id, canonical_cmai_name=item.display_name,
    )


class DemoPhysicalBehaviourDetector:
    """Run conservative, configurable rules over one P1 track."""

    def __init__(self, config=None):
        self.config = config or DemoPhysicalBehaviourConfig.from_path()
        self.diagnostics = {}

    def detect_person(self, person):
        if not self.config.enabled or not person.observations:
            return []
        rows = person.observations
        chunks = person.windows
        hitting = self._hitting(person.person_id, rows, chunks)
        kicking = self._kicking(person.person_id, rows, chunks)
        specific = hitting + kicking
        pacing = self._pacing(person.person_id, rows, chunks) if not specific else []
        repetitive = self._repetitive(person.person_id, rows, chunks) if not specific else []
        restlessness = self._restlessness(person.person_id, rows, chunks) if not specific and not pacing and not repetitive else []
        return sorted(specific + pacing + repetitive + restlessness,
                      key=lambda event: (event.start_timestamp, event.behaviour))

    def detect(self, result, source):
        source_people = {person.person_id: person for person in source.persons}
        for person in result.persons:
            source_person = source_people.get(person.person_id)
            if source_person is None:
                continue
            candidates = self.detect_person(source_person)
            # P2 events must reference P2 chunks, while the detector itself
            # evaluates the richer P1 rolling windows. Preserve the exact
            # source interval and attach only overlapping P2 chunk IDs.
            for candidate in candidates:
                candidate.chunk_ids = [chunk.chunk_id for chunk in person.chunks
                                       if chunk.start_timestamp < candidate.end_timestamp
                                       and chunk.end_timestamp > candidate.start_timestamp]
                if not candidate.chunk_ids:
                    continue
            candidates = [candidate for candidate in candidates if candidate.chunk_ids]
            person.events.extend(candidates)
            person.events.sort(key=lambda event: (event.start_timestamp, event.behaviour))
        return result

    def _hitting(self, person_id, rows, chunks):
        cfg = self.config.hitting
        speeds = [_value(o, f"{side}_wrist.velocity.speed") for o in rows for side in ("left", "right")]
        speed_baseline = _baseline(speeds, self.config.baseline_mad_multiplier) or 0
        acceleration_baseline = _baseline([_value(o, f"{side}_wrist.acceleration.magnitude")
                                           for o in rows for side in ("left", "right")],
                                          self.config.baseline_mad_multiplier) or 0
        speed_cut = max(float(cfg["speed_floor"]), speed_baseline * float(cfg["speed_baseline_multiplier"]))
        acceleration_cut = max(float(cfg["acceleration_floor"]), acceleration_baseline * float(cfg["acceleration_baseline_multiplier"]))
        flags, evidence = [], []
        for i, row in enumerate(rows):
            sides = []
            for side in ("left", "right"):
                speed = _value(row, f"{side}_wrist.velocity.speed")
                acceleration = _value(row, f"{side}_wrist.acceleration.magnitude")
                wrist, shoulder = _point(row, f"{side}_wrist"), _point(row, f"{side}_shoulder")
                opposite = _point(row, "right_shoulder" if side == "left" else "left_shoulder")
                extension = _distance(wrist, shoulder) / max(_distance(shoulder, opposite), 1e-6) if wrist and shoulder and opposite else None
                return_seen = False
                for later in rows[i + 1:]:
                    if later.timestamp - row.timestamp > float(cfg["return_seconds"]):
                        break
                    later_speed = _value(later, f"{side}_wrist.velocity.speed")
                    later_wrist, later_shoulder = _point(later, f"{side}_wrist"), _point(later, f"{side}_shoulder")
                    later_extension = _distance(later_wrist, later_shoulder) / max(_distance(later_shoulder, opposite), 1e-6) if later_wrist and later_shoulder and opposite else None
                    return_seen |= (speed is not None and later_speed is not None and later_speed <= speed * float(cfg["return_speed_ratio"])) or (extension is not None and later_extension is not None and later_extension <= extension * float(cfg["return_extension_ratio"]))
                if speed is not None and acceleration is not None and extension is not None and speed >= speed_cut and acceleration >= acceleration_cut and extension >= float(cfg["extension_threshold"]) and return_seen:
                    sides.append((side, speed, acceleration, extension))
            flags.append(bool(sides))
            if sides:
                evidence.append((row, sides))
        groups = [g for g in _groups(rows, flags) if len(g) >= int(cfg["min_consecutive_frames"])]
        out = []
        for group in groups:
            selected = list(group)
            end_index = rows.index(group[-1])
            while end_index + 1 < len(rows) and selected[-1].timestamp - selected[0].timestamp < self.config.min_event_duration:
                end_index += 1
                selected.append(rows[end_index])
            duration = selected[-1].timestamp - selected[0].timestamp
            if self.config.min_event_duration <= duration <= self.config.max_event_duration:
                values = [item for row, item in evidence if row in group]
                out.append(_candidate(person_id, "cmai_07_hitting", selected, chunks, {
                    "description": "Rapid wrist extension with acceleration and subsequent deceleration/return.",
                    "features": {"wrist": values, "contact_status": "not_established"},
                    "valid_fraction": _valid_fraction(group), "status": DEMO_STATUS,
                }, _strength(group, ["left_wrist.velocity.speed", "right_wrist.velocity.speed"])))
        return out

    def _kicking(self, person_id, rows, chunks):
        cfg = self.config.kicking
        speeds = [_value(o, f"{side}_ankle.velocity.speed") for o in rows for side in ("left", "right")]
        speed_baseline = _baseline(speeds, self.config.baseline_mad_multiplier) or 0
        acceleration_baseline = _baseline([_value(o, f"{side}_ankle.acceleration.magnitude")
                                           for o in rows for side in ("left", "right")],
                                          self.config.baseline_mad_multiplier) or 0
        speed_cut = max(float(cfg["speed_floor"]), speed_baseline * float(cfg["speed_baseline_multiplier"]))
        acceleration_cut = max(float(cfg["acceleration_floor"]), acceleration_baseline * float(cfg["acceleration_baseline_multiplier"]))
        flags, details = [], []
        for i, row in enumerate(rows):
            legs = []
            for side in ("left", "right"):
                speed = _value(row, f"{side}_ankle.velocity.speed")
                acceleration = _value(row, f"{side}_ankle.acceleration.magnitude")
                angular_velocity = _value(row, f"{side}_knee.angular_velocity")
                angle = _angle(_point(row, f"{side}_hip"), _point(row, f"{side}_knee"), _point(row, f"{side}_ankle"))
                return_seen = bool(speed is not None and any(
                    later.timestamp - row.timestamp <= float(cfg["return_seconds"])
                    and (_value(later, f"{side}_ankle.velocity.speed") or math.inf) <= speed * float(cfg["return_speed_ratio"])
                    for later in rows[i + 1:]))
                if speed is not None and acceleration is not None and angular_velocity is not None and angle is not None and speed >= speed_cut and acceleration >= acceleration_cut and abs(angular_velocity) >= float(cfg["knee_angular_velocity_floor"]) and angle >= float(cfg["knee_angle_threshold"]) and return_seen:
                    legs.append((side, speed, acceleration, angle))
            flags.append(bool(legs))
            if legs:
                details.append((row, legs))
        out = []
        for group in [g for g in _groups(rows, flags) if len(g) >= int(cfg["min_consecutive_frames"])]:
            selected = list(group)
            end_index = rows.index(group[-1])
            while end_index + 1 < len(rows) and selected[-1].timestamp - selected[0].timestamp < self.config.min_event_duration:
                end_index += 1
                selected.append(rows[end_index])
            duration = selected[-1].timestamp - selected[0].timestamp
            if self.config.min_event_duration <= duration <= self.config.max_event_duration:
                out.append(_candidate(person_id, "cmai_08_kicking", selected, chunks, {
                    "description": "Rapid ankle motion with acceleration and extended knee followed by return.",
                    "features": {"ankle_knee": [item for row, item in details if row in group]},
                    "valid_fraction": _valid_fraction(group), "status": DEMO_STATUS,
                }, _strength(group, ["left_ankle.velocity.speed", "right_ankle.velocity.speed"])))
        return out

    def _pacing(self, person_id, rows, chunks):
        cfg = self.config.pacing
        candidates = []
        for window in chunks:
            if window.low_quality or window.valid_fraction < self.config.min_valid_fraction:
                continue
            path = window.features.get("body_centroid.cumulative_path_length")
            ratio = window.features.get("body_centroid.net_path_ratio")
            if path is None or ratio is None or path < float(cfg["min_path_length"]) or ratio > float(cfg["max_path_ratio"]):
                continue
            selected = [o for o in rows if window.start_timestamp <= o.timestamp <= window.end_timestamp]
            if window.end_timestamp - window.start_timestamp < float(cfg["min_duration"]):
                continue
            directions = []
            for before, after in zip(selected, selected[1:]):
                before_points = [_point(before, name) for name in ("left_hip", "right_hip")]
                after_points = [_point(after, name) for name in ("left_hip", "right_hip")]
                before_points = [point for point in before_points if point]
                after_points = [point for point in after_points if point]
                if before_points and after_points:
                    directions.append(sum(point[0] for point in after_points) / len(after_points)
                                      - sum(point[0] for point in before_points) / len(before_points))
            changes = sum(a * b < 0 for a, b in zip(directions, directions[1:]) if a and b)
            if changes >= int(cfg["min_direction_changes"]):
                candidates.append(_candidate(person_id, "cmai_01_pacing_aimless_wandering", selected, chunks, {
                    "description": "Sustained centroid path with low net/path ratio and direction changes.",
                    "features": {k: window.features.get(k) for k in ("body_centroid.cumulative_path_length", "body_centroid.net_displacement", "body_centroid.net_path_ratio")},
                    "valid_fraction": window.valid_fraction, "status": DEMO_STATUS,
                }, window.valid_fraction))
        if candidates:
            return candidates
        # The P1 default rolling window is two seconds. Pacing is explicitly a
        # sustained behaviour, so evaluate the complete valid track as well
        # instead of requiring one oversized window.
        selected = [o for o in rows if _point(o, "left_hip") or _point(o, "right_hip")]
        if not selected or selected[-1].timestamp - selected[0].timestamp < float(cfg["min_duration"]):
            return []
        centers = []
        for row in selected:
            points = [point for point in (_point(row, "left_hip"), _point(row, "right_hip")) if point]
            centers.append((sum(point[0] for point in points) / len(points),
                            sum(point[1] for point in points) / len(points)))
        path = sum(_distance(a, b) for a, b in zip(centers, centers[1:]))
        net = _distance(centers[0], centers[-1])
        directions = [b[0] - a[0] for a, b in zip(centers, centers[1:])]
        changes = sum(a * b < 0 for a, b in zip(directions, directions[1:]) if a and b)
        ratio = net / path if path else 1.0
        if path >= float(cfg["min_path_length"]) and ratio <= float(cfg["max_path_ratio"]) and changes >= int(cfg["min_direction_changes"]):
            return [_candidate(person_id, "cmai_01_pacing_aimless_wandering", selected, chunks, {
                "description": "Sustained centroid path with low net/path ratio and direction changes.",
                "features": {"body_centroid.cumulative_path_length": path,
                             "body_centroid.net_displacement": net,
                             "body_centroid.net_path_ratio": ratio,
                             "direction_changes": changes},
                "valid_fraction": _valid_fraction(selected), "status": DEMO_STATUS,
            }, _valid_fraction(selected))]
        return candidates

    def _repetitive(self, person_id, rows, chunks):
        cfg = self.config.repetitive
        signal = [max((_value(o, "left_wrist.velocity.speed") or 0), (_value(o, "right_wrist.velocity.speed") or 0)) for o in rows]
        baseline = _baseline(signal, self.config.baseline_mad_multiplier) or 0
        peaks = [i for i in range(1, len(signal) - 1) if signal[i] >= baseline and signal[i] >= signal[i - 1] and signal[i] >= signal[i + 1]]
        separated = []
        for peak in peaks:
            if not separated or peak - separated[-1] >= int(cfg["min_peak_separation"]):
                separated.append(peak)
        peaks = separated
        if len(peaks) < int(cfg["min_repetitions"]):
            return []
        intervals = [rows[b].timestamp - rows[a].timestamp for a, b in zip(peaks, peaks[1:])]
        similarity = 1.0 - (max(intervals) - min(intervals)) / max(max(intervals), 1e-6)
        if similarity < float(cfg["similarity_tolerance"]):
            return []
        selected = rows[peaks[0]:peaks[-1] + 1]
        return [_candidate(person_id, "cmai_26_repetitious_mannerisms", selected, chunks, {
            "description": "Repeated similar wrist-speed peaks across multiple observations.",
            "features": {"repetition_count": len(peaks), "interval_similarity": similarity},
            "valid_fraction": _valid_fraction(selected), "status": DEMO_STATUS,
        }, min(1.0, similarity * _valid_fraction(selected)))]

    def _restlessness(self, person_id, rows, chunks):
        cfg = self.config.restlessness
        for window in chunks:
            energy = window.features.get("overall_motion_energy")
            if not window.low_quality and energy is not None and energy >= float(cfg["motion_energy_floor"]) and window.end_timestamp - window.start_timestamp >= float(cfg["min_duration"]):
                selected = [o for o in rows if window.start_timestamp <= o.timestamp <= window.end_timestamp]
                return [_candidate(person_id, "cmai_29_general_restlessness", selected, chunks, {
                    "description": "Sustained elevated whole-body motion without a stronger specific action rule.",
                    "features": {"overall_motion_energy": energy, "arm_motion_energy": window.features.get("arm_motion_energy"), "leg_motion_energy": window.features.get("leg_motion_energy")},
                    "valid_fraction": window.valid_fraction, "status": DEMO_STATUS,
                }, window.valid_fraction)]
        return []
