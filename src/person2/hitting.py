"""Experimental body-normalized arm-motion punch baseline (not a trained model)."""
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path


@dataclass(frozen=True)
class HittingConfig:
    enabled: bool = True
    min_velocity: float = 1.25          # torso lengths / second
    min_acceleration: float = 4.0        # torso lengths / second squared
    min_velocity_observations: int = 2
    min_extension_change: float = 0.28   # torso lengths
    min_burst_duration: float = 0.12
    max_burst_duration: float = 1.0
    min_score: float = 0.62
    max_gap_seconds: float = 0.5
    min_landmark_visibility: float = 0.45
    min_detection_confidence: float = 0.4
    max_rest_velocity: float = 0.75
    deceleration_ratio: float = 0.72

    def __post_init__(self):
        values = asdict(self)
        values.pop("enabled")
        if any(not math.isfinite(v) for v in values.values()):
            raise ValueError("Hitting thresholds must be finite")
        if min(self.min_velocity, self.min_acceleration, self.min_extension_change,
               self.min_burst_duration, self.max_burst_duration, self.max_gap_seconds) <= 0:
            raise ValueError("Hitting motion thresholds and durations must be positive")
        if self.min_burst_duration > self.max_burst_duration:
            raise ValueError("minimum burst duration must not exceed maximum")
        if not isinstance(self.min_velocity_observations,int) or self.min_velocity_observations < 2:
            raise ValueError("min_velocity_observations must be an integer of at least two")
        if not 0 < self.min_score <= 1 or not 0 <= self.min_landmark_visibility <= 1 or not 0 <= self.min_detection_confidence <= 1:
            raise ValueError("Hitting score/visibility thresholds are out of range")
        if not 0 < self.deceleration_ratio < 1:
            raise ValueError("deceleration_ratio must be between zero and one")

    @classmethod
    def load(cls, path=None):
        path = Path(path or os.getenv("HITTING_CONFIG") or
                    Path(__file__).resolve().parents[2] / "configs/hitting_motion.json")
        return cls(**json.loads(path.read_text()))


def _xyz(point):
    if point is None or point.x is None or point.y is None:
        return None
    return (float(point.x), float(point.y), float(point.z or 0.0))


def _distance(a, b):
    return math.sqrt(sum((x-y) ** 2 for x, y in zip(a, b)))


def _landmark(observation, name, config):
    pose = observation.normalized_pose
    point = pose.landmarks.get(name) if pose else None
    if point is None or not observation.quality.landmark_validity.get(name, True):
        return None
    if point.visibility is not None and point.visibility < config.min_landmark_visibility:
        return None
    return _xyz(point)


def _torso_scale(observation, config):
    ls, rs = _landmark(observation, "left_shoulder", config), _landmark(observation, "right_shoulder", config)
    lh, rh = _landmark(observation, "left_hip", config), _landmark(observation, "right_hip", config)
    if ls and rs and lh and rh:
        shoulder_center = tuple((a+b)/2 for a,b in zip(ls,rs))
        hip_center = tuple((a+b)/2 for a,b in zip(lh,rh))
        scale = _distance(shoulder_center, hip_center)
        if scale > 1e-4:
            return scale
    if ls and rs:
        scale = _distance(ls,rs)
        if scale > 1e-4:
            return scale
    return None


def analyze_hitting(person, config=None):
    """Return per-person punch events and maxima for dashboard calibration.

    Timestamps, rather than nominal FPS, drive all derivatives. Invalid poses,
    interpolated boxes and large temporal gaps split the evidence sequence.
    """
    config = config or HittingConfig()
    observations = person.observations
    by_side = {}
    diagnostics = {"max_left_wrist_velocity": 0.0, "max_right_wrist_velocity": 0.0,
                   "max_left_acceleration": 0.0, "max_right_acceleration": 0.0,
                   "max_left_wrist_displacement": 0.0, "max_right_wrist_displacement": 0.0,
                   "max_extension_change": 0.0,
                   "frames_with_yolo_person_detection": len({o.frame_index for o in observations if not o.quality.bbox_interpolated}),
                   "frames_with_valid_mediapipe_pose": len({o.frame_index for o in observations if o.quality.pose_detected and not o.quality.bbox_interpolated}),
                   "frames_with_valid_left_wrist": 0, "frames_with_valid_right_wrist": 0,
                   "frames_with_valid_left_arm_chain": 0, "frames_with_valid_right_arm_chain": 0,
                   "motion_observations": 0}
    for side in ("left", "right"):
        valid_wrist = [(o, _landmark(o, f"{side}_wrist", config)) for o in observations]
        valid_wrist = [(o, p) for o,p in valid_wrist if p is not None and o.quality.pose_detected and not o.quality.bbox_interpolated]
        diagnostics[f"frames_with_valid_{side}_wrist"] = len({o.frame_index for o,_ in valid_wrist})
        if valid_wrist:
            first_o, first_p = valid_wrist[0]; last_o, last_p = valid_wrist[-1]
            for label,obs,point in (("first",first_o,first_p),("last",last_o,last_p)):
                diagnostics[f"{label}_{side}_wrist_frame_index"] = obs.frame_index
                diagnostics[f"{label}_{side}_wrist_timestamp"] = obs.timestamp
                for axis,value in zip(("x","y","z"),point):
                    diagnostics[f"{label}_{side}_wrist_{axis}"] = value
        for o in observations:
            if o.quality.pose_detected and not o.quality.bbox_interpolated and all(_landmark(o,f"{side}_{j}",config) for j in ("shoulder","elbow","wrist")):
                diagnostics[f"frames_with_valid_{side}_arm_chain"] += 1
    for side in ("left", "right"):
        shoulder_name, elbow_name, wrist_name = (f"{side}_{j}" for j in ("shoulder", "elbow", "wrist"))
        rows = []
        for observation in observations:
            valid = (not observation.quality.bbox_interpolated and observation.quality.pose_detected
                     and observation.quality.detection_confidence >= config.min_detection_confidence)
            shoulder = _landmark(observation, shoulder_name, config) if valid else None
            elbow = _landmark(observation, elbow_name, config) if valid else None
            wrist = _landmark(observation, wrist_name, config) if valid else None
            scale = _torso_scale(observation, config) if valid else None
            rows.append({"observation": observation, "shoulder": shoulder, "elbow": elbow,
                         "wrist": wrist, "scale": scale,
                         "extension": _distance(shoulder,wrist)/scale if shoulder and wrist and scale else None})
        # Derivatives reset at any missing pose or dropped-frame-sized gap.
        for index,row in enumerate(rows):
            row.update(speed=None, acceleration=None, elbow_speed=None, wrist_displacement=None, elbow_displacement=None)
            if index == 0:
                continue
            previous = rows[index-1]
            dt = row["observation"].timestamp - previous["observation"].timestamp
            if dt <= 0 or dt > config.max_gap_seconds:
                continue
            if row["wrist"] and previous["wrist"]:
                displacement = _distance(row["wrist"],previous["wrist"])
                velocity = tuple((a-b)/dt for a,b in zip(row["wrist"],previous["wrist"]))
                row["velocity_vector"] = velocity
                row["speed"] = math.sqrt(sum(v*v for v in velocity)) / row["scale"] if row["scale"] else None
                row["wrist_displacement"] = displacement / row["scale"] if row["scale"] else None
                if row["wrist_displacement"] is not None:
                    key=f"max_{side}_wrist_displacement"
                    diagnostics[key]=max(diagnostics[key],row["wrist_displacement"])
            if index >= 2 and row.get("velocity_vector") and rows[index-1].get("velocity_vector"):
                prior_dt = previous["observation"].timestamp-rows[index-2]["observation"].timestamp
                if 0 < prior_dt <= config.max_gap_seconds and row["scale"]:
                    row["acceleration"] = math.sqrt(sum(((a-b)/dt)**2 for a,b in
                        zip(velocity,rows[index-1]["velocity_vector"]))) / row["scale"] * (2*dt/(dt+prior_dt))
            if row["elbow"] and previous["elbow"] and row["scale"]:
                row["elbow_displacement"] = _distance(row["elbow"],previous["elbow"])/row["scale"]
                row["elbow_speed"] = row["elbow_displacement"]/dt
            if row["speed"] is not None:
                diagnostics[f"max_{side}_wrist_velocity"] = max(diagnostics[f"max_{side}_wrist_velocity"],row["speed"])
            if row["acceleration"] is not None:
                diagnostics[f"max_{side}_acceleration"] = max(diagnostics[f"max_{side}_acceleration"],row["acceleration"])
        by_side[side] = rows

    diagnostics["motion_observations"] = sum(1 for i in range(len(observations)) if any(
        rows[i].get("speed") is not None for rows in by_side.values()))

    events = []
    for side, rows in by_side.items():
        for index,row in enumerate(rows):
            row["extension_change"] = None
            if row.get("extension") is None or row.get("speed") is None:
                continue
            t=row["observation"].timestamp
            rests=[prior for prior in rows[:index] if prior.get("extension") is not None
                   and prior.get("speed") is not None and prior["speed"]<=config.max_rest_velocity
                   and 0<t-prior["observation"].timestamp<=config.max_burst_duration]
            if rests:
                row["extension_change"]=row["extension"]-rests[-1]["extension"]
                diagnostics["max_extension_change"]=max(diagnostics["max_extension_change"],row["extension_change"])
        for peak_index,row in enumerate(rows):
            speed, acceleration = row.get("speed"), row.get("acceleration")
            if speed is None or acceleration is None or speed < config.min_velocity or acceleration < config.min_acceleration:
                continue
            peak_t = row["observation"].timestamp
            prior = [i for i in range(peak_index) if rows[i].get("extension") is not None
                     and 0 < peak_t-rows[i]["observation"].timestamp <= config.max_burst_duration
                     and rows[i].get("speed") is not None and rows[i]["speed"] <= config.max_rest_velocity]
            if not prior:
                continue
            start_index = prior[-1]
            extension0 = rows[start_index]["extension"]
            burst_rows = [r for r in rows[start_index:peak_index+1] if r.get("extension") is not None]
            peak_extension = max((r["extension"] for r in burst_rows), default=extension0)
            extension_change = peak_extension-extension0
            row["extension_change"] = extension_change
            diagnostics["max_extension_change"] = max(diagnostics["max_extension_change"],extension_change)
            if extension_change < config.min_extension_change:
                continue
            fast_count = 0
            for r in reversed(rows[start_index:peak_index+1]):
                if r.get("speed") is None or r["speed"] < config.min_velocity:
                    break
                fast_count += 1
            if fast_count < config.min_velocity_observations:
                continue
            later = []
            for next_row in rows[peak_index+1:]:
                dt = next_row["observation"].timestamp-peak_t
                if dt > config.max_burst_duration:
                    break
                if dt > 0 and next_row.get("speed") is not None and next_row.get("extension") is not None:
                    later.append((next_row,dt))
            slowed = next(((r,dt) for r,dt in later if r["speed"] <= speed*config.deceleration_ratio),None)
            retracted = next(((r,dt) for r,dt in later if r["extension"] <= peak_extension-extension_change*.25),None)
            end_pair = slowed or retracted
            if end_pair is None:
                continue
            end_row, end_delta = end_pair
            duration=end_row["observation"].timestamp-rows[start_index]["observation"].timestamp
            if not config.min_burst_duration <= duration <= config.max_burst_duration:
                continue
            components = [min(1.0,speed/(2*config.min_velocity)),
                          min(1.0,acceleration/(2*config.min_acceleration)),
                          min(1.0,extension_change/(2*config.min_extension_change)), 1.0]
            score=sum(components)/len(components)
            if score < config.min_score:
                continue
            selected=rows[start_index:next(i for i,r in enumerate(rows) if r is end_row)+1]
            feature_rows=[]
            for item in selected:
                obs=item["observation"]
                feature_rows.append({"frame_index":obs.frame_index,"timestamp":obs.timestamp,
                    "wrist_velocity":item.get("speed"),"wrist_acceleration":item.get("acceleration"),
                    "wrist_displacement":item.get("wrist_displacement"),"elbow_displacement":item.get("elbow_displacement"),
                    "elbow_speed":item.get("elbow_speed"),
                    "arm_extension":item.get("extension"),
                    "extension_change":(item.get("extension")-extension0) if item.get("extension") is not None else None,
                    "pose_quality":obs.quality.pose_quality,"detection_confidence":obs.detection_confidence})
            events.append({"behaviour":"cmai_07_hitting","candidate_source":"motion_baseline",
                "arm_side":side,"start_timestamp":rows[start_index]["observation"].timestamp,
                "end_timestamp":end_row["observation"].timestamp,"candidate_score":min(1.0,score),
                "evidence":{"score_semantics":"heuristic_motion_score_not_probability",
                    "peak_timestamp":peak_t,"peak_wrist_velocity":speed,"peak_wrist_acceleration":acceleration,
                    "arm_extension":peak_extension,"extension_change":extension_change,
                    "burst_duration_seconds":duration,"deceleration_observed":bool(slowed),
                    "retraction_observed":bool(retracted),"arm_side":side,"observation_features":feature_rows}})
    # Suppress repeated peaks in one arm burst; keep distinct strokes separated by gaps.
    events.sort(key=lambda e:(e["start_timestamp"],e["arm_side"]))
    kept=[]
    for event in events:
        duplicate=next((old for old in reversed(kept) if old["arm_side"]==event["arm_side"]
                       and event["start_timestamp"]<=old["end_timestamp"]),None)
        if duplicate:
            if event["candidate_score"]>duplicate["candidate_score"]:
                duplicate.update(event)
        else:
            kept.append(event)
    return kept,diagnostics
