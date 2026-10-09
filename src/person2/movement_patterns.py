"""Experimental trajectory and pose-motion candidates for camera review.

Thresholds are engineering defaults, not clinically calibrated cut-offs.
"""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class MovementPatternConfig:
    window_seconds: float = 12.0
    min_track_seconds: float = 8.0
    min_valid_coverage: float = 0.65
    min_detection_confidence: float = 0.4
    max_gap_seconds: float = 1.5
    min_step_body_scales: float = 0.035
    min_pacing_path_body_scales: float = 2.0
    min_movement_duration_seconds: float = 4.0
    min_pacing_reversals: int = 2
    min_recurrence: float = 0.25
    pose_visibility: float = 0.5
    min_restlessness_bursts: int = 4
    min_moving_fraction: float = 0.2
    min_movement_frequency_per_second: float = 0.25
    min_pose_frames: int = 8
    min_burst_motion_body_scales: float = 0.07
    min_posture_change_body_scales: float = 0.07
    locomotion_path_ratio: float = 0.72

    def __post_init__(self):
        values = asdict(self)
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError("movement thresholds must be finite")
        if min(self.window_seconds, self.min_track_seconds, self.max_gap_seconds,
               self.min_step_body_scales, self.min_pacing_path_body_scales, self.min_movement_duration_seconds,
               self.min_burst_motion_body_scales) <= 0:
            raise ValueError("movement durations and distances must be positive")
        if not 0 < self.min_valid_coverage <= 1 or not 0 < self.min_moving_fraction <= 1 or not 0 <= self.min_detection_confidence <= 1:
            raise ValueError("coverage and persistence must be in (0,1]")
        if min(self.min_movement_frequency_per_second, self.min_posture_change_body_scales) <= 0:
            raise ValueError("movement frequency and posture-change thresholds must be positive")
        if not 0 <= self.min_recurrence <= 1 or not 0 <= self.pose_visibility <= 1:
            raise ValueError("recurrence and visibility must be in [0,1]")
        if not 0 < self.locomotion_path_ratio <= 1 or self.min_pacing_reversals < 1 or self.min_restlessness_bursts < 2 or self.min_pose_frames < 2:
            raise ValueError("invalid movement count or trajectory threshold")

    @classmethod
    def load(cls, path=None):
        import json
        from pathlib import Path
        import os
        source=Path(path or os.getenv("MOVEMENT_PATTERN_CONFIG") or
                    Path(__file__).resolve().parents[2]/"configs/movement_patterns.json")
        return cls(**json.loads(source.read_text(encoding="utf-8")))


def _center(o):
    return ((o.bbox.x_min + o.bbox.x_max) / 2, (o.bbox.y_min + o.bbox.y_max) / 2)


def _scale(o):
    return max(o.bbox.y_max - o.bbox.y_min, o.bbox.x_max - o.bbox.x_min, 1e-4)


def _usable_track(person, config):
    return [o for o in sorted(person.observations, key=lambda x: (x.timestamp, x.frame_index))
            if not o.quality.bbox_interpolated and o.quality.detection_confidence >= config.min_detection_confidence]


def _pacing(rows, person_id, config, sampling_fps=None):
    duration = rows[-1].timestamp - rows[0].timestamp if len(rows) > 1 else 0.0
    dts = [b.timestamp-a.timestamp for a,b in zip(rows, rows[1:]) if 0 < b.timestamp-a.timestamp <= config.max_gap_seconds]
    expected = (max(1, round(duration * sampling_fps) + 1) if sampling_fps and sampling_fps > 0 else
                max(1, round(duration / (sorted(dts)[len(dts)//2] if dts else max(duration, 1.0))) + 1))
    coverage = min(1.0, len(rows) / expected)
    steps, vectors, moving_intervals = [], [], []
    for a,b in zip(rows, rows[1:]):
        dt=b.timestamp-a.timestamp
        if dt <= 0 or dt > config.max_gap_seconds:
            vectors.append(None)
            continue
        ca,cb=_center(a),_center(b)
        size=(_scale(a)+_scale(b))/2
        dx=(cb[0]-ca[0])/size
        dy=(cb[1]-ca[1])/size
        distance=math.hypot(dx,dy)
        if distance >= config.min_step_body_scales:
            steps.append(distance)
            moving_intervals.append((a.timestamp,b.timestamp))
            vectors.append((dx,dy))
    measured=[v for v in vectors if v is not None]
    primary_axis=0 if sum(abs(v[0]) for v in measured)>=sum(abs(v[1]) for v in measured) else 1
    directions=[None if v is None or abs(v[primary_axis])<config.min_step_body_scales*.65
                else 1 if v[primary_axis]>0 else -1 for v in vectors]
    reversals=sum(a is not None and b is not None and a != b for a,b in zip(directions,directions[1:]))
    path=sum(steps)
    movement_duration=sum(end-start for start,end in moving_intervals)
    net=math.hypot((_center(rows[-1])[0]-_center(rows[0])[0])/_scale(rows[0]),
                   (_center(rows[-1])[1]-_center(rows[0])[1])/_scale(rows[0])) if len(rows)>1 else 0.0
    ratio=net/path if path else 0.0
    cells=[(round(_center(o)[0]/_scale(o)/0.5),round(_center(o)[1]/_scale(o)/0.5)) for o in rows]
    recurrence=max(0.0, 1-len(set(cells))/len(cells)) if cells else 0.0
    reasons=[]
    if duration < config.min_track_seconds: reasons.append("track_duration_below_minimum")
    if movement_duration < config.min_movement_duration_seconds: reasons.append("movement_duration_below_minimum")
    if coverage < config.min_valid_coverage: reasons.append("valid_track_coverage_below_minimum")
    if reversals < config.min_pacing_reversals: reasons.append("insufficient_direction_reversals")
    if path < config.min_pacing_path_body_scales: reasons.append("path_length_below_minimum")
    if recurrence < config.min_recurrence: reasons.append("path_recurrence_below_minimum")
    accepted=not reasons
    evidence={"detector":"pacing_trajectory_v1","experimental":True,"path_length_body_scales":path,
        "net_displacement_body_scales":net,"movement_duration_seconds":movement_duration,
        "track_duration_seconds":duration,"direction_reversals":reversals,
        "path_recurrence":recurrence,"valid_track_coverage":coverage,"valid_track_observations":len(rows),
        "source_observation_ids":[f"{person_id}:frame:{o.frame_index}" for o in rows],
        "observation_timestamps":[o.timestamp for o in rows],"acceptance_reasons":["repeated_traversal_pattern"] if accepted else [],
        "abstention_reasons":reasons}
    return ("cmai_01_pacing_aimless_wandering", rows[0].timestamp, rows[-1].timestamp,
            min(1.0, path / max(config.min_pacing_path_body_scales*3, 1e-6)), evidence) if accepted else None, evidence


def _pose_points(o, config):
    pose=o.normalized_pose
    if not pose or not o.quality.pose_detected or o.quality.bbox_interpolated:
        return {}
    result={}
    for name,p in pose.landmarks.items():
        if o.quality.landmark_validity.get(name, True) is False:
            continue
        if p.visibility is not None and p.visibility < config.pose_visibility:
            continue
        if p.presence is not None and p.presence < config.pose_visibility:
            continue
        result[name]=(p.x,p.y,p.z or 0.0)
    return result


def _restlessness(rows, person_id, config, pacing_found, locomotion_dominant):
    valid=[(o,_pose_points(o,config)) for o in rows]
    relevant={"left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist",
              "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"}
    valid=[(o,p) for o,p in valid if len(set(p)&relevant)>=2]
    duration=rows[-1].timestamp-rows[0].timestamp if len(rows)>1 else 0.0
    per_frame=[]; arm_moving=leg_moving=posture_changes=0
    arms={"left_wrist","right_wrist","left_elbow","right_elbow"}
    legs={"left_knee","right_knee","left_ankle","right_ankle"}
    for (oa,pa),(ob,pb) in zip(valid,valid[1:]):
        dt=ob.timestamp-oa.timestamp
        if dt<=0 or dt>config.max_gap_seconds:
            continue
        shared=set(pa)&set(pb)
        if not shared: continue
        diffs={j:math.dist(pa[j],pb[j]) for j in shared}
        arm=max((diffs[j] for j in shared&arms),default=0.0)
        leg=max((diffs[j] for j in shared&legs),default=0.0)
        def center(names):
            points=[pb[j] for j in names if j in shared]
            return tuple(sum(p[k] for p in points)/len(points) for k in range(3)) if points else None
        shoulders=center(("left_shoulder","right_shoulder")); hips=center(("left_hip","right_hip"))
        old_shoulders=tuple(sum(pa[j][k] for j in ("left_shoulder","right_shoulder") if j in shared)/sum(j in shared for j in ("left_shoulder","right_shoulder")) for k in range(3)) if any(j in shared for j in ("left_shoulder","right_shoulder")) else None
        old_hips=tuple(sum(pa[j][k] for j in ("left_hip","right_hip") if j in shared)/sum(j in shared for j in ("left_hip","right_hip")) for k in range(3)) if any(j in shared for j in ("left_hip","right_hip")) else None
        posture_delta=math.dist(tuple(a-b for a,b in zip(shoulders,hips)),tuple(a-b for a,b in zip(old_shoulders,old_hips))) if shoulders and hips and old_shoulders and old_hips else 0.0
        posture_changes+=int(posture_delta>=config.min_posture_change_body_scales)
        is_arm=arm>=config.min_burst_motion_body_scales
        is_leg=leg>=config.min_burst_motion_body_scales
        is_posture=posture_delta>=config.min_posture_change_body_scales
        arm_moving+=int(is_arm); leg_moving+=int(is_leg)
        per_frame.append((ob.timestamp,arm,leg,is_arm or is_leg or is_posture))
    # Count separate motion bursts, allowing a brief pause between bursts.
    bursts=0; active=False; last_moving=None
    for timestamp,arm,leg,moving in per_frame:
        if moving and (not active or last_moving is None or timestamp-last_moving>1.25):
            bursts+=1
        active=moving
        if moving: last_moving=timestamp
    moving_count=sum(int(x[3]) for x in per_frame)
    moving_fraction=moving_count/max(1,len(per_frame))
    movement_frequency=max(arm_moving,leg_moving,posture_changes)/max(duration,1e-6)
    reasons=[]
    if duration<config.min_track_seconds: reasons.append("track_duration_below_minimum")
    if len(valid)<config.min_pose_frames: reasons.append("valid_pose_frames_below_minimum")
    coverage=sum(len(set(p)&relevant) for _,p in valid)/max(1,len(rows)*len(relevant))
    if coverage<config.min_valid_coverage: reasons.append("valid_pose_coverage_below_minimum")
    if pacing_found or locomotion_dominant: reasons.append("clear_locomotion_excluded_from_restlessness")
    if bursts<config.min_restlessness_bursts: reasons.append("repeated_movement_bursts_below_minimum")
    if moving_fraction<config.min_moving_fraction: reasons.append("movement_persistence_below_minimum")
    if movement_frequency<config.min_movement_frequency_per_second: reasons.append("movement_frequency_below_minimum")
    accepted=not reasons
    evidence={"detector":"restlessness_pose_motion_v1","experimental":True,"valid_pose_frame_count":len(valid),
        "valid_landmark_coverage":coverage,"valid_pose_frame_coverage":len(valid)/max(1,len(rows)),
        "motion_observations_per_window":moving_count,"repeated_movement_burst_count":bursts,
        "movement_persistence":moving_fraction,"upper_body_movement_frequency":arm_moving/max(duration,1e-6),
        "lower_body_movement_frequency":leg_moving/max(duration,1e-6),
        "movement_frequency_per_second":movement_frequency,"posture_change_count":posture_changes,
        "track_duration_seconds":duration,"source_observation_ids":[f"{person_id}:frame:{o.frame_index}" for o,_ in valid],
        "observation_timestamps":[o.timestamp for o,_ in valid],"acceptance_reasons":["repeated_pose_motion"] if accepted else [],
        "abstention_reasons":reasons}
    return ("cmai_29_general_restlessness",valid[0][0].timestamp,valid[-1][0].timestamp,
            min(1.0,moving_fraction),evidence) if accepted else None,evidence


def analyze_movement_patterns(person, config=None, sampling_fps=None):
    """Return P2 baseline events plus diagnostics; does not infer clinical agitation."""
    config=config or MovementPatternConfig()
    rows=_usable_track(person,config)
    if not rows:
        return [],{"pacing":{"abstention_reasons":["no_valid_track_observations"]},
                    "restlessness":{"abstention_reasons":["no_valid_track_observations"]}}
    duration=rows[-1].timestamp-rows[0].timestamp
    step=config.window_seconds*.5
    windows=[]
    start=rows[0].timestamp
    while start<=rows[-1].timestamp:
        selected=[o for o in rows if start<=o.timestamp<=start+config.window_seconds]
        if len(selected)>=2:
            windows.append(selected)
        if start+config.window_seconds>=rows[-1].timestamp: break
        start+=step
    pacing_candidates=[]; rest_candidates=[]; pacing_diags=[]; rest_diags=[]
    for window in windows:
        pacing,pdiag=_pacing(window,person.person_id,config,sampling_fps)
        locomotion_dominant=(pdiag.get("path_length_body_scales",0)>=0.5 and
             pdiag.get("net_displacement_body_scales",0)/max(pdiag.get("path_length_body_scales",0),1e-9)>=config.locomotion_path_ratio)
        restlessness,rdiag=_restlessness(window,person.person_id,config,pacing is not None,locomotion_dominant)
        if pacing: pacing_candidates.append(pacing)
        if restlessness: rest_candidates.append(restlessness)
        pacing_diags.append(pdiag); rest_diags.append(rdiag)
    # Overlapping windows corroborate one sustained episode; report one source interval.
    def combine(candidates):
        if not candidates:return None
        selected=max(candidates,key=lambda item:item[3])
        return selected
    pacing=combine(pacing_candidates); restlessness=combine(rest_candidates)
    pdiag=(pacing[4] if pacing else pacing_diags[0] if pacing_diags else {"abstention_reasons":["insufficient_temporal_window"]})
    rdiag=(restlessness[4] if restlessness else rest_diags[0] if rest_diags else {"abstention_reasons":["insufficient_temporal_window"]})
    pdiag={**pdiag,"evaluated_window_count":len(windows),"window_seconds":config.window_seconds}
    rdiag={**rdiag,"evaluated_window_count":len(windows),"window_seconds":config.window_seconds}
    if duration<config.min_track_seconds:
        for diagnostic in (pdiag,rdiag):
            diagnostic["abstention_reasons"]=list(dict.fromkeys([*diagnostic.get("abstention_reasons",[]),"track_duration_below_minimum"]))
    return [e for e in (pacing,restlessness) if e],{"pacing":pdiag,"restlessness":rdiag}
