import math
from collections.abc import Mapping
from statistics import mean, pstdev
import numpy as np
Point = tuple[float, float, float]

def _point(v: Mapping[str, float | None] | None) -> Point | None:
    if not v or v.get("x") is None or v.get("y") is None: return None
    return float(v["x"]), float(v["y"]), float(v.get("z") or 0)
def _distance(a: Point, b: Point) -> float: return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
def normalize_landmarks(landmarks: Mapping[str, Mapping[str, float | None]]) -> dict[str, dict[str, float | None]]:
    valid = {k:v for k,v in landmarks.items() if _point(v)}; left,right = _point(valid.get("left_hip")),_point(valid.get("right_hip")); center = tuple((a+b)/2 for a,b in zip(left,right)) if left and right else (0.,0.,0.)
    ls,rs = _point(valid.get("left_shoulder")),_point(valid.get("right_shoulder")); scale = _distance(ls,rs) if ls and rs else (_distance(left,right) if left and right else 1.) or 1.
    return {n:{**v,"x":(float(v["x"])-center[0])/scale,"y":(float(v["y"])-center[1])/scale,"z":(float(v.get("z") or 0)-center[2])/scale} for n,v in valid.items()}
def vector_features(current: Point, previous: Point | None, previous_velocity: Point | None, delta_t: float | None, min_delta_t: float) -> dict[str, dict[str,float] | None]:
    if previous is None or delta_t is None or delta_t < min_delta_t: return {"displacement":None,"velocity":None,"acceleration":None}
    d=tuple(a-b for a,b in zip(current,previous)); velocity=tuple(x/delta_t for x in d); acceleration=tuple((x-y)/delta_t for x,y in zip(velocity,previous_velocity)) if previous_velocity else None
    def pack(v): return {"x":v[0],"y":v[1],"z":v[2],"magnitude":math.sqrt(sum(x*x for x in v))} if v else None
    return {"displacement":pack(d),"velocity":pack(velocity),"acceleration":pack(acceleration)}
def joint_angle(a: Point | None,b: Point | None,c: Point | None) -> float | None:
    if not a or not b or not c: return None
    u=tuple(x-y for x,y in zip(a,b)); v=tuple(x-y for x,y in zip(c,b)); norm=math.sqrt(sum(x*x for x in u))*math.sqrt(sum(x*x for x in v))
    if norm <= 1e-12: return None
    return math.degrees(math.acos(max(-1,min(1,sum(x*y for x,y in zip(u,v))/norm))))
JOINT_TRIPLETS={"left_elbow":("left_shoulder","left_elbow","left_wrist"),"right_elbow":("right_shoulder","right_elbow","right_wrist"),"left_knee":("left_hip","left_knee","left_ankle"),"right_knee":("right_hip","right_knee","right_ankle"),"left_shoulder":("left_elbow","left_shoulder","left_hip"),"right_shoulder":("right_elbow","right_shoulder","right_hip"),"left_hip":("left_shoulder","left_hip","left_knee"),"right_hip":("right_shoulder","right_hip","right_knee")}
def joint_angles(landmarks: Mapping[str, Mapping[str,float|None]]) -> dict[str,float|None]:
    return {name:joint_angle(*(_point(landmarks.get(n)) for n in triple)) for name,triple in JOINT_TRIPLETS.items()}

KEY_JOINTS = ("nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")
ARM_JOINTS = ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist")
LEG_JOINTS = ("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")

def _pack(v: Point | None) -> dict[str, float] | None:
    if v is None: return None
    return {"x":v[0], "y":v[1], "z":v[2], "magnitude":math.sqrt(sum(x*x for x in v))}
def _mean_point(points: list[Point]) -> Point | None:
    return tuple(sum(p[i] for p in points)/len(points) for i in range(3)) if points else None
def _scalar(v: dict | None, key: str) -> float | None: return v.get(key) if v else None

def feature_names() -> list[str]:
    names=[]
    for joint in KEY_JOINTS:
        for group,fields in (("displacement",("x","y","z","magnitude")),("velocity",("x","y","z","speed")),("acceleration",("x","y","z","magnitude"))):
            names.extend(f"{joint}.{group}.{field}" for field in fields)
    names += ["left_wrist.jerk", "right_wrist.jerk", "body_centroid.jerk", "bbox_center.displacement", "body_centroid.displacement", "body_centroid.velocity.speed", "body_centroid.acceleration.magnitude"]
    names += [f"{name}.angular_velocity" for name in JOINT_TRIPLETS]
    return names

def window_feature_names() -> list[str]:
    names=["mean_speed","std_speed","min_speed","max_speed","speed_energy","mean_acceleration","std_acceleration","min_acceleration","max_acceleration","acceleration_energy","overall_motion_energy","arm_motion_energy","leg_motion_energy","wrist_speed_dominant_frequency","wrist_speed_zero_crossing_rate"]
    for prefix in ("body_centroid","left_wrist","right_wrist","bbox_center"):
        names += [f"{prefix}.cumulative_path_length",f"{prefix}.net_displacement",f"{prefix}.net_path_ratio"]
    return names

def _empty_feature_values() -> dict[str,float|None]: return {name:None for name in feature_names()}

def enrich_observations(observations, max_gap_seconds: float, min_delta_time: float) -> None:
    """Populate complete deterministic per-frame features in-place."""
    previous={}; previous_acceleration={}; previous_angles={}; previous_bbox=None
    for observation in observations:
        values=_empty_feature_values(); current={name:_point({"x":l.x,"y":l.y,"z":l.z}) for name,l in (observation.normalized_pose.landmarks.items() if observation.normalized_pose else [])}; dt=observation.timestamp-previous.get("_timestamp",observation.timestamp); usable=0 < dt >= min_delta_time and dt <= max_gap_seconds
        for joint in KEY_JOINTS:
            point=current.get(joint); old=previous.get(joint); fv=vector_features(point,old,previous.get(f"{joint}._velocity"),dt,min_delta_time) if point else {"displacement":None,"velocity":None,"acceleration":None}
            for group,fields in (("displacement",("x","y","z","magnitude")),("velocity",("x","y","z","speed")),("acceleration",("x","y","z","magnitude"))):
                item=fv[group] if usable else None
                for field in fields:
                    key=f"{joint}.{group}.{field}"
                    if item is not None: values[key]=item["magnitude"] if field in ("magnitude","speed") else item[field]
            if usable and fv["velocity"]: previous[f"{joint}._velocity"]=_point(fv["velocity"])
            if usable and fv["acceleration"]: previous_acceleration[joint]=_point(fv["acceleration"])
            elif not usable: previous.pop(f"{joint}._velocity",None); previous_acceleration.pop(joint,None)
            if point is not None: previous[joint]=point
        centroid=_mean_point([current[x] for x in ("left_hip","right_hip") if current.get(x)]) or _mean_point([current[x] for x in ("left_shoulder","right_shoulder") if current.get(x)])
        old_centroid=previous.get("_centroid"); centroid_fv=vector_features(centroid,old_centroid,previous.get("_centroid_velocity"),dt,min_delta_time) if centroid else {"displacement":None,"velocity":None,"acceleration":None}
        if centroid and usable:
            previous["_centroid_velocity"]=_point(centroid_fv["velocity"]); previous["_centroid"]=centroid
        elif centroid: previous["_centroid"]=centroid
        if centroid_fv["acceleration"] and usable: previous_acceleration["body_centroid"]=_point(centroid_fv["acceleration"])
        elif not usable: previous.pop("_centroid_velocity",None); previous_acceleration.pop("body_centroid",None)
        if centroid_fv["velocity"] and usable: values["body_centroid.velocity.speed"]=centroid_fv["velocity"]["magnitude"]
        if centroid_fv["acceleration"] and usable: values["body_centroid.acceleration.magnitude"]=centroid_fv["acceleration"]["magnitude"]
        for label,point,old in (("body_centroid",centroid,old_centroid),):
            if usable and point and old: values[f"{label}.displacement"]=_distance(point,old)
        bbox=( (observation.bbox.x_min+observation.bbox.x_max)/2, (observation.bbox.y_min+observation.bbox.y_max)/2, 0 )
        if previous_bbox is not None and usable: values["bbox_center.displacement"]=_distance(bbox,previous_bbox)
        previous_bbox=bbox
        for name,triple in JOINT_TRIPLETS.items():
            angle=joint_angle(*(current.get(x) for x in triple)); old_angle=previous_angles.get(name)
            if angle is not None and old_angle is not None and usable: values[f"{name}.angular_velocity"]=(angle-old_angle)/dt
            if angle is not None and usable: previous_angles[name]=angle
            elif not usable: previous_angles.pop(name,None)
        for joint in ("left_wrist","right_wrist","body_centroid"):
            acceleration=previous_acceleration.get(joint); old_acceleration=previous.get(f"{joint}._acceleration")
            if acceleration and old_acceleration and usable: values[f"{joint}.jerk"]=_distance(acceleration,old_acceleration)/dt
            if acceleration and usable: previous[f"{joint}._acceleration"]=acceleration
            elif not usable: previous.pop(f"{joint}._acceleration",None)
        observation.motion = observation.motion or _motion_from_values(values)
        observation.motion.feature_values=values
        observation.quality.feature_validity.update({k:v is not None for k,v in values.items()})
        previous["_timestamp"]=observation.timestamp

def _motion_from_values(values):
    from person1.contracts import MotionFeatures
    return MotionFeatures(feature_values=values)

def aggregate_windows(observations, window_seconds: float, overlap: float, low_quality_threshold: float):
    from person1.contracts import MotionWindow
    if not observations: return []
    result=[]; start=observations[0].timestamp; end=observations[-1].timestamp; step=max(window_seconds*(1-overlap),1e-9)
    while start <= end:
        selected=[o for o in observations if start <= o.timestamp < start+window_seconds or (start+window_seconds >= end and o.timestamp==end)]
        if selected:
            valid_frames=sum(any(v is not None for v in (o.motion.feature_values.values() if o.motion else [])) for o in selected); fraction=valid_frames/len(selected); speeds=[v for o in selected for k,v in (o.motion.feature_values.items() if o.motion else []) if k.endswith(".velocity.speed") and v is not None]; accels=[v for o in selected for k,v in (o.motion.feature_values.items() if o.motion else []) if k.endswith(".acceleration.magnitude") and v is not None]; wrists=[v for o in selected for k,v in (o.motion.feature_values.items() if o.motion else []) if k in ("left_wrist.velocity.speed","right_wrist.velocity.speed") and v is not None]
            def stats(items): return (mean(items) if items else None,pstdev(items) if len(items)>1 else 0.0 if items else None,min(items) if items else None,max(items) if items else None,mean([x*x for x in items]) if items else None)
            sm,ss,smin,smax,se=stats(speeds); am,ass,amin,amax,ae=stats(accels); arm_values=[x*x for o in selected for k,x in (o.motion.feature_values.items() if o.motion else []) if x is not None and k.endswith(".velocity.speed") and any(k.startswith(f"{j}.") for j in ARM_JOINTS)]; leg_values=[x*x for o in selected for k,x in (o.motion.feature_values.items() if o.motion else []) if x is not None and k.endswith(".velocity.speed") and any(k.startswith(f"{j}.") for j in LEG_JOINTS)]; values={"mean_speed":sm,"std_speed":ss,"min_speed":smin,"max_speed":smax,"speed_energy":se,"mean_acceleration":am,"std_acceleration":ass,"min_acceleration":amin,"max_acceleration":amax,"acceleration_energy":ae,"overall_motion_energy":se,"arm_motion_energy":mean(arm_values) if arm_values else None,"leg_motion_energy":mean(leg_values) if leg_values else None}
            if len(wrists)>=4:
                dt=np.median(np.diff([o.timestamp for o in selected if o.motion])); spectrum=np.abs(np.fft.rfft(np.asarray(wrists)-np.mean(wrists))); freqs=np.fft.rfftfreq(len(wrists),dt); values["wrist_speed_dominant_frequency"]=float(freqs[1+np.argmax(spectrum[1:])]) if len(freqs)>1 else None; signs=np.sign(np.asarray(wrists)-np.mean(wrists)); values["wrist_speed_zero_crossing_rate"]=float(np.count_nonzero(signs[1:]!=signs[:-1])/(selected[-1].timestamp-selected[0].timestamp)) if selected[-1].timestamp>selected[0].timestamp else None
            else: values["wrist_speed_dominant_frequency"]=None; values["wrist_speed_zero_crossing_rate"]=None
            for prefix in ("body_centroid","left_wrist","right_wrist","bbox_center"):
                positions=[]
                for o in selected:
                    if prefix=="bbox_center": positions.append(((o.bbox.x_min+o.bbox.x_max)/2,(o.bbox.y_min+o.bbox.y_max)/2,0))
                    elif prefix=="body_centroid":
                        lm=o.normalized_pose.landmarks if o.normalized_pose else {}; ps=[_point({"x":lm[n].x,"y":lm[n].y,"z":lm[n].z}) for n in ("left_hip","right_hip") if n in lm]; positions.append(_mean_point(ps))
                    else:
                        lm=o.normalized_pose.landmarks if o.normalized_pose else {}; p=lm.get(prefix); positions.append(_point({"x":p.x,"y":p.y,"z":p.z}) if p else None)
                positions=[p for p in positions if p is not None]; distances=[_distance(a,b) for a,b in zip(positions,positions[1:])]; path=sum(distances) if distances else None; net=_distance(positions[0],positions[-1]) if len(positions)>1 else None; values[f"{prefix}.cumulative_path_length"]=path; values[f"{prefix}.net_displacement"]=net; values[f"{prefix}.net_path_ratio"]=net/path if path else None
            result.append(MotionWindow(start_timestamp=start,end_timestamp=min(start+window_seconds,end),valid_fraction=fraction,low_quality=fraction<low_quality_threshold,features=values,validity={k:v is not None for k,v in values.items()}))
        start += step
    return result
