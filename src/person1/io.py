"""Stable serialization and loading helpers for Person 2."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from person1.contracts import Person1VideoResult

def _clean(value):
    if isinstance(value, float) and not np.isfinite(value): return None
    if isinstance(value, dict): return {k:_clean(v) for k,v in value.items()}
    if isinstance(value, list): return [_clean(v) for v in value]
    return value
def save_perception(result: Person1VideoResult, path: str|Path) -> None:
    Path(path).write_text(json.dumps(_clean(result.to_json_dict()),allow_nan=False,indent=2)+"\n")
def save_perception_jsonl(result: Person1VideoResult, path: str|Path) -> None:
    with Path(path).open("w") as output:
        output.write(json.dumps(_clean({"schema_version":result.schema_version,"video":result.video.model_dump(mode="json")}),allow_nan=False)+"\n")
        for person in result.persons:
            for observation in person.observations: output.write(json.dumps(_clean({"person_id":person.person_id,"observation":observation.model_dump(mode="json")}),allow_nan=False)+"\n")
def save_perception_npz(result: Person1VideoResult, path: str|Path) -> None:
    names=result.video.landmark_schema or []; feature_names=result.video.feature_names; rows=[]; timestamps=[]; masks=[]; features=[]; feature_masks=[]
    for person in result.persons:
        for observation in person.observations:
            timestamps.append(observation.timestamp); row=[]; mask=[]
            for name in names:
                landmark=(observation.normalized_pose.landmarks.get(name) if observation.normalized_pose else None); row.extend([landmark.x,landmark.y,landmark.z or 0,landmark.visibility or 0] if landmark else [0,0,0,0]); mask.append(bool(landmark))
            rows.append(row); masks.append(mask); values=observation.motion.feature_values if observation.motion else {}; features.append([values.get(name,np.nan) for name in feature_names]); feature_masks.append([values.get(name) is not None for name in feature_names])
    frame_count=len(timestamps); np.savez(path,landmarks_norm=np.asarray(rows,dtype=float).reshape((frame_count,len(names),4)) if names else np.empty((frame_count,0,4)),features=np.asarray(features,dtype=float).reshape((frame_count,len(feature_names))),feature_valid_masks=np.asarray(feature_masks,dtype=bool).reshape((frame_count,len(feature_names))),timestamps=np.asarray(timestamps),valid_masks=np.asarray(masks,dtype=bool).reshape((frame_count,len(names))),)
def load_perception(path: str|Path) -> Person1VideoResult:
    return Person1VideoResult.model_validate_json(Path(path).read_text())
