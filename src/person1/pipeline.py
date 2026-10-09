from pathlib import Path
from typing import Any
import cv2
from person1.config import Person1Config
from person1.contracts import BoundingBox, Landmark, MotionFeatures, ObservationQuality, Person1VideoResult, PersonObservation, PoseData, TrackedPerson, Vector3, VideoMetadata
from person1.features import aggregate_windows, enrich_observations, feature_names, joint_angles, normalize_landmarks, vector_features, window_feature_names
from person1.ingestion import VideoLoader
from person1.perception import Detector, Detection, MediaPipePoseEstimator, PoseEstimator, UltralyticsPersonDetector, UltralyticsPersonTracker, MEDIAPIPE_LANDMARK_NAMES
from person1.tracking import IoUTracker

class _State:
    def __init__(self, timestamp, landmarks, velocities): self.timestamp,self.landmarks,self.velocities=timestamp,landmarks,velocities

class Person1Pipeline:
    def __init__(self, config: Person1Config|None=None, detector: Detector|None=None, pose_estimator: PoseEstimator|None=None, tracker=None):
        self.config=config or Person1Config(); self.config.validate(); self.detector=detector; self.pose_estimator=pose_estimator; self.tracker=tracker or (IoUTracker() if self.config.tracker == "iou_fallback" or detector is not None else None)
    def process(self, path: str|Path, progress_callback=None) -> Person1VideoResult:
        with VideoLoader(path) as loader:
            if self.detector is None and self.config.tracker == "iou_fallback": self.detector=UltralyticsPersonDetector(self.config.yolo_model,self.config.yolo_confidence_threshold,self.config.yolo_iou_threshold)
            if self.detector is None: self.detector=UltralyticsPersonTracker(self.config.yolo_model,self.config.yolo_confidence_threshold,self.config.yolo_iou_threshold,self.config.tracker)
            if self.pose_estimator is None: self.pose_estimator=MediaPipePoseEstimator(self.config.pose_model_complexity,self.config.pose_min_detection_confidence,self.config.pose_min_tracking_confidence)
            persons: dict[str,TrackedPerson]={}; states: dict[str,_State]={}
            diagnostics={"frames_processed":0,"frames_with_yolo_person_detection":0,
                        "person_detections":0,"frames_with_valid_mediapipe_pose":0,
                        "frames_with_valid_left_wrist":0,"frames_with_valid_right_wrist":0,
                        "frames_with_valid_left_arm_chain":0,"frames_with_valid_right_arm_chain":0}
            counted_frames={key:set() for key in diagnostics if key.startswith("frames_with_valid_")}
            for frame in loader.frames(self.config.frame_sample_fps):
                diagnostics["frames_processed"] += 1
                if progress_callback is not None:
                    progress_callback(frame.timestamp)
                detections = self.detector.detect_tracks(frame.image) if hasattr(self.detector,"detect_tracks") else [(track_id,detection) for track_id,detection in self.tracker.update(self.detector.detect(frame.image))]
                if detections:
                    diagnostics["frames_with_yolo_person_detection"] += 1
                    diagnostics["person_detections"] += len(detections)
                for track_id,detection in detections:
                    persons.setdefault(track_id,TrackedPerson(person_id=track_id))
                    crop, crop_box=_crop(frame.image,detection,self.config.crop_padding); raw_pose=self.pose_estimator.estimate(crop); raw=_map_pose(raw_pose.landmarks,crop_box,frame.image.shape) if raw_pose else {}
                    if raw:
                        counted_frames["frames_with_valid_mediapipe_pose"].add(frame.frame_index)
                    usable_raw={n:v for n,v in raw.items() if _landmark_confident(v,self.config.min_landmark_visibility)}
                    normalized=normalize_landmarks(usable_raw) if usable_raw else {}; previous=states.get(track_id); gap=frame.timestamp-previous.timestamp if previous else None
                    valid={n:_point(v) for n,v in normalized.items() if _point(v)}; motion=None; feature_validity={}
                    for side in ("left","right"):
                        if f"{side}_wrist" in valid:
                            counted_frames[f"frames_with_valid_{side}_wrist"].add(frame.frame_index)
                    for side in ("left","right"):
                        if all(f"{side}_{joint}" in valid for joint in ("shoulder","elbow","wrist")):
                            counted_frames[f"frames_with_valid_{side}_arm_chain"].add(frame.frame_index)
                    if previous and gap is not None and gap <= self.config.max_temporal_gap_seconds:
                        body=valid.get("left_hip") or (0,0,0); old=_point(previous.landmarks.get("left_hip")); values=vector_features(body,old,previous.velocities.get("left_hip"),gap,self.config.min_delta_time_seconds)
                        per_landmark={}
                        for name,point in valid.items():
                            fv=vector_features(point,_point(previous.landmarks.get(name)),previous.velocities.get(name),gap,self.config.min_delta_time_seconds); per_landmark[name]={k:_vector(v) for k,v in fv.items()}; feature_validity[f"{name}.velocity"]=fv["velocity"] is not None
                        motion=MotionFeatures(joint_angles=joint_angles(normalized),body_center=_vector(values["velocity"]),landmarks=per_landmark); feature_validity["body_center.velocity"]=values["velocity"] is not None
                    if valid:
                        velocities={};
                        for name,point in valid.items():
                            old=_point(previous.landmarks.get(name)) if previous else None; oldv=previous.velocities.get(name) if previous else None; values=vector_features(point,old,oldv,gap,self.config.min_delta_time_seconds); feature_validity[f"{name}.velocity"]=values["velocity"] is not None
                            if values["velocity"]: velocities[name]=_point(values["velocity"])
                        states[track_id]=_State(frame.timestamp,normalized,velocities)
                    bbox=BoundingBox(x_min=max(0,detection.x_min),y_min=max(0,detection.y_min),x_max=min(1,detection.x_max),y_max=min(1,detection.y_max))
                    observation=PersonObservation(timestamp=frame.timestamp,frame_index=frame.frame_index,bbox=bbox,detection_confidence=detection.confidence,pose=_pose(raw),normalized_pose=_pose(normalized,False),motion=motion,quality=ObservationQuality(detection_confidence=detection.confidence,pose_quality=len(valid)/33 if raw else None,valid_landmarks=len(valid),temporal_gap_seconds=gap,feature_validity=feature_validity,pose_detected=bool(raw),landmark_validity={n:bool(_point(v)) for n,v in normalized.items()}))
                    persons[track_id].observations.append(observation)
        metadata=loader.metadata
        diagnostics.update({key:len(indices) for key,indices in counted_frames.items()})
        config_snapshot={k:v for k,v in self.config.__dict__.items() if k != "video_id"}
        for person in persons.values():
            person.observations=_interpolate_observations(person.observations,self.config.max_gap_frames)
            enrich_observations(person.observations,self.config.max_temporal_gap_seconds,self.config.min_delta_time_seconds)
            person.windows=aggregate_windows(person.observations,self.config.window_seconds,self.config.window_overlap,self.config.low_quality_valid_fraction)
            if person.observations:
                confidences=[o.detection_confidence for o in person.observations]; interpolated=sum(o.quality.bbox_interpolated for o in person.observations); person.statistics={"first_frame":person.observations[0].frame_index,"last_frame":person.observations[-1].frame_index,"visible_frames":len(person.observations)-interpolated,"mean_detection_confidence":sum(confidences)/len(confidences),"fraction_interpolated":interpolated/len(person.observations),"duration_seconds":person.observations[-1].timestamp-person.observations[0].timestamp}
        return Person1VideoResult(video=VideoMetadata(video_id=self.config.video_id or Path(path).stem,source_path=str(path),duration_seconds=metadata.duration_seconds,fps=metadata.fps,width=metadata.width,height=metadata.height,frame_count=metadata.frame_count,codec=metadata.codec,channels=metadata.channels,processed_fps=self.config.frame_sample_fps,detector_model=self.config.yolo_model,tracker_type=self.config.tracker,configuration=config_snapshot,landmark_schema=MEDIAPIPE_LANDMARK_NAMES,feature_names=feature_names(),window_feature_names=window_feature_names()),persons=list(persons.values()),diagnostics=diagnostics)

def process_video(path: str|Path, config: Person1Config|None=None, detector: Detector|None=None, pose_estimator: PoseEstimator|None=None, progress_callback=None) -> Person1VideoResult: return Person1Pipeline(config,detector,pose_estimator).process(path, progress_callback)
def _crop(image: Any,d: Detection,padding: float=0):
    h,w=image.shape[:2]; x1=max(0,d.x_min-padding*(d.x_max-d.x_min)); y1=max(0,d.y_min-padding*(d.y_max-d.y_min)); x2=min(1,d.x_max+padding*(d.x_max-d.x_min)); y2=min(1,d.y_max+padding*(d.y_max-d.y_min)); return image[int(y1*h):int(y2*h),int(x1*w):int(x2*w)],(x1,y1,x2,y2)
def _map_pose(landmarks,box,shape):
    x1,y1,x2,y2=box; return {name:{**value,"x":x1+float(value["x"])*(x2-x1),"y":y1+float(value["y"])*(y2-y1)} for name,value in landmarks.items()}
def _point(v):
    if not v or v.get("x") is None or v.get("y") is None:return None
    return (float(v["x"]),float(v["y"]),float(v.get("z") or 0))
def _landmark_confident(value, threshold):
    return all(value.get(key) is None or float(value[key]) >= threshold for key in ("visibility","presence"))
def _vector(v): return Vector3(**v) if v else None
def _pose(values,include_visibility=True):
    return PoseData(landmarks={n:Landmark(x=float(v["x"]),y=float(v["y"]),z=v.get("z"),visibility=v.get("visibility") if include_visibility else None,presence=v.get("presence") if include_visibility else None) for n,v in values.items() if v.get("x") is not None and v.get("y") is not None})
def _interpolate_observations(observations,max_gap_frames):
    if len(observations)<2: return observations
    output=[]
    for left,right in zip(observations,observations[1:]):
        output.append(left); gap=right.frame_index-left.frame_index-1
        if 0<gap<=max_gap_frames:
            for offset in range(1,gap+1):
                ratio=offset/(gap+1); bbox=BoundingBox(x_min=left.bbox.x_min+(right.bbox.x_min-left.bbox.x_min)*ratio,y_min=left.bbox.y_min+(right.bbox.y_min-left.bbox.y_min)*ratio,x_max=left.bbox.x_max+(right.bbox.x_max-left.bbox.x_max)*ratio,y_max=left.bbox.y_max+(right.bbox.y_max-left.bbox.y_max)*ratio)
                confidence=left.detection_confidence+(right.detection_confidence-left.detection_confidence)*ratio; output.append(PersonObservation(timestamp=left.timestamp+(right.timestamp-left.timestamp)*ratio,frame_index=left.frame_index+offset,bbox=bbox,detection_confidence=confidence,quality=ObservationQuality(detection_confidence=confidence,valid_landmarks=0,tracking_status="interpolated",bbox_interpolated=True)))
    output.append(observations[-1]); return output
