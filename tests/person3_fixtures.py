from person1.contracts import (BoundingBox, Landmark, MotionFeatures, ObservationQuality,
    Person1VideoResult, PersonObservation, PoseData, TrackedPerson, VideoMetadata)
from person3.contracts import CandidateBehaviour


def make_source(count=4):
    obs = []
    for i in range(count):
        points = {name: Landmark(x=0.1 + i * 0.01, y=0.2, z=0, visibility=1)
                  for name in ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")}
        obs.append(PersonObservation(timestamp=float(i), frame_index=i,
            bbox=BoundingBox(x_min=0, y_min=0, x_max=1, y_max=1), detection_confidence=.9,
            normalized_pose=PoseData(landmarks=points), motion=MotionFeatures(feature_values={"left_wrist.velocity.speed": .2}),
            quality=ObservationQuality(detection_confidence=.9, valid_landmarks=len(points), pose_detected=True,
                                       feature_validity={"left_wrist.velocity.speed": True})))
    return Person1VideoResult(video=VideoMetadata(video_id="v", source_path="v.mp4", fps=1, width=16, height=16),
                              persons=[TrackedPerson(person_id="person_0001", observations=obs)])


def candidate(**overrides):
    values = dict(candidate_id="cand_001", person_id="person_0001", behaviour="pushing", candidate_score=.8,
                  start_timestamp=0, end_timestamp=3, source_window_ids=["person_0001:0"],
                  source_observation_ids=[f"person_0001:frame:{i}" for i in range(4)])
    values.update(overrides)
    return CandidateBehaviour(**values)
