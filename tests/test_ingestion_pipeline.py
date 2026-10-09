import cv2
import numpy as np
import pytest
from person1.config import Person1Config
from person1.ingestion import VideoLoader
from person1.perception import Detection, PoseResult
from person1.pipeline import Person1Pipeline

class FakeDetector:
    def detect(self, image): return [Detection(.1,.1,.8,.9,.9)]
class FakePose:
    def estimate(self, crop):
        return PoseResult({"left_hip":{"x":.4,"y":.6,"z":0,"visibility":1},"right_hip":{"x":.6,"y":.6,"z":0,"visibility":1},"left_shoulder":{"x":.4,"y":.3,"z":0,"visibility":1},"right_shoulder":{"x":.6,"y":.3,"z":0,"visibility":1},"left_wrist":{"x":.3,"y":.4,"z":0,"visibility":1}})
def test_loader_and_pipeline(tmp_path):
    path=tmp_path/"fixture.avi"; writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"MJPG"),10,(32,32)); [writer.write(np.zeros((32,32,3),dtype=np.uint8)) for _ in range(3)]; writer.release()
    with VideoLoader(path) as loader:
        frames=list(loader.frames(sample_fps=None)); assert [f.frame_index for f in frames]==[0,1,2]; assert frames[-1].timestamp>=frames[0].timestamp
    result=Person1Pipeline(Person1Config(frame_sample_fps=None),FakeDetector(),FakePose()).process(path)
    assert result.schema_version=="1.0" and len(result.persons)==1 and len(result.persons[0].observations)==3
    assert result.diagnostics["frames_processed"]==3
    assert result.diagnostics["frames_with_yolo_person_detection"]==3
    assert result.diagnostics["frames_with_valid_mediapipe_pose"]==3
    assert result.diagnostics["frames_with_valid_left_wrist"]==3
    obs=result.persons[0].observations
    assert obs[0].normalized_pose.landmarks["left_wrist"].x is not None
    assert obs[0].timestamp < obs[-1].timestamp

def test_mediapipe_adapter_converts_bgr_crop_to_rgb():
    from types import SimpleNamespace
    from person1.perception import MediaPipePoseEstimator
    seen=[]
    estimator=MediaPipePoseEstimator.__new__(MediaPipePoseEstimator)
    estimator.names=[]
    estimator.pose=SimpleNamespace(process=lambda image:(seen.append(image.copy()) or SimpleNamespace(pose_landmarks=None)))
    crop=np.zeros((1,1,3),dtype=np.uint8); crop[0,0]=[10,20,30]
    assert estimator.estimate(crop) is None
    assert seen[0][0,0].tolist()==[30,20,10]

def test_unset_mediapipe_confidence_does_not_become_zero():
    from types import SimpleNamespace
    from person1.perception import _optional_confidence
    landmark=SimpleNamespace(presence=0.0,HasField=lambda name:False)
    assert _optional_confidence(landmark,"presence") is None
def test_missing_file_raises():
    with pytest.raises(Exception): VideoLoader("missing.mp4")
