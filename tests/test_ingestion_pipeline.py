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
def test_missing_file_raises():
    with pytest.raises(Exception): VideoLoader("missing.mp4")
