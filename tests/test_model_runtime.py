from person1.perception import (_model_load_error, runtime_diagnostics,
                                MediaPipePoseEstimator, MediaPipeTasksPoseEstimator,
                                UltralyticsPoseEstimator,
                                MEDIAPIPE_LANDMARK_NAMES)

def test_runtime_diagnostics_is_explicit():
    details=runtime_diagnostics("yolo11n.pt","bytetrack")
    assert details["python_executable"] and details["python_version"]
    assert details["model"]=="yolo11n.pt" and details["tracker"]=="bytetrack"
    assert details["pose_backend"] == "ultralytics_pose_cpu"
    assert details["pose_delegate"] == "CPU"

def test_missing_ultralytics_error_is_actionable():
    error=ModuleNotFoundError("No module named 'ultralytics'")
    error.name="ultralytics"
    message=str(_model_load_error("yolo11n.pt","bytetrack",error))
    assert "same Python interpreter" in message
    assert "python_executable=" in message and "ultralytics_importable=" in message


def test_mediapipe_estimator_preserves_landmark_metadata(monkeypatch):
    class Landmark:
        def __init__(self, name):
            self.name = name

    class FakePose:
        last = None
        def __init__(self, **kwargs):
            FakePose.last = kwargs
        def process(self, crop):
            point = type("Point", (), {"x": .1, "y": .2, "z": .3,
                                        "visibility": .9, "presence": .8})()
            landmarks = type("Landmarks", (), {"landmark": [point]})()
            return type("Result", (), {"pose_landmarks": landmarks})()

    fake_mp = type("MediaPipe", (), {"solutions": type("Solutions", (), {
        "pose": type("PoseModule", (), {
            "Pose": FakePose,
            "PoseLandmark": [Landmark("NOSE")],
        })()
    })()})()
    monkeypatch.setitem(__import__("sys").modules, "mediapipe", fake_mp)
    estimator = MediaPipePoseEstimator(1, .5, .6)
    import numpy as np
    result = estimator.estimate(np.zeros((2, 2, 3), dtype=np.uint8))
    assert FakePose.last == {"model_complexity": 1,
                             "min_detection_confidence": .5,
                             "min_tracking_confidence": .6}
    assert result.landmarks["nose"] == {"x": .1, "y": .2, "z": .3,
                                         "visibility": .9, "presence": .8}


def test_tasks_adapter_converts_canonical_landmarks_and_closes():
    class Point:
        x, y, z, visibility, presence = .1, .2, .3, .8, .7
    class FakeLandmarker:
        closed = False
        def detect(self, image):
            return type("Result", (), {"pose_landmarks": [[Point()] * len(MEDIAPIPE_LANDMARK_NAMES)]})()
        def close(self):
            self.closed = True
    estimator = MediaPipeTasksPoseEstimator.__new__(MediaPipeTasksPoseEstimator)
    estimator.landmarker = FakeLandmarker()
    estimator._mp = type("MP", (), {
        "ImageFormat": type("ImageFormat", (), {"SRGB": "SRGB"}),
        "Image": lambda self, image_format, data: (image_format, data),
    })()
    import numpy as np
    result = estimator.estimate(np.zeros((2, 2, 3), dtype=np.uint8))
    assert list(result.landmarks) == MEDIAPIPE_LANDMARK_NAMES
    assert result.landmarks["left_wrist"]["visibility"] == .8
    estimator.close()
    assert estimator.landmarker.closed


def test_tasks_adapter_missing_asset_fails_closed(tmp_path):
    import pytest
    with pytest.raises(Exception, match="model asset is missing"):
        MediaPipeTasksPoseEstimator(tmp_path / "missing.task", .5, .5)


def test_ultralytics_pose_adapter_maps_coco_points_without_fabricating_missing_landmarks():
    import numpy as np
    class Tensor:
        def __init__(self, value): self.value = value
        def __getitem__(self, index): return Tensor(self.value[index])
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return np.asarray(self.value)
    class Keypoints:
        xy = Tensor(np.asarray([[[10, 20]] * 17], dtype=float))
        conf = Tensor(np.asarray([[.9] * 17], dtype=float))
        def __len__(self): return 1
    keypoints = Keypoints()
    result = type("Result", (), {"keypoints": keypoints})()
    estimator = UltralyticsPoseEstimator.__new__(UltralyticsPoseEstimator)
    estimator.model = type("Model", (), {"predict": lambda self, *a, **k: [result]})()
    estimator.confidence = .5
    output = estimator.estimate(np.zeros((100, 100, 3), dtype=np.uint8))
    assert output.landmarks["left_shoulder"]["visibility"] == .9
    assert "left_pinky" not in output.landmarks
