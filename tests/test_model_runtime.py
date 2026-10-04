from person1.perception import _model_load_error, runtime_diagnostics, MediaPipePoseEstimator

def test_runtime_diagnostics_is_explicit():
    details=runtime_diagnostics("yolo11n.pt","bytetrack")
    assert details["python_executable"] and details["python_version"]
    assert details["model"]=="yolo11n.pt" and details["tracker"]=="bytetrack"

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
    result = estimator.estimate(object())
    assert FakePose.last == {"model_complexity": 1,
                             "min_detection_confidence": .5,
                             "min_tracking_confidence": .6}
    assert result.landmarks["nose"] == {"x": .1, "y": .2, "z": .3,
                                         "visibility": .9, "presence": .8}
