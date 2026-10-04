from dataclasses import dataclass
from typing import Any, Protocol
import importlib.metadata
import importlib.util
import sys
from person1.errors import DetectionError, ModelLoadError, PoseEstimationError

MEDIAPIPE_LANDMARK_NAMES = ["nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel", "right_heel", "left_foot_index", "right_foot_index"]

@dataclass(frozen=True)
class Detection:
    x_min: float; y_min: float; x_max: float; y_max: float; confidence: float
@dataclass(frozen=True)
class PoseResult:
    landmarks: dict[str, dict[str, float | None]]
class Detector(Protocol):
    def detect(self, image: Any) -> list[Detection]: ...
class PoseEstimator(Protocol):
    def estimate(self, crop: Any) -> PoseResult | None: ...

def runtime_diagnostics(model_name: str, tracker: str) -> dict[str, str]:
    """Return the runtime facts needed to diagnose model-loading failures."""
    try:
        version = importlib.metadata.version("ultralytics")
    except importlib.metadata.PackageNotFoundError:
        version = "not installed"
    return {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "ultralytics_importable": str(importlib.util.find_spec("ultralytics") is not None).lower(),
        "ultralytics_version": version,
        "model": model_name,
        "tracker": tracker,
    }

def _model_load_error(model_name: str, tracker: str, exc: Exception) -> ModelLoadError:
    details = runtime_diagnostics(model_name, tracker)
    if isinstance(exc, ModuleNotFoundError) and exc.name == "ultralytics":
        action = "Launch Streamlit with the same Python interpreter where Ultralytics is installed (python -m streamlit), or install the project's [models] extra in that environment."
    else:
        action = "Verify the model file/network access and the project's [models] dependency versions in this same Python environment."
    facts = ", ".join(f"{key}={value}" for key, value in details.items())
    return ModelLoadError(f"Could not load tracking model {model_name!r}: {exc}. {facts}. {action}")

class UltralyticsPersonDetector:
    def __init__(self, model_name: str, confidence: float, iou: float):
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_name)
        except Exception as exc: raise _model_load_error(model_name, "detector", exc) from exc
        self.confidence, self.iou = confidence, iou
    def detect(self, image: Any) -> list[Detection]:
        try:
            result = self.model.predict(image, conf=self.confidence, iou=self.iou, classes=[0], verbose=False)[0]; h, w = image.shape[:2]
            return [Detection(float(x1/w), float(y1/h), float(x2/w), float(y2/h), float(conf)) for (x1,y1,x2,y2), conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy())]
        except Exception as exc: raise DetectionError(f"YOLO inference failed: {exc}") from exc

class UltralyticsPersonTracker:
    """Ultralytics persistent tracking adapter using ByteTrack or BoT-SORT."""
    def __init__(self, model_name: str, confidence: float, iou: float, tracker: str):
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_name); self.confidence=confidence; self.iou=iou; self.tracker=f"{tracker}.yaml"
        except Exception as exc: raise _model_load_error(model_name, tracker, exc) from exc
    def detect_tracks(self, image: Any) -> list[tuple[str, Detection]]:
        try:
            result=self.model.track(image, persist=True, tracker=self.tracker, conf=self.confidence, iou=self.iou, classes=[0], verbose=False)[0]; h,w=image.shape[:2]
            if result.boxes.id is None: return []
            return [(f"person_{int(track_id):04d}",Detection(float(x1/w),float(y1/h),float(x2/w),float(y2/h),float(conf))) for (x1,y1,x2,y2),conf,track_id in zip(result.boxes.xyxy.cpu().numpy(),result.boxes.conf.cpu().numpy(),result.boxes.id.cpu().numpy())]
        except Exception as exc: raise DetectionError(f"Ultralytics tracking failed: {exc}") from exc

class MediaPipePoseEstimator:
    def __init__(self, model_complexity: int, min_detection_confidence: float, min_tracking_confidence: float):
        try:
            import mediapipe as mp
            self.pose = mp.solutions.pose.Pose(model_complexity=model_complexity, min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)
            self.names = [x.name.lower() for x in mp.solutions.pose.PoseLandmark]
        except Exception as exc: raise ModelLoadError(f"Could not initialize MediaPipe Pose: {exc}") from exc
    def estimate(self, crop: Any) -> PoseResult | None:
        try:
            result = self.pose.process(crop)
            if not result.pose_landmarks: return None
            return PoseResult({self.names[i] if i < len(self.names) else f"landmark_{i}": {"x": float(p.x), "y": float(p.y), "z": float(p.z), "visibility": float(getattr(p, "visibility", 0)), "presence": float(getattr(p, "presence", 0))} for i,p in enumerate(result.pose_landmarks.landmark)})
        except Exception as exc: raise PoseEstimationError(f"MediaPipe pose inference failed: {exc}") from exc
