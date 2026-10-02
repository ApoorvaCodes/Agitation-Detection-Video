from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Person1Config:
    """Engineering defaults; thresholds are not clinically validated."""
    frame_sample_fps: float | None = 5.0
    yolo_model: str = "yolo11n.pt"
    yolo_confidence_threshold: float = 0.35
    yolo_iou_threshold: float = 0.50
    tracker_type: str = "bytetrack.yaml"
    tracker: str = "bytetrack"
    max_gap_frames: int = 10
    min_track_frames: int = 1
    crop_padding: float = 0.15
    window_seconds: float = 2.0
    window_overlap: float = 0.5
    low_quality_valid_fraction: float = 0.6
    pose_model_complexity: int = 1
    pose_min_detection_confidence: float = 0.5
    pose_min_tracking_confidence: float = 0.5
    min_landmark_visibility: float = 0.5
    max_temporal_gap_seconds: float = 1.0
    min_delta_time_seconds: float = 1e-3
    video_id: str | None = None

    def validate(self) -> None:
        if self.frame_sample_fps is not None and self.frame_sample_fps <= 0: raise ValueError("frame_sample_fps must be positive or None")
        if not 0 <= self.yolo_confidence_threshold <= 1: raise ValueError("confidence must be in [0,1]")
        if self.max_temporal_gap_seconds <= 0 or self.min_delta_time_seconds <= 0: raise ValueError("temporal durations must be positive")
        if self.tracker not in {"bytetrack", "botsort", "iou_fallback"}: raise ValueError("tracker must be bytetrack, botsort, or iou_fallback")
        if self.max_gap_frames < 0 or self.min_track_frames < 1: raise ValueError("invalid track limits")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Person1Config":
        try:
            import yaml
            values = yaml.safe_load(Path(path).read_text()) or {}
        except ImportError:
            values = {}
            for line in Path(path).read_text().splitlines():
                line=line.strip()
                if not line or line.startswith("#") or ":" not in line: continue
                key,value=line.split(":",1); value=value.strip()
                if value.lower() in {"null","none"}: parsed=None
                elif value.lower() in {"true","false"}: parsed=value.lower()=="true"
                else:
                    try: parsed=float(value) if "." in value else int(value)
                    except ValueError: parsed=value.strip("\"'")
                values[key.strip()]=parsed
        config = cls(**values); config.validate(); return config
