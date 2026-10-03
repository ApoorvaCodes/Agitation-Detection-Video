from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Person2Config:
    window_seconds: float = 2.0
    overlap: float = 0.5
    max_gap_seconds: float = 1.0
    min_frames: int = 3
    min_valid_fraction: float = 0.6
    min_shared_fraction: float = 0.5
    similarity_threshold: float = 0.8
    smoothing_alpha: float = 0.5
    min_event_seconds: float = 1.0
    pose_weight: float = 1.0
    motion_weight: float = 1.0
    video_weight: float = 1.0

    def __post_init__(self):
        if not isinstance(self.min_frames, int) or isinstance(self.min_frames, bool):
            raise ValueError("min_frames must be an integer")
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("configuration must be finite")
        if self.window_seconds <= 0 or self.max_gap_seconds <= 0 or self.min_frames < 1:
            raise ValueError("durations and min_frames must be positive")
        if not 0 <= self.overlap < 1:
            raise ValueError("overlap must be in [0,1)")
        for name in ("min_valid_fraction", "min_shared_fraction", "smoothing_alpha"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0,1]")
        if not -1 <= self.similarity_threshold <= 1 or self.min_event_seconds < 0:
            raise ValueError("invalid detection thresholds")
        weights = (self.pose_weight, self.motion_weight, self.video_weight)
        if min(weights) < 0 or max(weights) <= 0:
            raise ValueError("weights must be nonnegative with at least one positive")
