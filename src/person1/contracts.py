from typing import Any
from pydantic import BaseModel, ConfigDict, Field

class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
class BoundingBox(ContractModel):
    x_min: float = Field(ge=0, le=1); y_min: float = Field(ge=0, le=1); x_max: float = Field(ge=0, le=1); y_max: float = Field(ge=0, le=1); coordinate_system: str = "normalized"
class Landmark(ContractModel):
    x: float; y: float; z: float | None = None; visibility: float | None = Field(default=None, ge=0, le=1); presence: float | None = Field(default=None, ge=0, le=1)
class Vector3(ContractModel):
    x: float; y: float; z: float | None = None; magnitude: float
class MotionFeatures(ContractModel):
    displacement: Vector3 | None = None; velocity: Vector3 | None = None; acceleration: Vector3 | None = None; joint_angles: dict[str, float | None] = Field(default_factory=dict); angular_velocity: dict[str, float | None] = Field(default_factory=dict); body_center: Vector3 | None = None; landmarks: dict[str, dict[str, Vector3 | None]] = Field(default_factory=dict); feature_values: dict[str, float | None] = Field(default_factory=dict)
class MotionWindow(ContractModel):
    start_timestamp: float; end_timestamp: float; valid_fraction: float = Field(ge=0, le=1); low_quality: bool; features: dict[str, float | None] = Field(default_factory=dict); validity: dict[str, bool] = Field(default_factory=dict)
class ObservationQuality(ContractModel):
    detection_confidence: float = Field(ge=0, le=1); pose_quality: float | None = Field(default=None, ge=0, le=1); valid_landmarks: int = Field(ge=0); temporal_gap_seconds: float | None = Field(default=None, ge=0); tracking_status: str = "detected"; feature_validity: dict[str, bool] = Field(default_factory=dict); pose_detected: bool = False; landmark_validity: dict[str, bool] = Field(default_factory=dict); bbox_interpolated: bool = False
class PoseData(ContractModel):
    landmarks: dict[str, Landmark] = Field(default_factory=dict)
class PersonObservation(ContractModel):
    timestamp: float = Field(ge=0); frame_index: int = Field(ge=0); bbox: BoundingBox; detection_confidence: float = Field(ge=0, le=1); pose: PoseData | None = None; normalized_pose: PoseData | None = None; motion: MotionFeatures | None = None; quality: ObservationQuality
class TrackedPerson(ContractModel):
    person_id: str; observations: list[PersonObservation] = Field(default_factory=list); windows: list[MotionWindow] = Field(default_factory=list); statistics: dict[str, float | int | None] = Field(default_factory=dict)
class VideoMetadata(ContractModel):
    video_id: str; source_path: str; duration_seconds: float | None = None; fps: float; width: int; height: int; frame_count: int | None = None; codec: str | None = None; channels: int | None = None; processed_fps: float | None = None; detector_model: str | None = None; tracker_type: str | None = None; configuration_version: str = "1"; configuration: dict[str, Any] = Field(default_factory=dict); landmark_schema: list[str] = Field(default_factory=list); feature_names: list[str] = Field(default_factory=list); window_feature_names: list[str] = Field(default_factory=list)
class Person1VideoResult(ContractModel):
    schema_version: str = "1.0"; video: VideoMetadata; persons: list[TrackedPerson] = Field(default_factory=list)
    def to_json_dict(self) -> dict[str, Any]: return self.model_dump(mode="json", exclude_none=False)
