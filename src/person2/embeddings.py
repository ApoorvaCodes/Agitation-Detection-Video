"""Deterministic baseline descriptors and an injectable video encoder boundary."""
from hashlib import sha256
import json
from importlib import import_module
from pathlib import Path
from typing import Protocol

import numpy as np

from person1.features import KEY_JOINTS
from person2.contracts import Embedding


POSE_NAMES = [f"{joint}.{axis}.{stat}" for joint in KEY_JOINTS
              for axis in ("x", "y") for stat in ("mean", "std", "mean_velocity")]


def space_id(kind, names, settings=None):
    digest = sha256(json.dumps([names, settings], sort_keys=True).encode()).hexdigest()[:16]
    return f"{kind}:{digest}"


def packed(space, values):
    valid = [bool(v is not None and np.isfinite(v)) for v in values]
    return Embedding(space=space, values=[float(v) if ok else 0.0 for v, ok in zip(values, valid)],
                     valid=valid)


def pose_embedding(observations):
    for a, b in zip(observations, observations[1:]):
        if not np.isfinite(a.timestamp) or not np.isfinite(b.timestamp) or b.timestamp <= a.timestamp:
            raise ValueError("pose encoder requires finite increasing timestamps")
    values = []
    for joint in KEY_JOINTS:
        for axis in ("x", "y"):
            samples = []
            velocities = []
            previous = None
            for o in observations:
                point = o.normalized_pose.landmarks.get(joint) if o.normalized_pose else None
                ok = usable_pose(o, joint)
                if ok:
                    value = getattr(point, axis)
                    samples.append(value)
                    if previous is not None:
                        velocities.append((value - previous[1]) / (o.timestamp - previous[0]))
                    previous = (o.timestamp, value)
                else:
                    previous = None
            values.extend([float(np.mean(samples)) if samples else None,
                           float(np.std(samples)) if len(samples) >= 2 else None,
                           float(np.mean(velocities)) if velocities else None])
    return packed(space_id("pose_stats_v1", POSE_NAMES), values)


def usable_pose(o, joint):
    point = o.normalized_pose.landmarks.get(joint) if o.normalized_pose else None
    return (point is not None and o.quality.pose_detected and not o.quality.bbox_interpolated
            and o.quality.landmark_validity.get(joint, True)
            and np.isfinite(point.x) and np.isfinite(point.y))


class StatsPoseEncoder:
    feature_names = POSE_NAMES
    identity = {"model": "pose_stats", "version": "1", "preprocessing": "body_relative_xy_masked"}

    def encode(self, observations):
        return pose_embedding(observations)


class TemporalPoseEncoder:
    """Ordered time-bin pose means, without interpolating missing observations."""
    def __init__(self, window_seconds=2.0, bins=4):
        if not np.isfinite(window_seconds) or window_seconds <= 0 or not isinstance(bins, int) or bins < 2:
            raise ValueError("temporal duration must be positive and bins an integer >= 2")
        self.window_seconds, self.bins = window_seconds, bins
        self.feature_names = [f"{joint}.{axis}.time_bin_{i}" for joint in KEY_JOINTS
                              for axis in ("x", "y") for i in range(bins)]
        self.identity = {"model": "pose_time_bins", "version": "1", "bins": bins,
                         "window_seconds": window_seconds, "preprocessing": "masked_means_no_interpolation"}

    def encode(self, observations):
        anchor = observations[0].timestamp if observations else 0
        buckets = [[] for _ in range(self.bins)]
        previous = None
        for o in observations:
            if not np.isfinite(o.timestamp) or (previous is not None and o.timestamp <= previous):
                raise ValueError("temporal encoder requires finite increasing timestamps")
            index = int(np.floor((o.timestamp - anchor) * self.bins / self.window_seconds))
            if not 0 <= index < self.bins:
                raise ValueError("observation outside temporal encoder window")
            buckets[index].append(o)
            previous = o.timestamp
        values = []
        for joint in KEY_JOINTS:
            for axis in ("x", "y"):
                for bucket in buckets:
                    samples = [getattr(o.normalized_pose.landmarks[joint], axis) for o in bucket
                               if usable_pose(o, joint)]
                    values.append(float(np.mean(samples)) if samples else None)
        return packed(space_id("pose_time_bins_v1", self.feature_names, self.identity), values)


def motion_embedding(observations, feature_names):
    names = [f"{name}.{stat}" for name in feature_names for stat in ("mean", "std")]
    values = []
    summary = {}
    for name in feature_names:
        samples = []
        for o in observations:
            value = o.motion.feature_values.get(name) if o.motion else None
            if (value is not None and np.isfinite(value) and not o.quality.bbox_interpolated
                    and o.quality.feature_validity.get(name, False)):
                samples.append(value)
        avg = float(np.mean(samples)) if samples else None
        std = float(np.std(samples)) if len(samples) >= 2 else None
        summary[f"{name}.mean"] = avg
        summary[f"{name}.std"] = std
        # Signed log compression limits the influence of acceleration/jerk units.
        values.extend([float(np.sign(v) * np.log1p(abs(v))) if v is not None else None
                       for v in (avg, std)])
    # Keep a fixed nonempty space even for a perception contract without features.
    return packed(space_id("motion_stats_log_v1", names), values or [None]), names or ["unavailable"], summary


class VideoEncoder(Protocol):
    """Encoder must return one vector per chunk, in a fixed, versioned space."""
    feature_names: list[str]

    def encode(self, observations) -> Embedding: ...


class RGBHistogramEncoder:
    """Local person-crop appearance baseline, not a pretrained action embedding."""
    def __init__(self, video_path, bins=8):
        if bins < 2:
            raise ValueError("bins must be at least two")
        self.video_path = str(video_path)
        self.bins = bins
        self.feature_names = [f"{channel}.bin_{i}" for channel in ("r", "g", "b") for i in range(bins)]
        self.identity = {"model": "rgb_crop_histogram", "version": "1", "bins": bins,
                         "checkpoint": None, "preprocessing": "source_bbox_rgb_0_256"}

    def encode(self, observations):
        import cv2
        capture = cv2.VideoCapture(self.video_path)
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"cannot open video: {self.video_path}")
        descriptors = []
        try:
            for o in observations:
                if o.quality.bbox_interpolated:
                    continue
                capture.set(cv2.CAP_PROP_POS_FRAMES, o.frame_index)
                ok, frame = capture.read()
                if not ok:
                    continue
                height, width = frame.shape[:2]
                b = o.bbox
                crop = frame[int(b.y_min * height):int(b.y_max * height),
                             int(b.x_min * width):int(b.x_max * width)]
                if crop.size:
                    hist = np.concatenate([np.histogram(crop[:, :, c], bins=self.bins,
                                                       range=(0, 256))[0] for c in (2, 1, 0)])
                    descriptors.append(hist / (crop.shape[0] * crop.shape[1]))
        finally:
            capture.release()
        return packed(space_id("rgb_crop_hist_v1", self.feature_names),
                      np.mean(descriptors, axis=0).tolist() if descriptors else [None] * len(self.feature_names))


class ConfiguredVideoEncoder:
    """Bind an explicit model/preprocessing identity to an injected encoder."""
    def __init__(self, encoder, identity):
        if not isinstance(identity, dict) or not all(identity.get(k) for k in ("model", "version", "preprocessing")):
            raise ValueError("encoder identity requires model, version, and preprocessing")
        json.dumps(identity, allow_nan=False)
        self.encoder = encoder
        self.identity = identity
        self.feature_names = list(encoder.feature_names)

    def encode(self, observations):
        embedding = self.encoder.encode(observations)
        return Embedding(space=space_id("configured_video_v1", self.feature_names,
                                         [embedding.space, self.identity]),
                         values=embedding.values, valid=embedding.valid)


def load_video_encoder(spec_path):
    """Load an explicitly configured local Python factory; no model downloads."""
    spec = json.loads(Path(spec_path).read_text())
    module, factory = spec["factory"].split(":", 1)
    encoder = getattr(import_module(module), factory)(**spec.get("kwargs", {}))
    return ConfiguredVideoEncoder(encoder, spec["identity"])


def encoder_metadata(result, pose_encoder=None, video_encoder=None):
    """Sidecar keeps the existing strict Person 2 JSON contract unchanged."""
    providers = {"pose": pose_encoder or StatsPoseEncoder(), "video": video_encoder}
    modalities = {}
    for name, coordinates in result.embedding_spaces.items():
        spaces = sorted({c.embeddings[name].space for p in result.persons for c in p.chunks})
        provider = providers.get(name)
        identity = getattr(provider, "identity", None) if provider else None
        if name == "motion":
            identity = {"model": "motion_stats_log", "version": "1", "preprocessing": "masked_mean_std_signed_log1p"}
        if provider and identity is None:
            identity = {"provider": f"{type(provider).__module__}:{type(provider).__qualname__}",
                        "note": "model and preprocessing identity must be encoded in returned space"}
        modalities[name] = {"identity": identity, "spaces": spaces, "feature_names": coordinates}
    return {"schema_version": "1.0", "video_id": result.video_id, "encoders": modalities,
            "fusion_spaces": sorted({c.fused_embedding.space for p in result.persons for c in p.chunks})}


def fuse(embeddings, weights):
    values, valid, signature = [], [], []
    for name, embedding in embeddings.items():
        weight = weights[name]
        vector = np.asarray(embedding.values)
        mask = np.asarray(embedding.valid)
        norm = np.linalg.norm(vector[mask])
        # Zero vectors carry no cosine evidence; missing data cannot become stillness.
        usable = mask & (norm > 1e-12) & (weight > 0)
        values.extend(np.where(usable, vector / norm * np.sqrt(weight), 0).tolist()
                      if norm > 1e-12 else [0.] * len(vector))
        valid.extend(usable.tolist())
        signature.append([name, embedding.space, len(vector), weight])
    return Embedding(space=space_id("fusion_v1", signature), values=values, valid=valid)
