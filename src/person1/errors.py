class Person1Error(Exception):
    """Base pipeline error."""
class VideoLoadError(Person1Error): pass
class VideoMetadataError(Person1Error): pass
class ModelLoadError(Person1Error): pass
class DetectionError(Person1Error): pass
class PoseEstimationError(Person1Error): pass
