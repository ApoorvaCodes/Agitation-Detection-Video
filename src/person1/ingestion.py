from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import logging
import cv2
from person1.errors import VideoLoadError, VideoMetadataError
LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True)
class VideoMetadataRaw:
    path: str; width: int; height: int; fps: float; frame_count: int | None; duration_seconds: float | None; codec: str | None; channels: int | None
@dataclass(frozen=True)
class VideoFrame:
    image: object; frame_index: int; timestamp: float

class VideoLoader:
    """Streaming OpenCV reader; frames are never accumulated in memory."""
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file(): raise VideoLoadError(f"Video file does not exist: {self.path}")
        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened(): raise VideoLoadError(f"Unable to open video decoder: {self.path}")
        self.metadata = self._metadata(); LOGGER.info("opened video path=%s metadata=%s", self.path, self.metadata)
    def _metadata(self) -> VideoMetadataRaw:
        width, height, fps = (int(self.capture.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        if width <= 0 or height <= 0: raise VideoMetadataError(f"Invalid dimensions: {width}x{height}")
        if fps <= 0: raise VideoMetadataError(f"Invalid FPS: {fps}")
        raw_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)); count = raw_count if raw_count > 0 else None
        code = int(self.capture.get(cv2.CAP_PROP_FOURCC)); codec = "".join(chr((code >> (8*i)) & 255) for i in range(4)).strip("\x00 ") or None
        return VideoMetadataRaw(str(self.path), width, height, fps, count, count / fps if count else None, codec, 3)
    def frames(self, sample_fps: float | None = None) -> Iterator[VideoFrame]:
        step = 1 / sample_fps if sample_fps else 0; next_sample = 0.0; index = 0
        try:
            while True:
                ok, image = self.capture.read()
                if not ok: break
                decoder_ms = float(self.capture.get(cv2.CAP_PROP_POS_MSEC)); timestamp = decoder_ms / 1000 if decoder_ms > 0 else index / self.metadata.fps
                if not step or timestamp + 1e-9 >= next_sample:
                    yield VideoFrame(image, index, timestamp)
                    if step:
                        while next_sample <= timestamp + 1e-9: next_sample += step
                index += 1
        finally: self.close()
    def close(self) -> None:
        if self.capture is not None: self.capture.release()
    def __enter__(self) -> "VideoLoader": return self
    def __exit__(self, *_: object) -> None: self.close()
