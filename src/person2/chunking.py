from dataclasses import dataclass
import math

from person1.contracts import PersonObservation
from person2.config import Person2Config


@dataclass
class TemporalChunk:
    segment_id: int
    start: float
    end: float
    observations: list[PersonObservation]


def temporal_chunks(observations, config: Person2Config, fps: float):
    """Half-open windows, anchored per continuous track segment, retaining tails."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("sampling fps must be finite and positive")
    segments = []
    for observation in observations:
        if not math.isfinite(observation.timestamp):
            raise ValueError("timestamps must be finite")
        if segments:
            previous = segments[-1][-1]
            delta = observation.timestamp - previous.timestamp
            if delta <= 0 or observation.frame_index <= previous.frame_index:
                raise ValueError("timestamps and frame indices must be strictly increasing per person")
        if not segments or delta > config.max_gap_seconds:
            segments.append([])
        segments[-1].append(observation)
    step = config.window_seconds * (1 - config.overlap)
    for segment_id, segment in enumerate(segments):
        last = segment[-1].timestamp
        support_end = last + min(1 / fps, config.max_gap_seconds)
        index = 0
        while (start := segment[0].timestamp + index * step) <= last:
            end = min(start + config.window_seconds, support_end)
            selected = [o for o in segment if start <= o.timestamp < end]
            if selected:
                yield TemporalChunk(segment_id, start, end, selected)
            index += 1
