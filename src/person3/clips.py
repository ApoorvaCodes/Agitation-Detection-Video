"""Extract source-frame intervals for machine and human evidence review."""
from pathlib import Path
import math
import re
import cv2


def extract_candidate_clip(video_path, event, output_dir, padding=1.0):
    return _extract_clip(video_path, event.event_id, event.start_timestamp, event.end_timestamp, output_dir, padding)


def extract_event_clip(video_path, event, output_dir, padding=2.0):
    if event.validation_status != "supported" or event.start_timestamp is None:
        return None
    return _extract_clip(video_path, event.event_id, event.start_timestamp, event.end_timestamp, output_dir, padding)


def _extract_clip(video_path, event_id, start_timestamp, end_timestamp, output_dir, padding):
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", event_id)
    target = target_dir / f"{safe_id}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    writer = None
    frames_written = 0
    expected = 0
    try:
        if not cap.isOpened():
            return None
        fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or count <= 0 or width <= 0 or height <= 0:
            return None
        start = max(0, start_timestamp - max(0, padding))
        end = min(count / fps, end_timestamp + max(0, padding))
        if end <= start:
            return None
        first, last = math.floor(start * fps), min(int(count), math.ceil(end * fps))
        expected = last - first
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened() or not cap.set(cv2.CAP_PROP_POS_FRAMES, first):
            return None
        for _ in range(expected):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            frames_written += 1
    except cv2.error:
        frames_written = 0
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not expected or frames_written != expected:
            target.unlink(missing_ok=True)
    return target if expected and frames_written == expected and target.is_file() and target.stat().st_size else None


def candidate_reference_frame(video_path, event, person):
    """Show which session-local track a candidate refers to in the source."""
    rows = [o for o in person.observations if o.frame_index in event.evidence.frame_indices and not o.quality.bbox_interpolated]
    if not rows:
        return None
    observation = rows[len(rows)//2]
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, observation.frame_index)
        ok, frame = cap.read()
        if not ok:
            return None
        height, width = frame.shape[:2]
        b = observation.bbox
        cv2.rectangle(frame, (int(b.x_min*width), int(b.y_min*height)),
                      (int(b.x_max*width)-1, int(b.y_max*height)-1), (0,255,0), 2)
        cv2.putText(frame, person.person_id, (max(0,int(b.x_min*width)), max(12,int(b.y_min*height))),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,0), 1)
        ok, encoded = cv2.imencode('.png', frame)
        return encoded.tobytes() if ok else None
    finally:
        cap.release()
