"""Extract deterministic short video evidence clips around validated events."""
from pathlib import Path
import re
import cv2


def extract_event_clip(video_path, event, output_dir, padding=2.0):
    if event.validation_status != "supported" or event.start_timestamp is None:
        return None
    source = Path(video_path)
    target_dir = Path(output_dir); target_dir.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", event.event_id)
    target = target_dir / f"{safe_id}.mp4"
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        cap.release(); return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    if fps <= 0:
        cap.release(); return None
    duration = count / fps if count > 0 else max(event.end_timestamp, event.start_timestamp)
    start = max(0.0, event.start_timestamp - max(0.0, padding))
    end = min(duration, event.end_timestamp + max(0.0, padding))
    if end <= start:
        cap.release(); return None
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release(); writer.release(); return None
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    try:
        while cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 < end:
            ok, frame = cap.read()
            if not ok: break
            writer.write(frame)
    except cv2.error:
        writer.release(); cap.release(); return None
    writer.release(); cap.release()
    return target if target.exists() and target.stat().st_size else None
