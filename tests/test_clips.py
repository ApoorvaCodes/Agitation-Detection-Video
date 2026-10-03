import cv2
import numpy as np
from person3.clips import extract_event_clip
from person3.contracts import ValidationResult


def test_clip_padding_clamps_to_video_boundaries(tmp_path):
    source = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 5, (32, 32))
    for _ in range(10): writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()
    event = ValidationResult(event_id="evt_test", person_id="p", candidate_id="c", behaviour="pushing", candidate_score=.8,
        validation_status="supported", reason="ok", start_timestamp=.2, end_timestamp=.6,
        selected_evidence_ids=["p:frame:1"])
    clip = extract_event_clip(source, event, tmp_path / "clips", padding=2)
    assert clip and clip.exists() and clip.name == "evt_test.mp4"
    cap = cv2.VideoCapture(str(clip)); assert cap.isOpened(); cap.release()
