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
    cap = cv2.VideoCapture(str(clip)); assert cap.isOpened()
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 10
    cap.release()


def test_candidate_evidence_exists_before_remote_verification(tmp_path):
    from test_cmai import candidates
    from cmai.results import build_camera_result, create_evidence, export_archive
    from person3.clips import candidate_reference_frame
    from zipfile import ZipFile
    from io import BytesIO
    p1, p2, bundle = candidates()
    path = tmp_path / "video.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (32,32))
    for _ in range(20):
        writer.write(np.zeros((32,32,3), dtype=np.uint8))
    writer.release()
    result = create_evidence(build_camera_result(p1,p2,bundle), path, tmp_path / "evidence")
    event = result.events[0]
    assert event.status == "candidate" and event.machine_validation is None
    assert event.evidence.clip_status == "available"
    image = candidate_reference_frame(path, event, p1.persons[0])
    assert image
    decoded = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (32,32,3)
    assert decoded[:,:,1].max() > 0  # Track overlay is visible in the reference.
    archive = ZipFile(BytesIO(export_archive(result,tmp_path / "evidence")))
    assert "result.json" in archive.namelist() and event.evidence.clip_path in archive.namelist()
