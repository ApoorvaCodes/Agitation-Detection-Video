"""Subprocess entry point; progress JSON is atomically replaced."""
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

from cmai.bundle import DetectorBundle, LoadedBundle
from cmai.detection import detect
from person1.config import Person1Config
from person1.ingestion import VideoLoader
from person1.pipeline import process_video
from person2.contracts import PrototypeBank


def run(request_path):
    directory = Path(request_path).parent
    last = 0

    def progress(phase, message, fraction=0):
        temp = directory / "progress.tmp"
        temp.write_text(json.dumps(dict(phase=phase, message=message, progress=min(1, max(0, fraction)))))
        temp.replace(directory / "progress.json")

    try:
        request = json.loads(Path(request_path).read_text())
        with VideoLoader(request["path"]) as loader:
            meta = loader.metadata
            if meta.duration_seconds is None or meta.duration_seconds <= 0:
                raise ValueError("Recording duration/frame count is unavailable; use a seekable video file.")
            if not loader.capture.read()[0]:
                raise ValueError("Recording contains no decodable video frames.")
        progress("running", f"Tracking and pose: {meta.width}×{meta.height}, {meta.fps:g} fps, {meta.duration_seconds:.1f} s")
        final_timestamp = None

        def on_frame(timestamp):
            nonlocal last, final_timestamp
            final_timestamp = timestamp
            if time.monotonic() - last > .5:
                progress("running", f"Tracking and pose at {timestamp:.1f} / {meta.duration_seconds:.1f} seconds", .85*timestamp/meta.duration_seconds)
                last = time.monotonic()

        p1 = process_video(request["path"], Person1Config(**request["p1_configuration"]), progress_callback=on_frame)
        sample_fps = request["p1_configuration"]["frame_sample_fps"] or meta.fps
        if final_timestamp is None or final_timestamp < meta.duration_seconds - max(2/sample_fps, 2/meta.fps):
            raise ValueError("Video decoding stopped before the declared recording end; partial results discarded.")
        if not p1.persons:
            progress("running", "No person tracks found; recording coverage will remain unknown.", .85)
        else:
            progress("running", "Embedding and aggregating each person track independently…", .9)
        metadata = DetectorBundle.model_validate(request["bundle"])
        bank = PrototypeBank.model_validate(request["bank"]) if request["bank"] else None
        bundle = LoadedBundle(metadata, bank, sha256(metadata.model_dump_json().encode()).hexdigest())
        p2 = detect(p1, bundle)
        (directory / "perception.json").write_text(p1.model_dump_json())
        (directory / "candidates.json").write_text(p2.model_dump_json())
        progress("complete", "Video analysis completed.", 1)
    except Exception as exc:
        progress("error", f"Video analysis could not complete: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    run(sys.argv[1])
