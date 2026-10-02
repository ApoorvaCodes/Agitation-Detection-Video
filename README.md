# Person 1: video perception

This repository implements the video-only perception stage of a CMAI research pipeline. It measures detections, session-local tracks, pose, normalized landmarks, and motion evidence. It deliberately does **not** classify CMAI behaviours, infer clinical probabilities, or call a cloud service.

Architecture: `VideoLoader → Detector → PoseEstimator → body normalization → temporal features → versioned Pydantic contract`.

Install with `pip install -e '.[models,test]'`; model integrations are optional for unit tests. Run `python scripts/run_person1.py --input sample.mp4`. The public API is `process_video(path, config, detector, pose_estimator)`. Inject fakes for deterministic tests. CPU works by default; GPU selection is delegated to the installed Ultralytics/MediaPipe runtime.

The output is `Person1VideoResult`, schema version `1.0`: video metadata plus `persons[]`, each with a session-local `person_id` and timestamped observations. Each observation includes decoder timestamp in seconds, frame index, normalized `[0,1]` bounding box (`x_min,y_min` top-left and `x_max,y_max` bottom-right), raw image-relative pose, body-relative pose, motion, and quality flags. No raw frames or model weights are stored in JSON.

Body normalization translates to hip center and scales by shoulder width, falling back to hip width; it does not perform orientation rotation. Displacement is the difference between consecutive normalized points; velocity and acceleration use actual timestamps. Invalid values remain `null`, never zero. OpenCV uses decoder timestamps when available and falls back to `frame_index / fps`; sampling never rewrites timestamps.

Defaults in `configs/default.yaml` are engineering defaults, not clinical thresholds: 5 FPS sampling, YOLO confidence 0.35, IoU 0.50, MediaPipe confidence 0.5, maximum temporal gap 1 second, and minimum delta 1 ms. Video stays local. Common input formats are delegated to OpenCV. Malformed metadata raises typed errors. Model/runtime nondeterminism prevents claiming exact reproducibility.

`tracker: bytetrack` is the default and calls Ultralytics `model.track(..., persist=True, tracker="bytetrack.yaml")`; `botsort` is selectable. `iou_fallback` is explicit and intended for tests or environments without model dependencies. Crop pose coordinates are mapped back to full-frame normalized coordinates before serialization. Model binaries and videos are excluded by `.gitignore`.

Run tests with `pytest -q`.
