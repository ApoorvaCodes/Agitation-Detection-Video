# Video perception and temporal behaviour evidence

This repository contains Person 1 video perception and the Person 2 temporal
embedding/prototype baseline on the `person-2` branch.

## Person 2: embeddings and temporal ML baseline

Person 2 consumes the existing perception JSON and produces a versioned evidence
handoff for Person 3. It includes gap-aware overlapping chunks, masked pose and
motion descriptors, optional local video crop embeddings, labelled exemplar
prototype building, cosine similarity, temporal smoothing, and candidate events.

```sh
pip install -e '.[test]'
python -m person2.cli run --input result.perception.json --config configs/person2.json --output result.behaviours.json
python -m person2.cli validate result.behaviours.json
```

Without labelled prototypes the pipeline extracts descriptors and returns
`no_prototypes`. Supply `--prototypes prototypes.json` to score candidate
behaviours. `--video source.mp4` enables the RGB crop histogram baseline; a
learned encoder can be injected through the Python API. No trained action model
or clinically validated detector is bundled. Scores are uncalibrated similarities.
See [the Person 2 → Person 3 contract](docs/PERSON2_CONTRACT.md) for the prototype
annotation workflow, embedding ordering, quality gates, and JSON field meanings.
Encoder identities are written to an `OUTPUT.encoders.json` sidecar while the
handoff remains schema `1.0`. Use `--pose-representation time-bins` for ordered
pose bins or `--video-encoder encoder.json` for a configured local encoder.

The [experiment workflow](docs/PERSON2_EXPERIMENTS.md) compares pose-only,
motion-only, pose-plus-motion, and temporal pose-plus-motion using explicit
person/session-disjoint annotations. Run a readiness report with:

```sh
PYTHONPATH=src .venv/bin/python -m person2.cli experiment --output-dir experiments/person2-readiness
```

No labelled research dataset is available in this repository; empirical model
selection remains pending. The readiness report contains null metrics.
The experiment documentation includes tested dependency pins and the source-path
setup needed if editable installation does not expose `src` in your environment.

## Person 1: video perception

This repository implements the video-only perception stage of a CMAI research pipeline. It measures detections, session-local tracks, pose, normalized landmarks, and motion evidence. It deliberately does **not** classify CMAI behaviours, infer clinical probabilities, or call a cloud service.

Architecture: `VideoLoader → Detector → PoseEstimator → body normalization → temporal features → versioned Pydantic contract`.

Install with `pip install -e '.[models,test]'`; model integrations are optional for unit tests. Run `python scripts/run_person1.py --input sample.mp4`. The public API is `process_video(path, config, detector, pose_estimator)`. Inject fakes for deterministic tests. CPU works by default; GPU selection is delegated to the installed Ultralytics/MediaPipe runtime.

The output is `Person1VideoResult`, schema version `1.0`: video metadata plus `persons[]`, each with a session-local `person_id` and timestamped observations. Each observation includes decoder timestamp in seconds, frame index, normalized `[0,1]` bounding box (`x_min,y_min` top-left and `x_max,y_max` bottom-right), raw image-relative pose, body-relative pose, motion, and quality flags. No raw frames or model weights are stored in JSON.

Body normalization translates to hip center and scales by shoulder width, falling back to hip width; it does not perform orientation rotation. Displacement is the difference between consecutive normalized points; velocity and acceleration use actual timestamps. Invalid values remain `null`, never zero. OpenCV uses decoder timestamps when available and falls back to `frame_index / fps`; sampling never rewrites timestamps.

Defaults in `configs/default.yaml` are engineering defaults, not clinical thresholds: 5 FPS sampling, YOLO confidence 0.35, IoU 0.50, MediaPipe confidence 0.5, maximum temporal gap 1 second, and minimum delta 1 ms. Video stays local. Common input formats are delegated to OpenCV. Malformed metadata raises typed errors. Model/runtime nondeterminism prevents claiming exact reproducibility.

`tracker: bytetrack` is the default and calls Ultralytics `model.track(..., persist=True, tracker="bytetrack.yaml")`; `botsort` is selectable. `iou_fallback` is explicit and intended for tests or environments without model dependencies. Crop pose coordinates are mapped back to full-frame normalized coordinates before serialization. Model binaries and videos are excluded by `.gitignore`.

Run tests with `pytest -q`.
