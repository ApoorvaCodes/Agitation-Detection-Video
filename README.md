# Agitation Detection Video Research Pipeline

This repository contains a staged, research-oriented pipeline for video evidence review. Person 1 extracts person tracks, pose and motion; Person 2 generates candidate behaviour intervals; Person 3 builds compact evidence, validates candidates, and presents traceable results. None of these stages establishes a diagnosis or clinical validity.

Architecture: `Video → P1 perception → P2 candidate generation → P3 evidence packet → CMAI-aware quality checks → optional Qwen/Groq verification → deterministic Python event timestamps → evidence clips/dashboard/optional Supabase`.

Install with `pip install -e '.[models,test]'`; model integrations are optional for unit tests. Run `python scripts/run_person1.py --input sample.mp4`. The public API is `process_video(path, config, detector, pose_estimator)`. Inject fakes for deterministic tests. CPU works by default; GPU selection is delegated to the installed Ultralytics/MediaPipe runtime.

P1's `Person1VideoResult` schema 1.0 contains video metadata and `persons[]` with session-local track IDs and timestamped observations. See [docs/CONTRACT.md](docs/CONTRACT.md) for the complete contract. Person 2 is implemented on the `person-2` branch and emits `Person2VideoResult` schema 1.0; its candidate events reference supporting chunk IDs. P3 consumes that existing handoff and maps each P2 chunk's frame indices to P1 observation IDs. It does not reimplement P2.

Body normalization translates to hip center and scales by shoulder width, falling back to hip width; it does not perform orientation rotation. Displacement is the difference between consecutive normalized points; velocity and acceleration use actual timestamps. Invalid values remain `null`, never zero. OpenCV uses decoder timestamps when available and falls back to `frame_index / fps`; sampling never rewrites timestamps.

Defaults in `configs/default.yaml` are engineering defaults, not clinical thresholds: 5 FPS sampling, YOLO confidence 0.35, IoU 0.50, MediaPipe confidence 0.5, maximum temporal gap 1 second, and minimum delta 1 ms. Video stays local. Common input formats are delegated to OpenCV. Malformed metadata raises typed errors. Model/runtime nondeterminism prevents claiming exact reproducibility.

`tracker: bytetrack` is the default and calls Ultralytics `model.track(..., persist=True, tracker="bytetrack.yaml")`; `botsort` is selectable. `iou_fallback` is explicit and intended for tests or environments without model dependencies. Crop pose coordinates are mapped back to full-frame normalized coordinates before serialization. Model binaries and videos are excluded by `.gitignore`.

## Person 3

Install optional dashboard dependencies with `pip install -e '.[dashboard]'`. With P1 and P2 result JSON files available, launch `streamlit run dashboard.py`, upload the matching results, and select **Run Person 3 validation**. The dashboard supports behaviour, validation status and session-local person filters, an event table/timeline, event details, source IDs and optional video clips. To run the full P1 detector/pose models, install `pip install -e '.[models]'`; otherwise load a prepared P1 result. P2 is consumed through a versioned adapter, and the matching P2 source code is on the repository's `person-2` branch.

Configure Qwen through Groq with `GROQ_API_KEY` (required only for remote verification), `QWEN_MODEL` (defaults to `qwen/qwen3-32b`), and `QWEN_TIMEOUT_SECONDS` (defaults to 30). Without a key or on API/parser errors, P3 returns `insufficient_evidence` with a controlled verifier error; local dashboard use continues. No key is stored in source.

P3 sends only selected pose/motion features, quality flags, times and evidence IDs for the candidate interval. Qwen may select supplied evidence IDs but cannot create timestamps. Python maps selected IDs back to P1 observations and derives event bounds from their minimum and maximum source timestamps. Unknown IDs force abstention. Candidate scores remain uncalibrated P2 similarity values; they are not probabilities or clinical confidence.

Evidence clips can be generated from supported events by the `person3.clips.extract_event_clip` helper or from the selected event in the dashboard. Clips use deterministic event IDs and configurable padding; generated video is ignored by Git. Optional Supabase persistence uses `SUPABASE_URL` and `SUPABASE_KEY` and stores metadata only. The default `person3_events` table needs columns corresponding to `video_id` plus the serialized Person 3 event fields; configure a JSON/JSONB-compatible row schema and grant the client write permission. Missing credentials disable persistence.

Read [docs/PERSON3.md](docs/PERSON3.md) for handoff fields, CMAI mappings, evidence rules, deduplication, Supabase schema guidance and known limitations. Run all tests with `pytest -q`; no test calls Groq.
