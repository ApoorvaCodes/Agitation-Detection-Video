# Agitation Detection Video Research Pipeline

This repository contains a staged, research-oriented pipeline for video evidence
review. Person 1 extracts person tracks, pose and motion; Person 2 generates
candidate behaviour intervals; Person 3 builds compact evidence, validates
candidates, and presents traceable results. None of these stages establishes a
diagnosis or clinical validity.

Architecture:

`Video → P1 perception → P2 candidate generation → P3 evidence packet → CMAI-aware quality checks → optional Qwen/Groq verification → deterministic Python event timestamps → evidence clips/dashboard/optional Supabase`


## Person 1: Video Perception

Person 1 extracts person tracks, pose landmarks, normalized body representations,
and motion features from video.

Install with:

```sh
pip install -e '.[models,test]'
```

The model extra is pinned to the validated Apple Silicon/Python 3.12 stack:
Ultralytics 8.3.0, Torch 2.3.1, torchvision 0.18.1, MediaPipe 0.10.21,
OpenCV 4.10, NumPy 1.26, protobuf 4.x, and lapx 0.10.0. Install it in the
same environment used to launch Streamlit. Check that environment with:

```sh
PYTHONPATH=src python scripts/diagnose_runtime.py
```

## Run the video dashboard

```sh
source .venv/bin/activate
python -m pip install -e '.[dashboard,models,test]'
python -m streamlit run dashboard.py
```

The `python` in both commands must be the same interpreter. The YOLO checkpoint
`yolo11n.pt` is downloaded by Ultralytics on first use when it is not supplied
as a local model path; model weights are never committed. The dashboard does
not silently fall back to IoU tracking when ByteTrack or BoT-SORT is configured.

Upload a video and click **Analyse video**. The app runs Person 1 perception and
Person 2 candidate generation without requiring result JSON uploads. If P2
produces candidates and a Groq key is configured, P3/Qwen verification starts
automatically; without a key, candidates remain explicitly unverified.
`GROQ_API_KEY` or Streamlit secrets also work. Verification sends structured
pose/motion evidence, not the raw video.

The dashboard also includes a separate demo-only physical-behaviour rule layer
(`configs/demo_physical_behaviour.json`) that derives timestamped hitting-like,
kicking-like, pacing, repetitive-mannerism, or restlessness candidates from P1
motion evidence. Candidates are visibly marked `DEMO_ONLY`; evidence strength
is not confidence, and no clinical CMAI claim is made. The rules abstain when
the measured evidence is insufficient and do not key on filenames or track IDs.

The app automatically loads `configs/cmai_detector_bundle.json`, or a local
bundle selected through `CMAI_DETECTOR_BUNDLE`. No evaluated behaviour assets are
bundled yet: the default result marks behaviours unavailable or not assessed by
camera, preserves coverage/abstentions, and does not invent labels. A prototype
upload under **Behaviour references (optional)** is an advanced research override.

Select a person track, inspect timestamped candidates and source evidence, save
reviewer decisions, and export the versioned CMAI camera JSON and evidence ZIP.
Analysis includes progress and cancellation. **Existing results (advanced)**
retains compatible P1/P2 JSON review. See [camera setup and contract](docs/CMAI_CAMERA.md),
[annotation/evaluation protocol](docs/CMAI_ANNOTATION.md), [Person 3 setup](docs/PERSON3.md),
and [Person 2 experiments](docs/PERSON2_EXPERIMENTS.md).

See [hitting/kicking candidate setup, event annotations and experiments](docs/CMAI_ACTIONS.md) for the optional supervised action path. Items 07/08 remain unavailable by default pending permitted labelled data and release evidence.
