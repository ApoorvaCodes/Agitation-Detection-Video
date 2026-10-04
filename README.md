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
Person 2 candidate generation without requiring result JSON uploads. Enter your
Groq API key in the password field in the sidebar to enable **Verify candidates
with Groq**; `GROQ_API_KEY` or Streamlit secrets also work. Verification sends
selected pose/motion evidence, not the raw video, only when requested.

Candidate generation requires a prototype bank from explicitly labelled
training examples. Upload it under **Behaviour references (optional)**.
Without it, video movement analysis and JSON exports still work, but no
behaviour labels are invented. **Existing results (advanced)** preserves the
P1/P2 JSON review workflow. See [Person 3 setup](docs/PERSON3.md) and
[prototype training instructions](docs/PERSON2_CONTRACT.md).
