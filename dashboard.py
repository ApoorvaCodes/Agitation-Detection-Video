"""Video-first research app: .venv/bin/python -m streamlit run dashboard.py"""
from hashlib import sha256
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

# Run from a checkout without relying on an editable-install .pth.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st

from person1.config import Person1Config
from person2.config import Person2Config
from person2.contracts import PrototypeBank
from person3.clips import extract_event_clip
from person3.contracts import ValidationResult
from person3.dashboard_flow import analyze_video, read_dashboard_results, resolve_groq_key
from person3.dashboard_helpers import filter_events, timeline_rows
from person3.pipeline import validate_p2_result
from person3.qwen_validator import GroqQwenValidator
from person3.supabase_store import SupabaseStore


def clear_results():
    for name in ("p1", "p2", "events", "clip_path", "clip_event_id"):
        st.session_state.pop(name, None)
    directory = st.session_state.pop("source_directory", None)
    if directory:
        directory.cleanup()
    st.session_state.pop("source_path", None)


def render_events(p1, video_path):
    if "events" not in st.session_state:
        return
    events = [ValidationResult.model_validate(x) for x in st.session_state.events]
    st.subheader("Evidence review")
    labels = sorted({e.behaviour for e in events})
    people = sorted({e.person_id for e in events})
    c1, c2, c3 = st.columns(3)
    with c1:
        behaviours = st.multiselect("Behaviour", labels, default=labels)
    with c2:
        statuses = st.multiselect("Validation status", ["supported", "unsupported", "insufficient_evidence"],
                                  default=["supported", "unsupported", "insufficient_evidence"])
    with c3:
        selected_people = st.multiselect("Person / session track", people, default=people)
    # The helper's empty lists mean no filter; an empty UI selection means show none.
    visible = filter_events(events, behaviours, statuses, selected_people) if behaviours and statuses and selected_people else []
    if visible:
        timeline = pd.DataFrame(timeline_rows(visible))
        st.subheader("Behaviour timeline")
        import plotly.graph_objects as go
        chart = go.Figure()
        for label in sorted({e.behaviour for e in visible}):
            rows = timeline[timeline.behaviour == label]
            chart.add_bar(name=label, y=rows.person_id, x=rows.end - rows.start, base=rows.start,
                          orientation="h", text=rows.status,
                          hovertext=rows.event_id, hovertemplate="%{hovertext}<br>%{text}<extra>%{fullData.name}</extra>")
        chart.update_layout(barmode="overlay", xaxis_title="Source time (seconds)", yaxis_title="Person / session track")
        st.plotly_chart(chart, use_container_width=True)
        st.dataframe(timeline, use_container_width=True)
        selected = st.selectbox("Select event", visible,
                               format_func=lambda e: f"{e.behaviour} · {e.person_id} · {e.validation_status}")
        st.json(selected.model_dump(mode="json"))
        if video_path and selected.validation_status == "supported" and st.button("Create evidence clip"):
            clip = extract_event_clip(video_path, selected, Path(video_path).parent / "clips")
            if clip:
                st.session_state.clip_path = str(clip)
                st.session_state.clip_event_id = selected.event_id
            else:
                st.error("The evidence clip could not be created.")
        if st.session_state.get("clip_event_id") == selected.event_id:
            st.video(st.session_state.clip_path)
    else:
        st.info("No reviewed events match these filters.")
    store = SupabaseStore()
    if store.enabled and st.button("Save event metadata to Supabase"):
        st.success("Saved." if store.save_events(p1.video.video_id, events) else "Supabase did not confirm the save.")
    elif not store.enabled:
        st.caption("Supabase persistence disabled: credentials are not configured.")


def main():
    st.set_page_config(page_title="Video behaviour review", layout="wide")
    st.title("Video behaviour review")
    st.caption("Upload a video to analyse movement and review behaviour candidates. Research use; not a diagnosis.")
    with st.sidebar:
        st.subheader("Groq verification")
        entered_key = st.text_input("Groq API key", type="password", key="groq_api_key",
                                    help="Used only for evidence verification. You can also configure GROQ_API_KEY.")
        try:
            configured_key = st.secrets.get("GROQ_API_KEY", "")
        except FileNotFoundError:
            configured_key = ""
        api_key = resolve_groq_key(entered_key, configured_key)
        model = st.text_input("Groq model", value=os.getenv("QWEN_MODEL", "qwen/qwen3-32b"))
        if api_key:
            st.caption("API key configured. Connection is checked when you request verification.")
        else:
            st.info("Enter your Groq API key to enable verification. Local video analysis works without it.")
        st.caption("Verify with Groq sends selected pose/motion evidence to Groq, not the raw video.")

    verifier_fingerprint = sha256((api_key + "\0" + model).encode()).hexdigest()
    if st.session_state.get("verifier_fingerprint") != verifier_fingerprint:
        st.session_state.pop("events", None)
        st.session_state["verifier_fingerprint"] = verifier_fingerprint

    mode = st.radio("Input", ["Video", "Existing results (advanced)"], horizontal=True)
    video_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv", "webm"], key="source_video")
    p1_file = p2_file = prototypes_file = None
    if mode == "Video":
        with st.expander("Behaviour references (optional)"):
            st.write("Upload a prototype bank built from labelled training examples to detect behaviour candidates.")
            prototypes_file = st.file_uploader("Labelled behaviour prototype bank", type="json", key="prototype_bank")
            st.caption("Without references, the app analyses movement and exports results but does not invent behaviour labels.")
    else:
        p1_file = st.file_uploader("Person 1 result JSON", type="json", key="p1_upload")
        p2_file = st.file_uploader("Person 2 candidate result JSON", type="json", key="p2_upload")

    inputs = [mode]
    for uploaded in (video_file, p1_file, p2_file, prototypes_file):
        inputs.append(sha256(uploaded.getvalue()).hexdigest() if uploaded is not None else None)
    fingerprint = sha256(json.dumps(inputs).encode()).hexdigest()
    if st.session_state.get("input_fingerprint") != fingerprint:
        clear_results()
        st.session_state["input_fingerprint"] = fingerprint

    if video_file is not None:
        st.video(video_file.getvalue())
        if "source_path" not in st.session_state:
            directory = TemporaryDirectory(prefix="behaviour-review-")
            path = Path(directory.name) / ("source" + Path(video_file.name).suffix.lower())
            path.write_bytes(video_file.getvalue())
            st.session_state.source_directory = directory
            st.session_state.source_path = str(path)

    if mode == "Video":
        if st.button("Analyse video", type="primary", disabled=video_file is None):
            for name in ("p1", "p2", "events", "clip_path", "clip_event_id"):
                st.session_state.pop(name, None)
            try:
                bank = PrototypeBank.model_validate_json(prototypes_file.getvalue()) if prototypes_file is not None else None
                config = Person1Config.from_yaml(ROOT / "configs/default.yaml")
                config = replace(config, video_id="video-" + sha256(video_file.getvalue()).hexdigest()[:16])
                config2 = Person2Config(**json.loads((ROOT / "configs/person2.json").read_text()))
                with st.spinner("Analysing video: tracking, pose, motion, and behaviour candidates…"):
                    p1, p2 = analyze_video(st.session_state.source_path, bank, config, config2)
                st.session_state.p1, st.session_state.p2 = p1, p2
                st.success("Video analysis completed.")
            except Exception as exc:
                st.error(f"Video analysis could not complete: {exc}")
        if video_file is None:
            st.info("Upload a video to start.")
    elif p1_file is not None and p2_file is not None:
        try:
            p1, p2 = read_dashboard_results(p1_file.getvalue(), p2_file.getvalue())
            st.session_state.p1, st.session_state.p2 = p1, p2
        except Exception as exc:
            st.error(f"Results could not be loaded: {exc}")
    else:
        st.info("Upload matching P1 and P2 results. A video is optional in this advanced mode.")

    if "p1" not in st.session_state or "p2" not in st.session_state:
        return
    p1, p2 = st.session_state.p1, st.session_state.p2
    candidate_count = sum(len(p.events) for p in p2.persons)
    st.write(f"Tracked people: {len(p1.persons)} · Behaviour candidates: {candidate_count}")
    if mode == "Video":
        st.download_button("Download perception results", p1.model_dump_json(indent=2), "perception.json", "application/json")
        st.download_button("Download candidate results", p2.model_dump_json(indent=2), "candidates.json", "application/json")
        if prototypes_file is None:
            st.info("Movement analysis is ready. Behaviour detection needs labelled behaviour references; no labels were generated.")
        elif candidate_count == 0:
            st.info("No reviewable candidates were produced. Missing or low-quality evidence remains unknown.")
    if st.button("Verify candidates with Groq", disabled=not api_key or not model.strip() or candidate_count == 0):
        with st.spinner("Verifying candidate evidence with Groq…"):
            verifier = GroqQwenValidator(api_key=api_key, model=model.strip())
            events = validate_p2_result(p1, p2, verifier=verifier)
            st.session_state.events = [e.model_dump(mode="json") for e in events]
        if any(e.verifier_error for e in events):
            st.warning("Some Groq requests failed. Those candidates remain unverified; review their reported status.")
    render_events(p1, st.session_state.get("source_path"))


if __name__ == "__main__":
    main()
