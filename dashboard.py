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

import streamlit as st

from person1.config import Person1Config
from person2.contracts import PrototypeBank
from person3.clips import candidate_reference_frame
from person3.contracts import ValidationResult
from person3.dashboard_flow import read_dashboard_results, resolve_groq_key
from person3.pipeline import validate_p2_result
from person3.qwen_validator import GroqQwenValidator
from person3.supabase_store import SupabaseStore
from cmai.bundle import load_bundle, research_override, legacy_result_bundle
from cmai.taxonomy import load_taxonomy
from cmai.results import (build_camera_result, review_event, create_evidence, export_archive, attach_machine_reviews)
from cmai.jobs import start_analysis


def clear_results():
    job = st.session_state.pop("analysis_job", None)
    if job:
        job.cancel()
    for name in ("p1", "p2", "events", "camera_result", "active_bundle", "selected_person", "matching_video", "video_sha256", "analysis_message", "clip_path", "clip_event_id"):
        st.session_state.pop(name, None)
    directory = st.session_state.pop("source_directory", None)
    if directory:
        directory.cleanup()
    st.session_state.pop("source_path", None)


def render_camera_result(p1, p2, bundle):
    if "camera_result" not in st.session_state:
        st.session_state.camera_result = build_camera_result(p1, p2, bundle,
            source_sha256=st.session_state.get("video_sha256"))
    result = st.session_state.camera_result
    taxonomy = load_taxonomy()
    names = {i.item_id: i.display_name for i in taxonomy.items}
    st.subheader("CMAI camera coverage")
    st.caption("This recording is the evidence window. No two-week caregiver frequency rating is inferred. Absence of a flag is not a confirmed negative.")
    st.dataframe([{"CMAI item": a.cmai_item_id, "Behaviour": names[a.cmai_item_id],
                   "Assessment": a.status.replace("_", " "), "Reason": a.reason}
                  for a in result.availability], use_container_width=True)
    people = [p.person_id for p in p1.persons]
    options = [None, *people] if len(people) != 1 else people
    selected_person = st.selectbox("Person / session track to review", options,
        format_func=lambda p: "Choose a person track" if p is None else p, key="selected_person") if people else None
    if len(people) > 1 and selected_person is None:
        st.info("Choose a person track before reviewing candidates. Tracks are never combined.")
    if selected_person is not None:
        coverage = next(c for c in result.coverage if c.person_id == selected_person)
        with st.expander("Track coverage and abstentions"):
            st.json(coverage.model_dump(mode="json"))
        candidates = [e for e in result.events if e.evidence.person_id == selected_person]
        if candidates:
            import plotly.graph_objects as go
            chart = go.Figure()
            for e in candidates:
                chart.add_bar(name=names[e.cmai_item_id], y=[names[e.cmai_item_id]],
                              x=[e.end_timestamp-e.start_timestamp], base=[e.start_timestamp],
                              orientation="h", text=[e.status], hovertext=[e.event_id])
            chart.update_layout(barmode="overlay", xaxis_title="Source time (seconds)")
            st.plotly_chart(chart, use_container_width=True)
            selected = st.selectbox("Select candidate", candidates,
                format_func=lambda e: f"{names[e.cmai_item_id]} · {e.start_timestamp:.1f}–{e.end_timestamp:.1f} s · {e.status}")
            source_path = st.session_state.get("source_path")
            if source_path and bundle.metadata.mode == "legacy_review":
                if not st.checkbox("I confirm this is the recording used to generate these legacy results", key="matching_video"):
                    source_path = None
            if source_path:
                # Selecting an interval seeks the source player to its evidence.
                st.video(source_path, start_time=selected.start_timestamp)
                person = next(p for p in p1.persons if p.person_id == selected_person)
                reference = candidate_reference_frame(source_path, selected, person)
                if reference:
                    st.image(reference, caption="Source frame with the selected session-local track")
                evidence_dir = Path(source_path).parent / "evidence"
                if selected.evidence.clip_status == "pending":
                    result = create_evidence(result, source_path, evidence_dir, selected_person)
                    st.session_state.camera_result = result
                    selected = next(e for e in result.events if e.event_id == selected.event_id)
                if selected.evidence.clip_status == "available":
                    clip = evidence_dir / Path(selected.evidence.clip_path).name
                    st.video(str(clip))
                    st.download_button("Download evidence clip", clip.read_bytes(), clip.name, "video/mp4")
                else:
                    st.warning("A clip could not be decoded; review the source at the candidate timestamp.")
            else:
                st.info("Upload the matching source video to inspect frames and create evidence clips.")
            st.json(selected.model_dump(mode="json"))
            with st.form("review_candidate"):
                reviewer = st.text_input("Reviewer", value="local-reviewer")
                decision = st.radio("Reviewer decision", ["confirmed", "rejected", "uncertain"], index=2)
                note = st.text_area("Review note")
                if st.form_submit_button("Save reviewer decision"):
                    try:
                        result = review_event(result, selected.event_id, decision, reviewer, note)
                        st.session_state.camera_result = result
                        st.success("Reviewer decision saved separately from the model candidate.")
                    except ValueError as exc:
                        st.error(str(exc))
        else:
            st.info("No reviewable candidates were produced for this track. Missing models, gaps and low-quality evidence remain unknown.")
    st.download_button("Download CMAI camera result", result.model_dump_json(indent=2), "cmai-camera-result.json", "application/json")
    source_path = st.session_state.get("source_path")
    if source_path:
        st.download_button("Download result and evidence", export_archive(result, Path(source_path).parent / "evidence"),
                           "cmai-evidence.zip", "application/zip")
    return selected_person


@st.fragment(run_every=.5)
def poll_analysis():
    job = st.session_state.get("analysis_job")
    if job is None:
        return
    status = job.status()
    if status["phase"] == "running":
        st.progress(status.get("progress", 0), text=status["message"])
        if st.button("Cancel analysis"):
            job.cancel()
            st.session_state.pop("analysis_job", None)
            st.info("Analysis cancelled; partial results discarded.")
    elif status["phase"] == "complete":
        st.session_state.p1, st.session_state.p2 = job.results()
        st.session_state.pop("analysis_job", None)
        st.session_state.analysis_message = "Video analysis completed."
        st.rerun()
    else:
        st.error(status["message"])
        st.session_state.pop("analysis_job", None)


def main():
    st.set_page_config(page_title="Video behaviour review", layout="wide")
    st.title("Video behaviour review")
    st.caption("Upload a video to analyse movement and review behaviour candidates. Research use; not a diagnosis.")
    try:
        default_bundle = load_bundle(os.getenv("CMAI_DETECTOR_BUNDLE") or None)
        load_taxonomy()
    except (ValueError, OSError) as exc:
        st.error(f"Detector/taxonomy configuration error: {exc}")
        return
    with st.sidebar:
        st.subheader("Groq verification")
        entered_key = st.text_input("Groq API key", type="password", key="groq_api_key",
                                    help="Used only for evidence verification. You can also configure GROQ_API_KEY.")
        try:
            configured_key = st.secrets.get("GROQ_API_KEY", "")
        except FileNotFoundError:
            configured_key = ""
        api_key = resolve_groq_key(entered_key, configured_key)
        model = st.text_input("Groq model", value=os.getenv("QWEN_MODEL", "qwen/qwen3-32b"), key="groq_model")
        if api_key:
            st.caption("API key configured. Connection is checked when you request verification.")
        else:
            st.info("Enter your Groq API key to enable verification. Local video analysis works without it.")
        st.caption("Verify with Groq sends selected pose/motion evidence to Groq, not the raw video.")
        st.caption(f"Detector: {default_bundle.metadata.detector_id} / {default_bundle.metadata.version} · {default_bundle.metadata.mode}")

    verifier_fingerprint = sha256((api_key + "\0" + model).encode()).hexdigest()
    if st.session_state.get("verifier_fingerprint") != verifier_fingerprint:
        st.session_state.pop("events", None)
        if "camera_result" in st.session_state:
            st.session_state.camera_result = attach_machine_reviews(st.session_state.camera_result, [])
            for event in st.session_state.camera_result.events:
                event.machine_validation = None
        st.session_state["verifier_fingerprint"] = verifier_fingerprint

    mode = st.radio("Input", ["Video", "Existing results (advanced)"], horizontal=True)
    video_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv", "webm"], key="source_video")
    p1_file = p2_file = prototypes_file = None
    if mode == "Video":
        with st.expander("Behaviour references (optional)"):
            st.write("Upload a prototype bank built from labelled training examples to detect behaviour candidates.")
            prototypes_file = st.file_uploader("Labelled behaviour prototype bank", type="json", key="prototype_bank")
            st.caption("Custom references are research overrides. Items without evaluated model assets are unavailable, not checked and absent.")
    else:
        p1_file = st.file_uploader("Person 1 result JSON", type="json", key="p1_upload")
        p2_file = st.file_uploader("Person 2 candidate result JSON", type="json", key="p2_upload")

    inputs = [mode, default_bundle.bundle_sha256, sha256(load_taxonomy().model_dump_json().encode()).hexdigest()]
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
            st.session_state.video_sha256 = sha256(video_file.getvalue()).hexdigest()

    if mode == "Video":
        if st.button("Analyse video", type="primary", disabled=video_file is None or "analysis_job" in st.session_state):
            for name in ("p1", "p2", "events", "camera_result", "selected_person"):
                st.session_state.pop(name, None)
            try:
                bank = PrototypeBank.model_validate_json(prototypes_file.getvalue()) if prototypes_file is not None else None
                bundle = research_override(bank, default_bundle.metadata.configuration) if bank is not None else default_bundle
                config = Person1Config.from_yaml(ROOT / "configs/default.yaml")
                config = replace(config, video_id="video-" + sha256(video_file.getvalue()).hexdigest()[:16])
                st.session_state.active_bundle = bundle
                st.session_state.analysis_job = start_analysis(st.session_state.source_path, bundle, config)
            except Exception as exc:
                st.error(f"Video analysis could not start: {exc}")
        poll_analysis()
        if video_file is None:
            st.info("Upload a video to start.")
    elif p1_file is not None and p2_file is not None:
        try:
            p1, p2 = read_dashboard_results(p1_file.getvalue(), p2_file.getvalue())
            st.session_state.active_bundle = legacy_result_bundle(p2)
            if video_file is not None and p1.video.video_id != "video-" + sha256(video_file.getvalue()).hexdigest()[:16]:
                st.warning("Legacy results do not carry the uploaded video checksum. Confirm that it is the matching recording before reviewing clips.")
            st.session_state.p1, st.session_state.p2 = p1, p2
        except Exception as exc:
            st.error(f"Results could not be loaded: {exc}")
    else:
        st.info("Upload matching P1 and P2 results. A video is optional in this advanced mode.")

    if "p1" not in st.session_state or "p2" not in st.session_state:
        return
    p1, p2 = st.session_state.p1, st.session_state.p2
    candidate_count = sum(len(p.events) for p in p2.persons)
    if st.session_state.get("analysis_message"):
        st.success(st.session_state.pop("analysis_message"))
    st.write(f"Tracked people: {len(p1.persons)} · Behaviour candidates: {candidate_count}")
    if mode == "Video":
        st.download_button("Download perception results", p1.model_dump_json(indent=2), "perception.json", "application/json")
        st.download_button("Download candidate results", p2.model_dump_json(indent=2), "candidates.json", "application/json")
        if not st.session_state.active_bundle.metadata.rules:
            st.info("Movement analysis is ready. No evaluated behaviour models are configured; no labels were generated.")
    selected_person = render_camera_result(p1, p2, st.session_state.active_bundle)
    selected_count = sum(len(p.events) for p in p2.persons if p.person_id == selected_person)
    if st.button("Verify candidates with Groq", disabled=not api_key or not model.strip() or selected_count == 0):
        with st.spinner("Verifying candidate evidence with Groq…"):
            verifier = GroqQwenValidator(api_key=api_key, model=model.strip())
            selected_p2 = p2.model_copy(deep=True)
            selected_p2.persons = [p for p in selected_p2.persons if p.person_id == selected_person]
            events = validate_p2_result(p1, selected_p2, verifier=verifier)
            st.session_state.events = [e.model_dump(mode="json") for e in events]
            st.session_state.camera_result = attach_machine_reviews(st.session_state.camera_result, events,
                                                 {"provider": "Groq", "model": model.strip(), "evidence_packet_schema": "1.0"})
        if any(e.verifier_error for e in events):
            st.warning("Some Groq requests failed. Those candidates remain unverified; review their reported status.")
        st.rerun()
    store = SupabaseStore()
    consent = st.checkbox("Allow saving machine-reviewed event metadata to configured Supabase", value=False)
    if store.enabled and st.button("Save event metadata to Supabase", disabled=not consent or "events" not in st.session_state):
        events = [ValidationResult.model_validate(e) for e in st.session_state.events]
        st.success("Saved." if store.save_events(p1.video.video_id, events) else "Supabase did not confirm the save.")
    elif not store.enabled:
        st.caption("Supabase persistence disabled: credentials are not configured. Reviewer decisions are included in local exports.")


if __name__ == "__main__":
    main()
