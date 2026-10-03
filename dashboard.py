"""Optional Streamlit researcher dashboard: streamlit run dashboard.py"""
import json
from pathlib import Path
import streamlit as st
import pandas as pd
from person1.contracts import Person1VideoResult
from person3.clips import extract_event_clip
from person3.dashboard_helpers import filter_events, timeline_rows
from person3.pipeline import validate_p2_result
from person3.p2_adapter import read_p2_handoff
from person3.supabase_store import SupabaseStore

st.set_page_config(page_title="Behaviour evidence review", layout="wide")
st.title("Person 3 · Behaviour evidence review")
st.caption("Research review only. Qwen verifies supplied evidence; it does not diagnose.")
p1_file = st.file_uploader("Person 1 result JSON", type="json")
p2_file = st.file_uploader("Person 2 candidate result JSON", type="json")
video_file = st.file_uploader("Source video (optional)", type=["mp4", "avi", "mov", "mkv"])
if video_file and st.button("Run Person 1 perception"):
    from person1.pipeline import process_video
    source_path = Path(".person3_dashboard_source" + Path(video_file.name).suffix)
    source_path.write_bytes(video_file.getvalue())
    try:
        st.session_state.p1_json = process_video(source_path).model_dump_json()
        st.success("Person 1 processing completed.")
    except Exception as exc:
        st.error(f"Person 1 could not run: {exc}")
if (p1_file or st.session_state.get("p1_json")) and p2_file:
    p1_raw = p1_file.getvalue() if p1_file else st.session_state.p1_json
    p1 = Person1VideoResult.model_validate_json(p1_raw)
    p2 = read_p2_handoff(json.loads(p2_file.getvalue()))
    if st.button("Run Person 3 validation"):
        st.session_state.events = [e.model_dump(mode="json") for e in validate_p2_result(p1, p2)]
        st.session_state.video_id = p1.video.video_id
    if "events" in st.session_state:
        from person3.contracts import ValidationResult
        events = [ValidationResult.model_validate(x) for x in st.session_state.events]
        labels = sorted({e.behaviour for e in events}); people = sorted({e.person_id for e in events})
        c1, c2, c3 = st.columns(3)
        with c1: selected_behaviours = st.multiselect("Behaviour", labels, default=labels)
        with c2: statuses = st.multiselect("Validation status", ["supported", "unsupported", "insufficient_evidence"], default=["supported", "unsupported", "insufficient_evidence"])
        with c3: selected_people = st.multiselect("Person / session track", people, default=people)
        visible = filter_events(events, selected_behaviours, statuses, selected_people)
        rows = timeline_rows(visible)
        st.subheader("Behaviour timeline")
        if rows:
            timeline = pd.DataFrame(rows)
            try:
                st.plotly_chart(__import__("plotly.express", fromlist=["px"]).px.timeline(timeline, x_start="start", x_end="end", y="person_id", color="behaviour", hover_data=["event_id", "status", "candidate_score"]), use_container_width=True)
            except Exception:
                st.dataframe(timeline, use_container_width=True)
            st.dataframe(timeline, use_container_width=True)
            selected = st.selectbox("Select event", visible, format_func=lambda e: f"{e.behaviour} · {e.person_id} · {e.validation_status}")
            st.json(selected.model_dump(mode="json"))
            st.write("Source evidence IDs", selected.selected_evidence_ids)
            if video_file and selected.validation_status == "supported":
                temp = Path(".person3_dashboard_source" + Path(video_file.name).suffix); temp.write_bytes(video_file.getvalue())
                clip = extract_event_clip(temp, selected, Path(".person3_clips"))
                if clip: st.video(str(clip))
        else: st.info("No events match these filters.")
        store = SupabaseStore()
        if store.enabled and st.button("Save event metadata to Supabase"):
            st.success("Saved." if store.save_events(st.session_state.video_id, events) else "Supabase did not confirm the save.")
        elif not store.enabled:
            st.caption("Supabase persistence disabled: credentials are not configured.")
else:
    st.info("Upload matching P1 and P2 JSON results to begin review.")
