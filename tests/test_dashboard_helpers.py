from person3.dashboard_helpers import filter_events, timeline_rows
from person3.contracts import ValidationResult
from person3.supabase_store import SupabaseStore


def event(status="supported", behaviour="pushing", person="p"):
    kw = dict(start_timestamp=0, end_timestamp=1, selected_evidence_ids=["x"]) if status == "supported" else {}
    return ValidationResult(event_id="e", person_id=person, candidate_id="c", behaviour=behaviour, candidate_score=.5,
                           validation_status=status, reason="review", **kw)


def test_filters_and_timeline_projection():
    events = [event(), event("unsupported", "hitting", "q")]
    assert filter_events(events, ["pushing"], ["supported"], ["p"])[0] == events[0]
    assert timeline_rows(events)[0]["candidate_score"] == .5


def test_supabase_is_optional_without_credentials():
    store = SupabaseStore(url="", key="")
    assert not store.enabled and store.save_events("v", []) is False
