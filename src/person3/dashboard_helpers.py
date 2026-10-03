"""Pure helpers shared by Streamlit and dashboard tests."""
def filter_events(events, behaviours=None, statuses=None, persons=None):
    behaviours, statuses, persons = set(behaviours or []), set(statuses or []), set(persons or [])
    return [e for e in events if (not behaviours or e.behaviour in behaviours)
            and (not statuses or e.validation_status in statuses)
            and (not persons or e.person_id in persons)]


def timeline_rows(events):
    return [{"event_id": e.event_id, "person_id": e.person_id, "behaviour": e.behaviour,
             "start": e.start_timestamp, "end": e.end_timestamp, "candidate_score": e.candidate_score,
             "status": e.validation_status} for e in events]
