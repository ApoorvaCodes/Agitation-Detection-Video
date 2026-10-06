"""Traceable migration, local evidence and human review for CMAI candidates."""
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import math

from cmai.contracts import (CameraEvent, CameraResult, CandidateEvidence, CandidateQuality,
                            CoverageInterval, ItemAvailability, ReviewDecision, TrackCoverage)
from cmai.taxonomy import (canonical_item, load_taxonomy,
                           validate_canonical_cmai_behaviour)
from person2.embeddings import encoder_metadata


def build_camera_result(p1, p2, bundle, source_sha256=None, action_assessments=None):
    from person3.evidence import candidates_from_p2, build_evidence_packet
    from person3.validator import validate_packet
    taxonomy = load_taxonomy()
    if p1.video.video_id != p2.video_id:
        raise ValueError("P1/P2 must refer to the same recording")
    rules = {r.item_id for r in bundle.metadata.rules}
    availability = []
    for item in taxonomy.items:
        status = ("not_assessed_by_camera" if item.camera_status in {"not_camera_only", "out_of_initial_scope"}
                  else "unavailable" if item.item_id not in rules else
                  "available" if bundle.metadata.mode == "released" else "research_only")
        reason = ("Needs audio or context outside the initial camera scope." if status == "not_assessed_by_camera"
                  else "No evaluated detector asset is configured for this item." if status == "unavailable"
                  else "Custom labelled prototypes: research candidates, empirical validation pending." if status == "research_only"
                  else "Evaluated asset with recorded release approval; flags still require review.")
        availability.append(ItemAvailability(cmai_item_id=item.item_id, status=status, reason=reason))
    coverage, events = [], []
    source_people = {p.person_id: p for p in p1.persons}
    source_candidates = {c.candidate_id: c for c in candidates_from_p2(p2, p1, action_assessments)}
    for person in p2.persons:
        if person.person_id not in source_people:
            raise ValueError("P2 references an unknown person track")
        observations = source_people[person.person_id].observations
        coverage.append(TrackCoverage(person_id=person.person_id,
                        observed_frame_count=sum(not o.quality.bbox_interpolated for o in observations),
                        first_timestamp=observations[0].timestamp if observations else None,
                        last_timestamp=observations[-1].timestamp if observations else None,
                        intervals=[CoverageInterval(scores=[s.model_dump() for s in c.scores], **{k: getattr(c, k) for k in
                                   ("chunk_id", "segment_id", "start_timestamp", "end_timestamp", "valid_fraction", "status")})
                                   for c in person.chunks]))
        chunks = {c.chunk_id: c for c in person.chunks}
        for index, event in enumerate(person.events):
            try:
                item_id = event.canonical_cmai_id if event.candidate_status == "DEMO_ONLY" else canonical_item(event.behaviour, allow_legacy=True)
                item = validate_canonical_cmai_behaviour(item_id,
                    event.canonical_cmai_name if event.candidate_status == "DEMO_ONLY" else None)
            except ValueError:
                # Invalid/free-text candidates are an abstention, never a
                # user-facing behaviour label.
                continue
            selected = [chunks[cid] for cid in event.chunk_ids]
            frames = {f for c in selected for f in c.frame_indices}
            rows = [o for o in observations if o.frame_index in frames
                    and event.start_timestamp <= o.timestamp < event.end_timestamp]
            if not rows:
                raise ValueError("candidate lacks timestamped source evidence")
            event_key = f"{p1.video.video_id}:{person.person_id}:{item_id}:{event.start_timestamp}:{event.end_timestamp}"
            candidate_id = f"{person.person_id}:{event.behaviour}:{index:04d}"
            packet = build_evidence_packet(source_candidates[candidate_id], source_people[person.person_id])
            reviewable, flags = validate_packet(packet)
            contacts = []
            if action_assessments is not None:
                contacts = {c.evidence_id:c for a in action_assessments.assessments if a.person_id == person.person_id
                            and a.chunk_id in event.chunk_ids and a.item_id == item_id and a.status == "scored"
                            for c in a.contacts if c.contact == "observed" and event.start_timestamp <= c.timestamp < event.end_timestamp}
                contacts = sorted(contacts.values(),key=lambda c:(c.timestamp,c.evidence_id))
            events.append(CameraEvent(event_id=sha256(event_key.encode()).hexdigest()[:24],
                          cmai_item_id=item_id, source_label=event.behaviour,
                          canonical_cmai_id=item.item_id, canonical_cmai_name=item.display_name,
                          source_candidate_id=candidate_id,
                          start_timestamp=event.start_timestamp, end_timestamp=event.end_timestamp,
                          score=event.peak_similarity,
                          score_semantics=event.score_semantics,
                          evidence=CandidateEvidence(video_id=p1.video.video_id, person_id=person.person_id,
                                    chunk_ids=event.chunk_ids, frame_indices=sorted({o.frame_index for o in rows}), contacts=contacts,
                                    demo_evidence=event.evidence if event.candidate_status == "DEMO_ONLY" else None),
                          quality=CandidateQuality(min_valid_fraction=min(c.valid_fraction for c in selected)),
                          evidence_check={"status": "reviewable" if reviewable else "insufficient_evidence",
                                          "quality_flags": flags, "source_evidence_ids": [s.evidence_id for s in packet.segments]}))
    encoders = encoder_metadata(p2, bundle.pose_encoder())
    root = Path(__file__).resolve().parents[1]
    encoders["candidate_pipeline"] = {"model": bundle.action_model.model_identity if bundle.action_model else "cmai_masked_prototype_rules",
                                      "version": bundle.action_model.version if bundle.action_model else "1",
                                      "code_sha256": {str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest()
                                                      for p in sorted([*root.glob("person2/*.py"), *root.glob("cmai/*.py")])}}
    if bundle.action_model:
        encoders["candidate_pipeline"]["training_encoder_identity"] = bundle.action_model.encoder_identity
        encoders["candidate_pipeline"]["score_semantics"] = "positive-centroid cosine; contrast margin is positive minus negative cosine, neither is a probability"
        encoders["candidate_pipeline"]["required_video_encoder"] = bundle.metadata.video_encoder.model_dump() if bundle.metadata.video_encoder else None
    return CameraResult(contract_version="cmai-camera-result-1.1" if bundle.metadata.detector_mode == "interaction_actions" else "cmai-camera-result-1.0",
                        action_assessments=action_assessments,
                        video_id=p1.video.video_id, source_sha256=source_sha256 if bundle.metadata.mode != "legacy_review" else None,
                        taxonomy_edition=taxonomy.edition,
                        taxonomy_sha256=sha256(taxonomy.model_dump_json().encode()).hexdigest(),
                        recording_duration_seconds=p1.video.duration_seconds,
                        source_metadata={k: getattr(p1.video, k) for k in ("fps", "processed_fps", "width", "height", "frame_count", "detector_model", "tracker_type", "configuration")},
                        detector=bundle.metadata, detector_bundle_sha256=bundle.bundle_sha256,
                        encoder_metadata=encoders,
                        availability=availability, coverage=coverage, events=events,
                        limitations=["Candidates for observable actions, not diagnosis or clinical CMAI classification.",
                                     "One recording is the evidence window; no two-week caregiver frequency rating is inferred.",
                                     "No event is not a confirmed negative. Gaps and abstentions remain unknown.",
                                     "Occlusion is not measured by the current P1 contract; visibility and occlusion remain unknown.",
                                     "Session-local track IDs are not verified identities; track swaps remain possible.",
                                     "No evaluated behaviour detector is shipped; a release needs labelled data and approval."])


def review_event(result, event_id, decision, reviewer, note=""):
    raw = result.model_dump()
    matches = [e for e in raw["events"] if e["event_id"] == event_id]
    if len(matches) != 1:
        raise ValueError("unknown event ID")
    e = matches[0]
    review = ReviewDecision(decision=decision, reviewer=reviewer.strip(), note=note,
                            reviewed_at=datetime.now(timezone.utc).isoformat())
    e["review_history"].append(review.model_dump())
    e["quality"]["review_status"] = decision
    e["status"] = {"confirmed": "reviewed_confirmed", "rejected": "reviewed_rejected", "uncertain": "uncertain"}[decision]
    return CameraResult.model_validate(raw)


def attach_machine_reviews(result, reviews, verifier_identity=None):
    raw = result.model_dump()
    by_id = {r.candidate_id: r for r in reviews}
    for e in raw["events"]:
        r = by_id.get(e["source_candidate_id"])
        if r:
            e["machine_validation"] = {"verifier": verifier_identity or {"provider": "unknown", "model": "unspecified"},
                                       "result": r.model_dump(mode="json")}
    return CameraResult.model_validate(raw)


def create_evidence(result, video_path, output_dir, person_id=None):
    from person3.clips import extract_candidate_clip
    raw = result.model_dump()
    for e in raw["events"]:
        if person_id is not None and e["evidence"]["person_id"] != person_id:
            continue
        event = CameraEvent.model_validate(e)
        clip = extract_candidate_clip(video_path, event, output_dir, padding=1.0)
        if clip:
            fps = result.source_metadata["fps"]
            e["evidence"].update(clip_path=f"evidence/{clip.name}", clip_status="available",
                                  clip_start_timestamp=math.floor(max(0, e["start_timestamp"] - 1.0)*fps)/fps)
        else:
            e["evidence"].update(clip_path=None, clip_status="unavailable", clip_start_timestamp=None)
    return CameraResult.model_validate(raw)


def export_archive(result, evidence_directory):
    """JSON plus available clips, no automatic external storage."""
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        archive.writestr("result.json", result.model_dump_json(indent=2))
        for e in result.events:
            if e.evidence.clip_status == "available":
                path = Path(evidence_directory) / Path(e.evidence.clip_path).name
                if not path.is_file():
                    raise ValueError(f"evidence clip missing: {path.name}")
                archive.write(path, e.evidence.clip_path)
    return output.getvalue()
