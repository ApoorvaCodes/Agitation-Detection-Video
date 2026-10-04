# Camera-Based CMAI Behaviour Flagging: Implementation Plan

## Goal

Let a user upload one recording and receive timestamped flags for behaviours
that correspond to the CMAI. Each flag must point to reviewable video evidence.
The application is a video behaviour review aid; it does not diagnose agitation
or complete a caregiver-rated CMAI questionnaire.

The uploaded recording is the evidence window. Report what is visible in that
recording and its coverage. Do not convert a short recording into CMAI's
caregiver frequency rating for the preceding two weeks.

## Current repository baseline

The repository already has the main pipeline:

```text
video upload → Person 1 detection/tracking/pose/motion
             → Person 2 chunk embeddings and prototype candidate intervals
             → Person 3 validation, evidence clips, dashboard, optional storage
```

The Streamlit dashboard accepts a video and runs analysis. Person 2 currently
needs a bank of explicitly labelled behaviour prototypes to emit behaviour
labels. The starter labels documented for Person 2 are `pacing_aimless_wandering`,
`repetitious_mannerisms`, and `general_restlessness`. There is not yet a complete
CMAI camera-observability map, nor a trained or evaluated detector for every
CMAI item. This plan adds the mapping and turns the candidate workflow into an
explicit CMAI-labelled upload result.

## CMAI target and label rules

Use the 29-item CMAI long form as the initial taxonomy because it is the
29-behaviour list described in the supplied manual. Store its edition/form and
taxonomy version with every result. Do not silently mix in the community,
relatives, short, or disruptiveness forms.

Use stable machine IDs such as `cmai_01_pacing_aimless_wandering`. Keep the
CMAI item number and canonical display name in the taxonomy. Model labels are
**candidate flags**; they are not claims that an observed action was
inappropriate, intentional, clinically significant, or caused by agitation.

### Camera observability map

This is an engineering scope map, not a claim that the current model detects
these items accurately. A label may only be enabled after it has an annotation
protocol, suitable training/evaluation examples, and an evidence path.

| CMAI item | Behaviour | Camera status | Candidate evidence / key limitation |
|---|---|---|---|
| 01 | Pacing, aimless wandering | Conditional | Repeated paths or back-and-forth walking; camera cannot establish aimlessness or exclude purposeful walking without context. |
| 02 | Inappropriate dressing or disrobing | Out of initial scope | Social appropriateness needs context; do not infer nudity or sexual behaviour from appearance alone. |
| 03 | Spitting | Conditional | Visible expulsion event; may be occluded and cannot always distinguish spitting from other mouth actions. |
| 04 | Cursing or verbal aggression | Not camera-only | Requires intelligible speech/audio. |
| 05 | Unwarranted requests for attention/help | Not camera-only | Usually requires speech and context about whether a request is warranted. |
| 06 | Repetitive sentences or questions | Not camera-only | Requires speech/audio and language-level repetition. |
| 07 | Hitting | Conditional | Visible striking action and target; self-hitting needs a clear body-part interaction. |
| 08 | Kicking | Conditional | Visible forceful foot action and target/object. |
| 09 | Grabbing onto people or things inappropriately | Conditional | Visible snatching/yanking or grasp; appropriateness and ownership may be unknown. |
| 10 | Pushing | Conditional | Visible forceful contact and displacement of another person. |
| 11 | Throwing things | Conditional | Requires object tracking and evidence that an object was propelled, not merely dropped. |
| 12 | Strange noises | Not camera-only | Audio is required. |
| 13 | Screaming | Not camera-only | Audio is required. |
| 14 | Biting | Conditional | Requires visible contact; intent and target may be ambiguous. |
| 15 | Scratching | Conditional | Requires clear contact/action; small movements and occlusion make this difficult. |
| 16 | Trying to get to a different place | Conditional | Can flag repeated approach/exit attempts if room boundaries are configured; intent/context are not inherent in trajectory. |
| 17 | Intentional falling | Candidate only for fall-like event | Camera may flag a fall-like transition; it cannot establish that the fall was intentional. |
| 18 | Complaining | Not camera-only | Speech/audio and semantic interpretation are required. |
| 19 | Negativism | Not camera-only | Requires interaction context and often speech; refusal cannot be inferred from posture alone. |
| 20 | Eating/drinking inappropriate substances | Out of initial scope | Requires object identity and ingestion evidence; appropriateness depends on context. |
| 21 | Hurting self or another | Conditional | Flag only clearly visible harmful-object interaction; do not infer injury or harm from proximity alone. |
| 22 | Handling things inappropriately | Conditional | Object interaction may be visible; ownership and appropriateness need scene context. |
| 23 | Hiding things | Conditional | Requires persistent object tracking and evidence that an object was deliberately put out of sight. |
| 24 | Hoarding things | Out of initial scope | Requires longer-term context and knowledge of what is an inappropriate quantity. |
| 25 | Tearing things/destroying property | Conditional | Requires object-level evidence of destructive interaction and resulting damage. |
| 26 | Repetitious mannerisms | Camera candidate | Repeated body/object movement over time, such as rocking or tapping; repeated speech is excluded. |
| 27 | Verbal sexual advances | Not camera-only | Requires speech/audio. |
| 28 | Physical sexual advances/exposing genitals | Out of initial scope | Requires sensitive interpretation and context; do not enable automatic detection in the initial camera scope. |
| 29 | General restlessness | Camera candidate | Repeated sit/stand transitions or movement while seated, with posture visibility and duration evidence. |

`Camera candidate` means suitable for the first camera-focused model work, not
already supported by a reliable detector. `Conditional` means it may be added
only with the required scene/object/context evidence. `Not camera-only` and `Out
of initial scope` labels must not be emitted as negative findings when video is
the only input.

## User-facing flow

1. User uploads a supported video and starts analysis.
2. Validate the file, duration, frame rate, dimensions, and decodability. Show
   processing progress and fail with a useful message if it cannot be analyzed.
3. Run Person 1 and identify session-local person tracks. Let the user choose a
   person when there are multiple plausible subjects; never silently combine
   tracks.
4. Run Person 2 over the full recording using the selected taxonomy version and
   enabled behaviour detectors.
5. Run Person 3 evidence validation on candidates and create a short clip or
   frame sequence for each candidate where available.
6. Show a timeline and a list of candidate flags. Selecting a flag seeks to its
   timestamp and displays the evidence and quality/uncertainty information.
7. Allow the reviewer to mark a candidate as confirmed, rejected, or uncertain.
   Preserve model output separately from reviewer decisions.
8. Export a JSON result and evidence references. Persist only according to the
   configured local/Supabase workflow and consent settings.

No prototype bank is currently bundled. For an upload-and-flag experience, the
application must load a versioned, trained detector/prototype bank by default;
the optional prototype upload can remain an advanced override for research.
Until a behaviour has validated model assets, label it unavailable in the UI
rather than implying it was checked and not found.

## Result contract

Add a versioned CMAI camera taxonomy section to the Person 3-facing result, or
provide a strictly versioned companion contract if backwards compatibility
requires it. Each candidate event should contain:

```json
{
  "taxonomy_version": "cmai-long-form-camera-v1",
  "cmai_item_id": "cmai_29_general_restlessness",
  "status": "candidate",
  "start_timestamp": 12.4,
  "end_timestamp": 18.8,
  "score": 0.83,
  "score_semantics": "uncalibrated_model_score",
  "evidence": {
    "video_id": "recording-id",
    "person_id": "track-1",
    "chunk_ids": ["track-1:12", "track-1:13"],
    "clip_path": "evidence/track-1-cmai-29-12.4-18.8.mp4"
  },
  "quality": {
    "visibility": "adequate",
    "occlusion_fraction": 0.0,
    "review_status": "pending"
  }
}
```

Use status values that distinguish `candidate`, `reviewed_confirmed`,
`reviewed_rejected`, `uncertain`, and `unavailable`. Absence of an event is not
a confirmed negative. Keep model score semantics explicit; do not rename an
uncalibrated similarity as confidence or probability. Include detector/model
identity, configuration, source recording identity, and taxonomy version in
the result metadata.

## Implementation work

### Phase 1: Taxonomy and contract

- Add `configs/cmai_long_form_camera_v1.json` with all 29 item IDs, names,
  camera status, positive definition, exclusion rules, and evidence needs.
- Add a Python taxonomy loader and schema validation. Fail startup or result
  validation on unknown item IDs or incompatible taxonomy versions.
- Change training manifests to require canonical CMAI item IDs and split
  assignments. Reject items not enabled for the chosen detector mode.
- Extend the Person 2 → Person 3 contract, or add a versioned companion event
  contract, and keep migration tests for existing result JSON.

### Phase 2: Detector baseline

- Keep Person 1 responsible for perception inputs and quality flags.
- Extend Person 2 with behaviour-specific candidate detectors using the
  existing chunk, pose, motion, and prototype interfaces.
- Start with CMAI 01, 26, and 29; they best match the current pose/motion
  representation. Add physical interaction items only after person/object
  interaction features and suitable labelled data exist.
- Use consecutive evidence and temporal aggregation to emit bounded events.
  Reset support at track gaps, quality failures, and incompatible evidence.
- Emit no label for unavailable evidence. Preserve low-quality, missing-model,
  and insufficient-evidence states as abstentions.

### Phase 3: Upload application integration

- Make the dashboard load the selected detector bundle automatically for
  enabled behaviours. Keep custom prototype upload as an advanced override.
- Run the complete P1 → P2 → P3 workflow after upload and display candidates
  with timeline seeking, frame/clip evidence, and reviewer decisions.
- Surface unsupported CMAI items as “not assessed by camera” rather than
  blending them into a no-behaviour result.
- Add cancellation, progress, and clear errors for unsupported or corrupted
  recordings.

### Phase 4: Annotation and evaluation

- Create a video annotation protocol based on the selected CMAI long-form
  item definitions, with operational rules for observable actions, event start
  and end, ambiguous cases, occlusion, and `uncertain` labels.
- Obtain permitted recordings with subject/session-disjoint train, validation,
  and test partitions. Annotate at the event level; never split overlapping
  chunks from the same recording across partitions.
- Train/build prototypes on train only. Select thresholds on validation only;
  report held-out test metrics once. Record provenance, permissions, label
  agreement, coverage, precision/recall/F1, event timing quality, and
  abstentions per CMAI item.
- Do not mark an item enabled for upload results until its evaluation and
  evidence path meet agreed acceptance criteria. Keep the criteria and results
  in versioned experiment reports.

## Acceptance criteria

- An uploaded recording produces a result without requiring a user to upload
  JSON or manually supply a prototype bank for shipped/validated behaviours.
- Every emitted behaviour uses a canonical CMAI long-form item ID and links to
  the right timestamped evidence for the correct person track.
- A reviewer can inspect and confirm/reject/mark uncertain each candidate.
- Camera-ineligible items are explicitly marked not assessed; they are never
  presented as “not present”.
- Missing, low-quality, or occluded evidence results in an abstention, not a
  negative label.
- Dataset split checks prevent subject/session/video leakage, and test labels
  do not affect training or threshold selection.
- Detector version, taxonomy version, and output contract version are recorded
  for every result.
- The user interface describes flags as video candidates and does not present
  them as diagnoses or caregiver-rated two-week frequency scores.

## Out of scope

- Completing CMAI frequency or disruptiveness ratings from one recording.
- Audio-only behaviours in camera-only mode.
- Inferring intent, clinical cause, severity, or whether conduct is
  inappropriate without the required human/contextual judgment.
- Claiming clinical validation without an appropriate study and review.
- Building reliable detectors for all 29 items before suitable data and
  sensors exist.
