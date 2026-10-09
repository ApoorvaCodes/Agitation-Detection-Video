# CMAI camera candidate workflow

This application reports observable candidate actions within an uploaded recording.
It does not diagnose agitation, infer intent or appropriateness, assess severity,
or fill out a caregiver-rated two-week CMAI frequency questionnaire.

The engineering taxonomy in `configs/cmai_long_form_camera_v1.json` contains all
29 item IDs and the camera observability map supplied in `IMPLEMENTATION.md`.
Each item has an operational positive definition, exclusions and evidence needs.
Numbering was cross-checked against the long-form list in the
[RECage study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10376951/).
The exact edition of the referenced manual was not provided; its verification
is explicitly pending in `edition`. This is a project camera map, not an official
new CMAI instrument or a claim of detector accuracy. Changes in item semantics
require a new taxonomy version and a migration.

## Setup and upload

Use the existing Python 3.11 environment:

```sh
source .venv/bin/activate
python -m pip install -e '.[dashboard,models,test]'
python -m streamlit run dashboard.py
```

Upload one video, then click **Analyse video**. The worker validates metadata
and initial decodability before constructing model objects. Progress reports
sampled source time; cancellation terminates the isolated worker and discards
partial results. Decoding that ends substantially before declared duration is
an error, not a completed result. P1 perception algorithms are unchanged; an
optional progress callback was added to their orchestration boundary. Metadata
validation preserves fractional FPS and closes decoders on invalid metadata.

P2 processes all tracks independently across the recording. Choose a track when
there is more than one. The review view shows item availability, explicit chunk
coverage/abstentions, candidate intervals, source-player seeking, a reference
frame with the selected track's bounding box, and local evidence clips. Select
a candidate to review it. Save a **confirmed**, **rejected**, or **uncertain**
reviewer decision; the original model candidate, score and interval remain
separate from the human decision and its audit history. Confirmation means the
reviewer judged the visible candidate action; it is not clinical validation.

Download legacy P1/P2 JSON, the CMAI companion JSON, or a ZIP containing JSON and
available clips. Evidence paths are relative within the ZIP. A plain JSON download
alone does not persist the clip files. Uploaded videos, generated clips and
review decisions live in temporary session storage until exported. Changing
inputs cancels pending work and removes previous temporary files. Browser/session
loss is not durable storage. Cancellation of a worker on abrupt server/browser
failure is not guaranteed; normal UI cancellation/input changes are handled.

Groq is optional. Enter its API key in the masked sidebar field and explicitly
click **Verify candidates with Groq**. Only candidates for the chosen track are
sent, as compact pose/motion evidence. No raw video is sent. The Qwen decision
and its provider/model identity are stored as `machine_validation`; it never marks a human review confirmed.
Changing the key/model clears machine verification, preserving human decisions.
Supabase saves remain separate explicit actions with an opt-in checkbox, using
the existing machine-review table. Human decisions are currently persisted in
local JSON/ZIP exports, not silently mapped into the old Supabase schema.

## Detector availability and assets

The app automatically loads `configs/cmai_detector_bundle.json`, or the local
path in `CMAI_DETECTOR_BUNDLE`. The bundled configuration is deliberately
`unavailable`: no labelled research dataset, evaluated prototype bank or release
approval was supplied. Video analysis still produces coverage and a result
without a JSON or prototype upload. It does **not** produce invented flags.
Upload-and-flag acceptance for shipped behaviours remains blocked on real data.

Bundle modes:

| Mode | Meaning |
| --- | --- |
| `unavailable` | No runnable behaviour asset; no labels emitted |
| `research` | Explicit labelled prototypes; candidates are research-only |
| `released` | Hashed evaluated assets, matching configuration/rules, per-item acceptance criteria and recorded human approval |
| `legacy_review` | Imported P2 result; original assets/evaluation are unknown; cannot run inference |

The optional prototype upload is an advanced **research override**. Three old
starter labels have explicit migrations to IDs 01, 26 and 29. Unknown labels,
audio-only labels and physical-interaction labels unsupported by the initial
pose/motion detector mode fail closed. Legacy generic P1/P2 contracts remain
schema `1.0`; legacy imports do not become released models. Old JSON has no
recording checksum, so a reviewer must confirm the uploaded source matches it
before clips are produced; the companion's verified `source_sha256` remains
null for this mode. Track IDs remain session-local, not real identities.

For a versioned local bank, use the four-way experiment runner, select a variant
based on evidence, and package it for research:

```sh
PYTHONPATH=src python -m cmai.experiments --annotations /path/to/annotations.json --output /path/to/chunks.json --config /path/to/person2-config.json
PYTHONPATH=src python -m person2.cli experiment --manifest /path/to/chunks.json --config /path/to/person2-config.json --output-dir /path/to/experiment
PYTHONPATH=src python -m cmai.bundle --report /path/to/experiment/report.json --variant pose_plus_motion --output /path/to/bundle.json
CMAI_DETECTOR_BUNDLE=/path/to/bundle.json python -m streamlit run dashboard.py
```

The named variant is an explicit researcher choice, not a best-model claim.
Its pose representation, temporal bins, modality weights, thresholds, minimum
support/duration, versions and asset hashes are preserved. Custom learned video
encoders remain injectable through P2's existing interface; they are not wired
into this pose/motion camera bundle without their own evaluation/integration.

Release requires `release_evidence`: dataset provenance, permissions, annotation
protocol, hashed evaluation report, hashed acceptance criteria, evidence-path
test, approver and date. Criteria are supplied by the project reviewers; none
are fabricated or pre-approved here. The criteria file is a list of objects
with `item_id`, `min_precision`, `min_recall`, `min_event_f1`, `min_coverage`, and
`min_test_positive_events`. The loader checks the report's taxonomy, completed
status, split audit, exact asset/configuration/rules and measured held-out
metrics against those criteria. Missing or failing values prevent release.
This is an engineering release gate, not an independent clinical study or an
approval/authentication service. Local signed-off configuration is trusted.

## Candidate detection semantics

Items 01, 26 and 29 are supported for initial prototype research. Each rule has its own
threshold, minimum consecutive chunks (at least two), minimum event duration
and minimum evidence fraction. Raw and EMA scores must both pass; smoothing
alone cannot count as a fresh observation. Chunk gaps, quality failures and
missing/incompatible evidence break support. Incompatible spaces are errors;
missing evidence is an abstention. Supporting intervals are bounded at barriers.

The evidence gate conservatively requires:

| Item | Visible landmarks | Valid P1 motion evidence (at least one) |
| --- | --- | --- |
| 01 | Both hips and ankles | `bbox_center.displacement` |
| 26 | Both shoulders and wrists | Left/right wrist speed |
| 29 | Both hips and knees | Body-centroid or left/right knee speed |

Masks and interpolation flags are respected. These gates establish evidence
availability, not that movement is aimless, repeated or restless. Labelled
prototypes and empirical evaluation must establish candidate specificity. The
current representations may confuse purposeful walking, ordinary posture
changes and calm motion with targets. Bilateral visibility requirements may
abstain conservatively. No object-interaction or audio detector is included.

### Experimental movement baselines

The camera dashboard also runs two separate experimental P2 detectors without
requiring labelled prototype assets. They use the existing P1 source timestamps,
session-local track boxes and normalized MediaPipe landmarks. Their availability
is shown as `research_only`; the per-track diagnostics report missing duration,
gaps, pose coverage and other abstention reasons.

Pacing (`cmai_01_pacing_aimless_wandering`) compares consecutive person-box
centres in person-size units. It requires sufficient duration and track coverage,
minimum accumulated path, repeated direction reversals along the dominant axis,
and revisits to previously traversed regions. It abstains on a single straight
walk, stationary tracks, short tracks, poor coverage and large timestamp gaps.
It cannot determine whether walking is purposeful or clinically agitated.

Restlessness (`cmai_29_general_restlessness`) compares only visible,
confidence-qualified normalized upper/lower body landmarks across adjacent valid
poses. It requires repeated movement bursts and persistence, minimum pose
coverage and track duration. It suppresses candidates for clear dominant-axis
locomotion so walking is handled by the trajectory detector. Missing joints are
omitted; motion is never interpolated across absent landmarks. Isolated gestures
or one posture change are below the default burst requirement.

Both default threshold sets are experimental engineering starting points, not
clinical values. Configure them in `configs/movement_patterns.json`, or point
`MOVEMENT_PATTERN_CONFIG` to a local JSON file. Defaults are: 12-second window,
8-second minimum track, 0.65 minimum valid coverage, 0.40 minimum person-detection
confidence, 1.5-second maximum gap,
0.035 body-scale minimum path step, 2.0 body-scale minimum pacing path, 4-second
minimum moving time, two direction reversals, 0.25 recurrence, 0.50 landmark visibility, four
restlessness bursts, 0.20 movement persistence, 0.25 movement changes per
second, eight valid pose frames, 0.07 body-scale movement and posture-change
thresholds, and 0.72 locomotion path ratio. Candidate evidence
contains detector version, source observation IDs/timestamps, features and
acceptance reasons; per-track diagnostics preserve abstention reasons.

Hitting and Kicking remain distinct: Hitting uses its experimental arm-motion
baseline, while Kicking requires the existing action model and target/contact
gate. The four dashboard rows are Hitting, Kicking, Pacing / Aimless Wandering
and Restlessness. Audio-only behaviours are not included.

All scores are `uncalibrated_cosine_similarity` in [-1,1], never confidence or
probability. Camera-ineligible items are `not_assessed_by_camera`. Other items
without assets are `unavailable`. Research and released availability are separate
from events; zero events is never a confirmed absence. Visibility/occlusion are
unknown because the current P1 contract does not measure those quantities.

## Companion contract and compatibility

`src/cmai/contracts.py` defines `cmai-camera-result-1.0` with taxonomy version
`cmai-long-form-camera-v1`; generated schema is `cmai.camera-result.schema.json`.
It includes form/edition/hash, source identity and metadata, recording duration,
detector configuration/identity/hash, encoder identities/spaces, all 29 item
availability entries, per-track coverage and per-item scores, and events.

Every event includes its canonical item ID, source label/candidate ID, interval,
uncalibrated score, source video/person/chunk/frame references, relative clip
reference/status/offset, deterministic P3 evidence checks, quality, separate reviewer history and optional machine
verification. Event statuses are `candidate`, `reviewed_confirmed`,
`reviewed_rejected` and `uncertain`; unavailable items have no invented interval
and are represented in `availability`. Review-status inconsistency, unknown
items/versions, camera-ineligible events, cross-person/video references,
unsupported chunks, gap bridging and intervals beyond source coverage fail
validation. Clip failure leaves a candidate reviewable in the source video;
it is never a negative outcome.

P1 result JSON and prototype-bank schema `1.0` are preserved. P2 schema `1.0`
adds the optional `movement_diagnostics` map with a default empty value, so older
P2 documents remain valid. Training
manifests now use schema `2.0`, canonical IDs, explicit splits and label-agreement
provenance. Old training manifests must be explicitly updated and re-audited;
the runner never silently assumes missing `split` means training.

## Verification and remaining work

Run `python -m pytest -q`. Tests use synthetic fixtures solely for algorithm,
contract, UI, cancellation, corrupt-video, migration and leakage regressions;
their metrics are not empirical detector results.

No permitted labelled recording dataset, agreement study, calibrated score or
clinical validation was supplied. The default readiness report retains null
metrics and no ranking. Model training, acceptance thresholds/approval, exact
manual-edition verification, external evidence-path validation and any durable
Supabase companion/reviewer schema remain pending. The existing Apple Silicon
MediaPipe wheel metadata warning is documented in `PERSON3.md`.

See [hitting/kicking candidate setup, event annotations and experiments](CMAI_ACTIONS.md) for the optional supervised action path. Items 07/08 remain unavailable by default pending permitted labelled data and release evidence.
