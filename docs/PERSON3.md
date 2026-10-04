# Person 3: evidence validation and application

## Purpose and research boundary

Person 2 is the high-recall candidate generator. Person 3 checks whether the supplied P1 evidence is adequate to review each candidate, requests a second-stage evidence verification from Qwen through Groq when configured, and emits traceable events. The governing principle is: **Person 2 finds candidate behaviours, Person 3 validates the supplied evidence, Qwen provides a second-stage verification, and Python guarantees traceable timestamps.** Qwen is not a clinical diagnostic model. Neither Qwen decisions nor P2 scores are clinical probabilities or evidence of clinical validity.

## P2 handoff and P1 source data

The adapter consumes the existing Person 2 `Person2VideoResult` schema 1.0 without altering its semantics. For each P2 person's `events[]`, it retains behaviour, interval, `peak_similarity`, and `chunk_ids`. Each source chunk supplies `chunk_id`, `frame_indices`, and optional behaviour scores. The score copied to the P3 candidate is the maximum matching non-null `smoothed_similarity`, falling back to the event's peak similarity. P2 calls this score `uncalibrated_cosine_similarity`; P3 carries it as a candidate score without calibration.

The current checkout contains both P1 and P2. The video dashboard runs them in sequence. `person3.p2_adapter.read_p2_handoff` reads the exact versioned JSON handoff fields needed by P3 for advanced JSON review; `person3.evidence.candidates_from_p2` also accepts the P2 Python result object. Advanced uploads must have matching video IDs. P1 observations are identified deterministically as `{person_id}:frame:{frame_index}`. Person IDs are session-local tracker IDs and must not be interpreted as real identities.

The versioned `CandidateBehaviour` adapter preserves candidate ID, person ID, label, score, interval, P2 source-window IDs, P1 observation IDs and additional evidence. It rejects unknown fields, nonfinite scores, invalid score bounds and nonpositive intervals.

## Taxonomy and mapping

`person3.taxonomy` is the single label registry used by checks, verifier-facing packets and dashboard labels. It includes the project's requested physical and verbal categories. It also explicitly maps the Person 2 project's `pacing_aimless_wandering`, `repetitious_mannerisms`, and `general_restlessness` candidate labels to review categories. These are names for research review, not a claim that a short video establishes a two-week CMAI frequency rating. Unrecognized labels are not fuzzy-matched; they fail the taxonomy gate and abstain.

## Compact evidence construction

`person3.evidence` selects P1 observations referenced by the source P2 chunks. Each evidence segment carries its deterministic source ID, original timestamp/frame index, selected body-normalized landmarks, motion feature values relevant to speed, acceleration, displacement and jerk, and P1 quality flags. The packet also retains person/candidate IDs, behaviour, score, candidate interval and source window IDs. It does not send raw video or all of the P1 JSON to Qwen.

The deterministic CMAI-aware gate currently requires a recognized taxonomy label, at least two source observations with strictly increasing times, and at least two detected non-interpolated poses. A failed gate yields `insufficient_evidence` without a remote call. These are engineering safeguards, not clinically validated thresholds; the available pose/motion information cannot establish all verbal behaviours.

## Qwen through Groq

`GroqQwenValidator` uses `GROQ_API_KEY`, `QWEN_MODEL`, and `QWEN_TIMEOUT_SECONDS`. The system prompt limits the model to validating the supplied behavioural evidence, forbids diagnosis, invented evidence and timestamps, and asks for JSON containing a decision, a short reason, and evidence segment IDs. The parser accepts a JSON object, surrounding prose, or a Markdown JSON fence; schema errors, API errors and missing credentials become controlled `insufficient_evidence` outcomes. The dashboard does not display chain-of-thought. Tests use mocked validators and do not call Groq.

Allowed decisions are `supported`, `unsupported`, and `insufficient_evidence`. A verifier's unknown evidence ID causes abstention. A supported result must select at least one valid source ID.

## Deterministic timestamps and event identity

Qwen never supplies event timestamps. Python maps the selected source evidence IDs to their P1 timestamps and calculates the minimum and maximum. If these do not form a positive interval, P3 abstains. Every supported event keeps selected observation IDs, P2 source-window IDs, person/candidate IDs, score, status, reason and quality flags.

Deduplication first collapses duplicate non-supported results for the same candidate, outcome and selected evidence. Supported results with the same person and behaviour are duplicates when their source evidence IDs match or their temporal intersection-over-union reaches 0.5. In a collision, P3 retains the event with more selected source IDs, then the higher candidate score. Different behaviours and different people are never merged solely because intervals overlap. This is a deterministic review heuristic, not a temporal event model.

## Clips and dashboard

`person3.clips.extract_event_clip` extracts a supported event with default two-second padding, clamps to video boundaries, preserves source timestamps in the event metadata and writes `{event_id}.mp4`. Decode or writer failures return no clip. Source videos are not modified; generated videos are covered by `.gitignore`.

Use Python 3.11 for the tested video dependencies:

```sh
python3.11 -m venv .venv  # only if it does not exist
source .venv/bin/activate
python -m pip install -e '.[dashboard,models,test]'
python -m streamlit run dashboard.py
```

The app adds the checkout's `src` path itself, so the launch command does not
depend on an editable-install path being added to `sys.path`. The model extra
pins MediaPipe 0.10.21 for Person 1's existing legacy Pose API; the dashboard
range supports its protobuf dependency. The configured YOLO checkpoint is
downloaded on first use if absent. No labelled prototype bank is bundled.

On the tested Apple Silicon machine, `pip check` reports MediaPipe 0.10.21 as
unsupported because its internal wheel metadata declares an x86_64 tag, despite
the [published universal2 wheel](https://pypi.org/project/mediapipe/0.10.21/).
Native MediaPipe Pose initialization and YOLO tracking smoke checks both passed.
This packaging warning remains; no dependency metadata was rewritten to hide it.

**Video** is the default input. Upload a source video and click **Analyse
video** to run P1 perception followed by P2 candidate generation. JSON files
are internal outputs, available as downloads; they are not required inputs.
Under **Behaviour references (optional)**, upload a compatible P2 prototype
bank built from explicitly labelled training examples. Without that bank, the
app extracts movement evidence and exports P1/P2 data but produces no invented
candidate labels. Incompatible banks cause a visible analysis error. P1 and P2
contracts, thresholds, and model algorithms are preserved.

Enter a **Groq API key** in the sidebar's password field, or configure
`GROQ_API_KEY` in the environment or `.streamlit/secrets.toml`. Typed entry takes
precedence over Streamlit secrets, then environment. The key remains in the
current session and is not written to result files. The model field defaults
to the configured `QWEN_MODEL`, or `qwen/qwen3-32b`; it is editable. See
[Groq's Qwen model documentation](https://console.groq.com/docs/model/qwen/qwen3-32b).
Setting a key does not claim a successful connection. **Verify candidates with
Groq** is enabled only when a key, model, and candidates are present. Clicking
it sends compact pose/motion evidence; raw video is not sent. Local analysis
does not require the key, and API failures remain unverified outcomes.

**Existing results (advanced)** retains the matching P1/P2 JSON workflow, with
an optional source video for clips. The dashboard filters events, shows a
timeline/table, and exposes evidence IDs and validation status. Clips are
created on request. Uploaded videos and clips use isolated temporary session
directories; changing inputs clears old results and cleans the previous
directory. Changing the key or model clears previous verification results.
Supabase persistence remains optional and requires its separate explicit
save button.

## Optional Supabase

Set `SUPABASE_URL` and `SUPABASE_KEY` to enable the REST store. The default table name is `person3_events`; rows contain `video_id` plus all Person 3 event metadata fields. A deployment must create a table with a unique `event_id`, a `video_id`, and JSON/JSONB-capable columns for list-valued traceability fields, and allow authenticated inserts/upserts. Video binaries are never sent to Supabase. Without either credential, the store is disabled and returns `False` without network access.

## Limitations

P1 landmarks and geometric motion are not semantic evidence for every taxonomy item. In particular, verbal agitation cannot generally be validated from pose alone. CMAI is broader than short video intervals, and this code does not establish item frequency, severity, diagnosis, or clinical validity. P2 scores are uncalibrated similarities. Qwen can make errors and is only a second-stage verifier; human review remains necessary. The taxonomy gate, minimum observation count and IoU deduplication threshold are engineering choices that require evaluation on independently labelled data. Clip timestamp seeking depends on the local decoder. Supabase deployment schema and access policies are managed by the researcher.
