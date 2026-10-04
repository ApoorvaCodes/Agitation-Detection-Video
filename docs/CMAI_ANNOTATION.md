# Camera event annotation protocol, version 1 (research draft)

Before annotation, obtain permission for the recording and this use. Record
source, permitted uses, access/retention restrictions and consent in dataset
provenance. Use the long-form taxonomy version exactly; verify the supplied
manual edition with the dataset owner. This protocol labels visible actions,
not intent, social appropriateness, clinical cause, severity or two-week ratings.

Assign globally stable subject and session IDs, and allocate train/validation/test
at the subject/session level before deriving overlapping windows. All cameras,
exports and tracks from the same session remain in one partition; all sessions
of a subject remain in that subject's partition. Never identify a subject by a
session-local tracker ID alone. Audit aliases and re-encodings manually; content
hashes cannot prove identities. Avoid test reuse for model tuning.

Annotate the full reviewable track interval at event level with source seconds,
canonical item ID, reviewer/annotation reference and `observable` or `uncertain`
certainty. Mark ambiguous onset/offset, track swaps, occlusion, camera cuts or
missing evidence as uncertain intervals. Record reviewed start/end explicitly.
Only fully reviewed intervals can contain negative labels; unannotated footage
is unknown. Have at least two reviewers annotate independently, adjudicate
conflicts and record event agreement with a specified matching/time tolerance.
Report measured agreement; otherwise state it has not been measured.

Operational starting rules (these require pilot review before data collection):

| Item | Visible positive | Start/end | Exclusions/uncertainty |
| --- | --- | --- | --- |
| 01 | At least two traversals/back-and-forth legs of a visible walking path | First observed traversal in the sequence to last observed walking leg | Do not assert aimlessness. Ordinary one-way transit, purposeful task movement, camera motion, unobserved paths and track swaps are excluded or uncertain |
| 26 | At least two cycles of a repeated body/object movement, such as rocking or tapping | First visible cycle to last cycle | Speech repetition excluded. Obscured contact or mixed unrelated gestures uncertain; this baseline only has upper-body pose/motion evidence |
| 29 | Repeated sit/stand transitions or sustained seated movement over an agreed duration | First clearly visible transition/movement to last | Ordinary single posture adjustment, hidden lower body or unclear seated/standing state uncertain. Agree a duration criterion before annotation; do not invent one from a model threshold |

Operational duration, cycle tolerance and acceptable gaps must be agreed and
recorded in `annotation_protocol`/`annotation_version` before annotators work.
Dataset owners must pilot inter-rater agreement before enabling a detector.
Physical-interaction, object, sexual and audio-dependent items remain outside
this initial detector mode, even when the camera map says conditional.

Use `docs/cmai.annotations.schema.json`. The event-manifest top level requires
`schema_version: cmai-event-annotations-1.0`, `taxonomy_version:
cmai-long-form-camera-v1`, `provenance`, `behaviours`, and `recordings`.
Each recording contains `perception`, `person_id`, global `subject_id` and
`session_id`, explicit `split`, `annotation_id`, `reviewed_start`, `reviewed_end`
and `events`. Each event contains its own annotation ID, matching perception,
person/split, start/end, canonical `labels`, and `certainty`. Provenance includes
name, source, protocol, annotation version, permission and `label_agreement`.
The schema documents structure; placeholder examples are not supplied as data.

Prepare a chunk manifest with the intended P2 configuration:

```sh
PYTHONPATH=src python -m cmai.experiments --annotations /path/to/annotations.json --output /path/to/chunks.json --config /path/to/person2-config.json
PYTHONPATH=src python -m person2.cli experiment --manifest /path/to/chunks.json --config /path/to/person2-config.json --output-dir /path/to/new-experiment --thresholds 0.6 0.7 0.8 0.9
```

Preparation excludes windows outside reviewed coverage, touching uncertainty or
partially crossing a positive event edge. This avoids turning unobserved/event
edge windows into training negatives. Event annotations and complete reviewed
coverage are retained separately for event evaluation. Use the same chunk
configuration for preparation and evaluation; changing it requires preparation
again. Every enabled label needs usable training examples with its required
camera evidence. Train alone builds prototypes; validation alone selects a
threshold. When event annotations exist, threshold selection uses event micro-F1
at temporal IoU ≥0.5; otherwise it uses chunk micro-F1. Deterministic ties favor
the higher threshold. The selected threshold evaluates the held-out test once
per representation. Re-running a test during development does not create a new
independent test set; freeze a final report and record any subsequent use.

Reports compare pose-only, motion-only, pose-plus-motion and temporal pose-plus-
motion. They record provenance/permission/agreement, source/configuration/code
hashes, split audits, prototype assets, thresholds, per-item precision/recall/F1,
coverage and abstentions. Event reports use one-to-one temporal-IoU matching and
record onset/offset errors for matched events; predictions outside reviewed
coverage or touching uncertainty are excluded and counted. Unmatched positive
annotations count as missed evidence, including abstentions. Chunk abstained
negatives are not true negatives. No calibration or best-model claim is made.

Select release acceptance criteria in advance with the project reviewers. Store
per-item minimum precision, recall, event F1, coverage and positive test sample
count, plus an evidence-path test and approval identity/date. Criteria and report
must be versioned and hashed in the released bundle. No numerical acceptance
criteria are pre-approved in this repository. Independent replication,
representative conditions, uncertainty analysis and clinical review require
additional real data and study design. The current protocol is an engineering
starting point, not a validated clinical annotation standard.

See [hitting/kicking candidate setup, event annotations and experiments](CMAI_ACTIONS.md) for the optional supervised action path. Items 07/08 remain unavailable by default pending permitted labelled data and release evidence.
