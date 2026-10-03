# Person 2 reproducible embedding experiments

## Current empirical status

No labelled research dataset, training labels, or model checkpoints are present
in this repository. **Empirical model selection is pending labelled data.**
There is no measured best representation, accuracy, F1, or clinical validity.
`experiments/person2-readiness/report.json` is an actual no-dataset readiness
report: all metric fields are null and no prototypes or rankings were produced.
Synthetic fixtures under `tests/` verify algorithms only; they are not research
data and their results are not reported as model-selection evidence.

Person 2 implements embeddings, prototype matching, temporal candidate support,
and experiments. Person 1 provides perception JSON; Person 3 handles downstream
Qwen/Groq, CMAI validation, verification, clips, dashboard, and Supabase.

## Setup and commands

From the repository root, use Python 3.11 and the tested dependency pins:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r configs/person2-experiments.requirements.txt
.venv/bin/python -m pip install --no-deps -e .
export PYTHONPATH="$PWD/src"
.venv/bin/python -m pytest -q
.venv/bin/python -m person2.cli experiment --output-dir experiments/person2-readiness
```

The explicit source path also supports running from the checkout when editable
installation is unavailable. In the current local environment the editable
`.pth` file exists but its source path was not added to `sys.path`; using
`PYTHONPATH=src` resolved CLI imports. Dependencies are installed in `.venv`,
and tests and CLI smoke checks work with that source path. No model downloads,
cloud service, API key, or GPU is needed for these baselines.

After obtaining a real annotated dataset, run:

```sh
.venv/bin/python -m person2.cli experiment \
  --manifest /absolute/path/to/annotations/manifest.json \
  --config configs/person2.json \
  --thresholds 0.6 0.7 0.8 0.9 \
  --temporal-bins 4 \
  --output-dir experiments/person2-labelled-run
```

Supply the exact same manifest, source JSON files, configuration, code, and
dependency versions to repeat an experiment. The algorithm uses no random
split or stochastic training: partition assignments are explicit, ties use a
documented deterministic rule, and input/code hashes are recorded. Keep a copy
of the source files and this code revision alongside the report. For new data
or annotation revisions, use another output directory.

## Annotation manifest and split policy

The manifest is validated against `person2.experiments.schema.json`. Its
required fields are:

| Field | Purpose |
| --- | --- |
| `schema_version` | Exactly `1.0` |
| `provenance` | Dataset `name`, `source`, `annotation_protocol`, `annotation_version`, `license_or_permission` |
| `behaviours` | Explicit ordered list of candidate labels to evaluate |
| `records` | Explicit annotations for individual source track/chunk windows |

Each record requires `perception` (path relative to the manifest), `person_id`
(Person 1 session-local tracker ID), `chunk_id`, `subject_id`, `session_id`,
`annotation_id`, `split` (`train`, `validation`, or `test`), and `labels`.
`subject_id` must identify the same person globally across sessions, rather
than reuse Person 1's local track ID. `session_id` must be globally stable across
all files from one recording session. Dataset owners supply and audit those
identifiers; Person 2 does not infer identity from video.

Annotations are multilabel and must apply to the complete selected chunk under
the annotation protocol. `labels: []` explicitly means an annotated negative;
omitting labels is invalid and unannotated windows are not treated as negatives.
Labels must belong to `behaviours`. Use actual annotated CMAI-related physical
candidate labels and appropriate negative/control examples. The runner does
not assign a clinical label or validate an annotation.

First run descriptor extraction with the intended chunk configuration to get
the real chunk IDs. A structural record example with placeholders is:

```json
{
  "perception": "<actual perception JSON filename>",
  "person_id": "<actual session-local track ID>",
  "chunk_id": "<actual chunk ID>",
  "subject_id": "<global subject identifier>",
  "session_id": "<global recording session identifier>",
  "annotation_id": "<annotation record reference>",
  "split": "train",
  "labels": ["<explicitly annotated label from behaviours>"]
}
```

Do not use these placeholders as data. Include nonempty train, validation, and
test partitions. Before any training, the runner rejects subjects, sessions,
source `video_id`s, or identical source-file hashes appearing across partitions;
it also rejects duplicate annotated windows and inconsistent subject/session
assignments. Splitting overlapping chunks from one recording across partitions
is forbidden. Assign partitions at the subject/session level before annotating
or extracting windows. There is no random window split.

Hash equality catches copied files; it cannot prove two differently encoded
videos show different people or sessions. Correct global IDs and dataset
provenance remain essential. The manifest, annotation references, input hashes,
and split assignments are copied into the report for audit.

## Four comparisons

| Variant | Pose | Motion | Temporal representation |
| --- | --- | --- | --- |
| `pose_only` | Summary descriptor, weight 1 | Weight 0 | Mean/std/mean velocity |
| `motion_only` | Weight 0 | Summary descriptor, weight 1 | Mean/std of supplied features |
| `pose_plus_motion` | Summary, weight 1 | Summary, weight 1 | Concatenated summary blocks |
| `temporal_pose_plus_motion` | Ordered time bins, weight 1 | Summary, weight 1 | Masked body-relative x/y means in time order |

The temporal encoder uses fixed bins of the configured duration anchored to
the first observation in the chunk; assignments use actual timestamps, not
frame ordinal positions. Empty bins stay masked. It never interpolates missing
pose. Tail windows retain empty later bins instead of stretching their time
axis. This is a temporal representation baseline, not a trained neural network.
Bin count and duration are part of its embedding-space identity.

All variants use the same chunking, quality, prototype, smoothing, and minimum
event-duration configuration. The runner overrides only modality weights and
pose representation for the ablation. Optional video is disabled in these four
required comparisons so their effects remain interpretable. Video encoders can
be compared separately through the same inference interface; no learned video
encoder is asserted to outperform these representations.

## Training, tuning, and metrics

Only usable, explicitly labelled `train` chunks contribute to masked prototype
centroids. Every configured behaviour needs a usable training exemplar. Missing
classes or unavailable evidence produce a per-variant pending status, never
invented prototypes. Training negatives contribute no positive prototype;
negative examples in evaluation measure false candidate support.

For each representation, the threshold grid is evaluated on `validation` only.
The runner selects the threshold with the highest validation micro-F1; a tie
favors the higher threshold. If validation has no usable coverage or no defined
F1, threshold selection and test evaluation remain pending. Only the selected
threshold is applied to `test`. Test labels cannot affect prototypes or tuning.
Inference smoothing can use unannotated preceding chunks within that same
held-out track, but it never uses their labels.

Metrics are **chunk-level multilabel candidate metrics**:

- Per-label TP, FP, FN, and TN, precision, recall, and F1.
- Micro precision/recall/F1 and macro-F1 over labels with defined F1; the number
  of included labels is recorded.
- Per-label and micro coverage, plus separate abstained-positive and
  abstained-negative counts.
- Original truth, candidate labels, abstentions, statuses, and scores for each
  annotated validation/test window.

An abstained positive counts as missed evidence (FN) for recall/F1, so low
coverage cannot hide missed positives. An abstained negative is **not** counted
as a true negative. Precision is null when there are no predicted positives;
recall is null with no positive annotations; F1 is null when its denominator is
zero. Counts accompany these values. No accuracy, probability calibration,
AUROC, clinical score, or event-verification claim is inferred from cosine.

The report compares representations and records chosen thresholds; it does not
automatically call one model best. Interpret differences only with sufficient,
representative held-out data and independent replication.

## Recorded artifacts and limitations

`report.json` records dataset provenance, manifest/source hashes, split audit,
runtime/library versions, Person 1/2 Python code hashes, variant configurations,
encoder identities/spaces, threshold grid/selection, metrics, predictions, and
limitations. Each evaluated variant writes a training-only prototype bank with
a reported checksum and example counts. Readiness reports contain null metrics
and no banks. Incomplete comparisons retain per-variant pending reasons.

Ordered bins are coarse; summary statistics discard temporal order. Pose
normalization removes some whole-body translation, so pose alone may miss
pacing evidence. Motion units still depend on Person 1's schema; signed-log
compression and block weights are engineering choices. Bbox and pose errors,
occlusion, camera motion, sparse sampling, and domain shift can affect scores.
Overlapping chunks are correlated; confidence intervals and event-localization
metrics are not currently implemented. No trained action checkpoint, class
calibration, deep temporal model, or empirical model ranking is bundled.

Person 3 must review candidate intervals, preserve unknowns, perform CMAI
validation and final verification, and handle all downstream product/storage
work. Those tasks are outside this module.
