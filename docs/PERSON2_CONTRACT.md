# Person 2 → Person 3 contract (schema 1.0)

Person 2 consumes Person 1 JSON using `person1.io.load_perception` and emits
`person2.contracts.Person2VideoResult`. The JSON Schema is in
`person2.schema.json`; prototype banks use `person2.prototypes.schema.json`.
Load and validate with `person2.io.load_result` or
`python -m person2.cli validate result.behaviours.json`.

The existing handoff fields and schema version are preserved. Older schema 1.0
JSON and prototype banks remain loadable; default summary embedding identities
remain unchanged. Encoder details are written to a separate
`OUTPUT.encoders.json` sidecar to avoid adding fields that older strict consumers
would reject. The core JSON still records each chunk's versioned embedding
`space` and ordered coordinate metadata. See
[reproducible experiments](PERSON2_EXPERIMENTS.md) for setup, dataset requirements,
comparison commands, metrics, and the current pending-data status.

## Pipeline and scope

`Person1VideoResult → per-person temporal chunks → pose/video embeddings + motion
descriptor → masked weighted fusion → exemplar prototype cosine similarity →
temporal smoothing → candidate intervals`.

This is an initial local research baseline. It does not contain trained temporal
ML weights, a labelled training dataset, clinical calibration, or pretrained
text-to-video alignment. Prototypes must be built from explicitly labelled
exemplars in the same embedding space. No labels are fabricated in the absence
of prototypes. Initial physical behaviour labels for annotated examples can be
`pacing_aimless_wandering`, `repetitious_mannerisms`, and `general_restlessness`.
Label strings remain extensible so a dataset can also include normal activity
and other controls. An annotation and prototype do not establish clinical validity.

CMAI includes physical and verbal items and caregiver frequency ratings over two
weeks. Short local video candidates cannot establish those ratings or verbal
behaviours. See the [APA measure description](https://www.apa.org/pi/about/publications/caregivers/practice-settings/assessment/tools/cohen-mansfield)
and [published CMAI factor study](https://pubmed.ncbi.nlm.nih.gov/16286443/).

## Top-level fields

| Field | Meaning |
| --- | --- |
| `schema_version` | Person 2 output version, exactly `1.0` |
| `source_schema_version` | Consumed Person 1 version, exactly `1.0` |
| `video_id` | Original video ID; person IDs stay session-local |
| `score_semantics` | `uncalibrated_cosine_similarity`, never a probability |
| `configuration` | Complete engineering configuration used for inference |
| `embedding_spaces` | Ordered coordinate names for each modality |
| `prototype_version` | Labelled bank version, or null if absent |
| `persons` | Each track's `person_id`, `chunks`, and `events` |

## Time and quality

Each chunk has `chunk_id` (`person_id:ordinal`), `segment_id`, `start_timestamp`,
`end_timestamp`, and its original `frame_indices`. Times are source seconds.
Intervals are half-open `[start,end)`. Defaults are 2 seconds with 50% overlap.
Windows are anchored at the first observation in each continuous track segment.
A source gap greater than 1 second starts another segment. Inputs must have
strictly increasing finite timestamps and frame indices for each person.

Tail windows are retained with shortened end times. The last observation's
support extends by `min(1 / processed_fps, max_gap_seconds)`; FPS falls back to
`video.fps` only when `processed_fps` is absent. This is sample support, not a
claim that a behaviour continued beyond the last observation.

`valid_fraction` is usable **enabled** pose/motion frames divided by the larger of observed
frames and expected sampled frames in the interval. Interpolated bboxes do not
count as pose/motion evidence. Pose and feature validity masks are respected;
nonfinite input values are excluded. Video alone does not bypass pose/motion
quality gating in this baseline. Defaults require at least three observations
and a 0.6 valid fraction. These are engineering defaults.
Evidence from a disabled modality cannot pass that quality gate. At least one
pose or motion modality must be enabled. Nonfinite x/y invalidate a pose point;
missing intermediate samples break summary pose velocities.

`status` is one of:

- `low_quality`: too few observations or insufficient valid coverage; no scores.
- `no_prototypes`: usable chunk but no labelled bank; no scores.
- `insufficient_evidence`: bank present but no valid cosine comparison.
- `scored`: at least one usable comparison; individual labels can still be null.

Person 3 must preserve abstention as unknown rather than interpret it as no
agitation or normal activity.

## Embeddings and fusion

Every `Embedding` contains `space`, `values`, and `valid` of equal nonzero
length. Zero storage with `valid=false` is an unavailable coordinate, not an
observed zero. JSON numerical values are finite. Summary values and unavailable
scores use null.

- `pose`: for Person 1's 13 key joints in their established order, x then y,
  each with mean, population standard deviation, and mean signed velocity.
  Coordinates are body-normalized; velocities use actual timestamp differences.
  Missing or invalid intermediate pose samples break derivatives. Z is omitted
  because it is optional in the source contract. This descriptor summarizes a
  window; it is not a learned action representation and does not fully preserve
  temporal order.
- `motion`: for each authoritative `video.feature_names` entry, mean and
  population standard deviation across valid samples. A standard deviation
  needs at least two samples. Signed `log1p(abs(value))` compresses the descriptor;
  `motion_features` exposes the uncompressed summaries and their nulls. Person 1
  precomputed windows are not reused because Person 2 owns its gap boundaries.
- `video` (optional): the explicit `--video` path enables a per-channel RGB
  histogram averaged over person bbox crops at original frame indices (8 bins
  per channel, R/G/B). This is an appearance baseline, not an action model.
  OpenCV decodes local video only. The user must supply the matching source
  video. Failed individual reads contribute no histogram; opening a missing
  video raises an error. No implicit reading of `source_path` occurs.

For a learned video embedding, inject an object implementing `VideoEncoder`:
`feature_names` supplies fixed ordered coordinates, and `encode(observations)`
returns an `Embedding` in a versioned model space. Include model/checkpoint and
preprocessing identity in `space`. Encode per-person crops and preserve the
chunk's timestamps. Prototype building and inference must use that same provider.
Every encoder's space and dimension must remain constant across all tracks and
chunks in a run. Prototype compatibility is checked even for low-quality chunks.
Coordinate names must be nonempty and unique.

The CLI also supports `--pose-representation time-bins --temporal-bins 4`.
The ordered, timestamp-based pose-bin representation replaces the summary pose
block; it requires new prototypes. Its configured duration must equal the
pipeline window duration. Empty bins remain masked; later bins of short tails
are not filled. Other handoff fields and motion semantics remain the same.

To compare video encoders without changing the pipeline, supply
`--video-encoder encoder.json`, mutually exclusive with `--video`. That JSON
configures a local factory and an explicit model identity, for example:

```json
{
  "factory": "person2.embeddings:RGBHistogramEncoder",
  "kwargs": {"video_path": "source.mp4", "bins": 8},
  "identity": {
    "model": "rgb_crop_histogram",
    "version": "1",
    "preprocessing": "source_bbox_rgb_0_256_8_bins"
  }
}
```

Factories take the configured keyword arguments and return `feature_names` and
`encode(observations)`. Paths in factory kwargs are relative to the current
working directory. The wrapper binds the declared identity and the provider's
returned space to a new comparison space. For trained encoders, record the actual
model/checkpoint checksum and complete preprocessing settings in `identity`;
the factory owns loading that checkpoint. No model is implicitly installed or
downloaded. Direct Python injection remains supported; an optional `identity`
dictionary enriches its sidecar. Include checkpoint and preprocessing versions
in the returned `space` even when there is no identity dictionary.

`encoder_metadata(result, pose_encoder, video_encoder)` describes modality
identities, observed space IDs, coordinate names, and fusion space IDs.
The CLI writes it by default; `--metadata-output` changes the sidecar path.

`fused_embedding` concatenates modality blocks in pose, motion, optional video
order. Each nonzero block is L2-normalized, then multiplied by the square root
of its configured weight. Zero-norm or disabled blocks are masked unavailable.
The fusion space includes modality space IDs, dimensions, weights, and configured
window duration. Reordering features or changing encoders, weights, or window
duration requires rebuilding prototypes. Incompatible comparisons raise an
error instead of silently scoring.

## Prototypes, similarities, and temporal aggregation

A `PrototypeBank` contains `schema_version`, a user-managed `version`, and
unique `behaviour` prototypes, each with an `embedding` and `example_count`.
The builder averages unit-normalized labelled exemplars coordinate by coordinate
using masks. Invalid and zero-norm exemplars and cancelled centroids are rejected.
All prototypes must share the same space. Prototype banks record means rather
than raw videos. Keep training exemplars separate from evaluation tracks/videos.

Per chunk, `scores` includes every bank behaviour:

| Field | Meaning |
| --- | --- |
| `behaviour` | Prototype label |
| `similarity` | Cosine on jointly valid coordinates, `[-1,1]`, or null |
| `shared_fraction` | Jointly valid coordinates / union of valid coordinates |
| `smoothed_similarity` | EMA of valid similarities, or null |
| `candidate` | Smoothed score meets the configured threshold |

Comparison requires `shared_fraction >= 0.5` by default and nonzero norms.
EMA is `alpha * current + (1-alpha) * previous`, default alpha 0.5; the first
score initializes it. History is isolated by person and behaviour and resets
at gaps, low-quality windows, or unavailable comparisons. Overlapping windows
are correlated observations; this EMA is not a probability model.

Multiple labels can be candidates. The default threshold is 0.8 and must be
tuned on independent labelled data. Consecutive candidate support intervals
are merged; an abstaining/noncandidate chunk or segment boundary closes an
event. Event duration is the union extent of these intervals, not the sum of
overlapping window lengths. Events shorter than 1 second are dropped by default.
When an overlapping abstaining or noncandidate chunk begins, the preceding
event is clipped at that chunk's start so old window support cannot extend
through the new barrier. `chunk_ids` remain references to its earlier supporting
windows, which may individually have longer source intervals.

Each event includes `behaviour`, `start_timestamp`, `end_timestamp`,
`peak_similarity` (maximum smoothed score), and supporting `chunk_ids`.
Person 3 should use these as reviewable candidate evidence, not confirmed CMAI
items, item frequencies, severity, or a total score.

## Experimental Hitting motion-baseline source

Person 2 retains its labeled prototype similarity path and also runs the explicitly
experimental Hitting motion baseline on timestamped P1 normalized pose observations.
This baseline does not require a prototype bank or contact annotations. It emits
Experimental `cmai_01_pacing_aimless_wandering` and `cmai_29_general_restlessness`
events use the same `motion_baseline` source marker and include a detector
version, source observation IDs/timestamps, accepted features, and reasons.
The optional per-track `movement_diagnostics` map records accepted-window
features or abstention reasons. Hitting events retain `candidate_source:
"motion_baseline"`, the actor's
left/right arm, a bounded engineering score, P2 window IDs, and detailed source
observation features. Prototype events retain `candidate_source: "prototype"`.
Per-event source metadata disambiguates motion scores from prototype cosine
similarity; neither is a calibrated probability. Thresholds are in
`configs/hitting_motion.json` and documented in `CMAI_ACTIONS.md`. It is a
research baseline and does not replace the prototype architecture or constitute
clinical validation.

## Example prototype workflow

Run without a bank to extract descriptors, then annotate selected chunks:

```sh
python -m person2.cli run --input result.perception.json --output training.behaviours.json
```

Create a manifest (paths are relative to the manifest file):

```json
[
  {"result": "training.behaviours.json", "chunk_id": "1:0", "behaviour": "cmai_29_general_restlessness", "split": "train"}
]
```

```sh
python -m person2.cli build-prototypes --manifest labels.json --version annotated-v1 --output prototypes.json
python -m person2.cli run --input evaluation.perception.json --prototypes prototypes.json --config configs/person2.json --output evaluation.behaviours.json
python -m person2.cli validate evaluation.behaviours.json
```

Replace the illustrative chunk ID with an actual output ID. Quality-gated
chunks are rejected by the CLI prototype builder. Optional video must be used
consistently for both extraction and evaluation. CLI and schema validation errors
exit nonzero; an empty track produces empty chunks and events.
A missing `split` is rejected; an explicit validation/test split is
rejected. Never supply held-out annotations to this builder. The experiment
runner uses a separate manifest with required, audited train/validation/test
assignments and uses only its training rows for prototype construction.

## CMAI camera companion and training labels

P1/P2 result and prototype handoffs remain schema `1.0`. The camera application adds `cmai-camera-result-1.0` as a separate companion; see [camera contract](CMAI_CAMERA.md). New training manifests require explicit training splits and canonical item IDs (`cmai_01_pacing_aimless_wandering`, `cmai_26_repetitious_mannerisms`, `cmai_29_general_restlessness`). Old starter aliases migrate only during explicit result/bank review, not as newly labelled training manifests. The standalone prototype builder no longer defaults omitted splits to training. Experimental manifests use schema `2.0` and require taxonomy, detector mode, permissions and label-agreement provenance.
