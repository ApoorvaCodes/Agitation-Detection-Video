# Person 1 → Person 2 contract (schema 1.0)

The schema remains `1.0`. Additions are optional; existing fields and meanings are preserved. Person IDs are session-local tracker IDs, not identities. Missing values are JSON `null`, never zero-filled.

## Coordinates and timestamps

`bbox` uses image-normalized coordinates: `(x_min,y_min)` is top-left and `(x_max,y_max)` bottom-right. Raw pose landmarks are full-frame image-normalized coordinates after crop-to-frame mapping. `normalized_pose` is body-relative: translated to mid-hip and divided by shoulder width, with existing documented fallbacks. Timestamps are decoder timestamps in seconds, with frame-index/FPS fallback. Derivatives use actual timestamp differences.

## Per-frame feature ordering

`video.feature_names` is authoritative and deterministic. It contains, in this exact order:

1. For each key joint in order (`nose`, left/right shoulders, elbows, wrists, hips, knees, ankles): `displacement.x`, `.y`, `.z`, `.magnitude`; `velocity.x`, `.y`, `.z`, `.speed`; `acceleration.x`, `.y`, `.z`, `.magnitude`.
2. `left_wrist.jerk`, `right_wrist.jerk`, `body_centroid.jerk`, `bbox_center.displacement`, `body_centroid.displacement`, `body_centroid.velocity.speed`, `body_centroid.acceleration.magnitude`.
3. `{joint}.angular_velocity` for the eight documented joint triplets.

Units are body-normalized-coordinate units, units/second, units/second², units/second³, bbox-normalized units, degrees/second, and degrees for underlying angles. `feature_values` uses these names; null means unavailable. `quality.feature_validity` gives a boolean for every name.

## Windows

`video.window_feature_names` is authoritative. Windows are `[start_timestamp,end_timestamp]`, with configured duration (default 2 seconds) and step `duration * (1-overlap)` (default 50% overlap). `valid_fraction` is valid frames divided by frames in the window; `low_quality` is true below the configured default 0.60 threshold. Low-quality windows are retained.

The ordered window names are: `mean_speed`, `std_speed`, `min_speed`, `max_speed`, `speed_energy`, `mean_acceleration`, `std_acceleration`, `min_acceleration`, `max_acceleration`, `acceleration_energy`, `overall_motion_energy`, `arm_motion_energy`, `leg_motion_energy`, `wrist_speed_dominant_frequency`, `wrist_speed_zero_crossing_rate`, followed by `{body_centroid,left_wrist,right_wrist,bbox_center}.{cumulative_path_length,net_displacement,net_path_ratio}`.

Speed and acceleration summaries are arithmetic statistics over valid joint samples. Energy is mean square. Dominant frequency uses the non-DC FFT bin of wrist speeds when at least four samples exist; zero-crossing rate is sign changes around the sample mean divided by elapsed seconds. Path length is the sum of consecutive valid point distances; net displacement is first-to-last distance; ratio is net/path and null for zero path. Bbox-centre displacement is the distance between consecutive bbox centres in image-normalized coordinates. These are geometric features, not behaviour labels.

## NPZ and loading

`save_perception_npz` writes `landmarks_norm [T,33,4]`, `features [T,F]`, `feature_valid_masks [T,F]`, `timestamps [T]`, and `valid_masks [T,33]`, where `F=len(video.feature_names)` and rows follow serialized person/observation order. Missing feature values are NaN in the numerical NPZ matrix and false in `feature_valid_masks`; JSON uses null. Load JSON with `person1.io.load_perception` and validate with `python -m person1.cli validate result.perception.json`.

Short detection gaps up to `max_gap_frames` (default 10) receive interpolated bboxes, `tracking_status="interpolated"`, and `bbox_interpolated=true`; pose and motion remain unavailable for inserted frames. Temporal derivatives never cross invalid pose samples or gaps greater than `max_temporal_gap_seconds`.

Person 1 does not produce CMAI labels, clinical scores, probabilities, embeddings, or behaviour classifications.
