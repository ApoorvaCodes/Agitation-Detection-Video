from person1.contracts import (BoundingBox, Landmark, MotionFeatures, MotionWindow,
                               ObservationQuality, PersonObservation, PoseData, TrackedPerson)
from cmai.demo_physical_behaviour import DemoPhysicalBehaviourConfig, DemoPhysicalBehaviourDetector


def _row(index, *, pose=True, features=None, points=None):
    points = points or {}
    landmarks = {name: Landmark(x=x, y=y, z=0, visibility=1, presence=1)
                 for name, (x, y) in points.items()}
    return PersonObservation(
        timestamp=index * .2, frame_index=index,
        bbox=BoundingBox(x_min=.1, y_min=.1, x_max=.9, y_max=.9),
        detection_confidence=1,
        normalized_pose=PoseData(landmarks=landmarks) if pose else None,
        motion=MotionFeatures(feature_values=features or {}),
        quality=ObservationQuality(detection_confidence=1, valid_landmarks=len(landmarks),
                                   pose_detected=pose,
                                   landmark_validity={name: True for name in landmarks}),
    )


def _person(rows, windows=None, person_id="person_a"):
    return TrackedPerson(person_id=person_id, observations=rows, windows=windows or [])


def _detector(**overrides):
    config = DemoPhysicalBehaviourConfig.from_path()
    values = {name: getattr(config, name) for name in ("hitting", "kicking", "pacing", "repetitive", "restlessness")}
    values.update(overrides)
    return DemoPhysicalBehaviourDetector(DemoPhysicalBehaviourConfig(
        enabled=True, min_valid_fraction=.5, min_event_duration=.2, **values))


def test_upper_limb_motion_emits_demo_hitting_candidate():
    rows = []
    for i in range(5):
        high = i in (1, 2, 3)
        rows.append(_row(i, features={
            "right_wrist.velocity.speed": 4 if high else .1,
            "right_wrist.acceleration.magnitude": 5 if high else .1,
        }, points={"right_shoulder": (.2, .5), "left_shoulder": (.0, .5),
                   "right_wrist": ((.5 if high else .25), .5)}))
    candidates = _detector().detect_person(_person(rows))
    assert [candidate.behaviour for candidate in candidates] == ["cmai_07_hitting"]
    assert candidates[0].candidate_status == "DEMO_ONLY"
    assert candidates[0].canonical_cmai_id == "cmai_07_hitting"
    assert candidates[0].canonical_cmai_name == "Hitting"
    assert candidates[0].score_semantics == "demo_rule_evidence_strength_not_probability"
    assert candidates[0].evidence["features"]["contact_status"] == "not_established"


def test_lower_limb_motion_emits_kicking_not_hitting():
    rows = []
    for i in range(5):
        high = i in (1, 2, 3)
        rows.append(_row(i, features={
            "right_ankle.velocity.speed": 4 if high else .1,
            "right_ankle.acceleration.magnitude": 5 if high else .1,
            "right_knee.angular_velocity": 1000 if high else .1,
        }, points={"right_hip": (.2, .2), "right_knee": (.2, .5),
                   "right_ankle": ((.2 if high else .2), (.9 if high else .8))}))
    candidates = _detector().detect_person(_person(rows, person_id="renamed_track"))
    assert [candidate.behaviour for candidate in candidates] == ["cmai_08_kicking"]
    assert candidates[0].canonical_cmai_id == "cmai_08_kicking"
    assert candidates[0].canonical_cmai_name == "Kicking"
    assert all(candidate.behaviour != "cmai_29_general_restlessness" for candidate in candidates)


def test_sustained_path_and_direction_changes_emit_pacing():
    points = [(.1, .5), (.5, .5), (.9, .5), (.5, .5), (.1, .5)]
    rows = [_row(i, points={"left_hip": point, "right_hip": (point[0] + .05, point[1])})
            for i, point in enumerate(points)]
    window = MotionWindow(start_timestamp=0, end_timestamp=4, valid_fraction=1,
                          low_quality=False, features={
                              "body_centroid.cumulative_path_length": 2.0,
                              "body_centroid.net_displacement": .05,
                              "body_centroid.net_path_ratio": .025,
                          })
    candidates = _detector().detect_person(_person(rows, [window]))
    assert [candidate.behaviour for candidate in candidates] == ["cmai_01_pacing_aimless_wandering"]


def test_repeated_wrist_peaks_emit_repetitive_mannerism():
    speeds = [.1, 3, .1, .1, 3, .1, .1, 3, .1]
    rows = [_row(i, features={"right_wrist.velocity.speed": speed}) for i, speed in enumerate(speeds)]
    candidates = _detector().detect_person(_person(rows))
    assert [candidate.behaviour for candidate in candidates] == ["cmai_26_repetitious_mannerisms"]
    assert candidates[0].evidence["features"]["repetition_count"] == 3


def test_sustained_generic_energy_emits_restlessness_only_as_fallback():
    rows = [_row(i) for i in range(16)]
    window = MotionWindow(start_timestamp=0, end_timestamp=3, valid_fraction=1,
                          low_quality=False, features={"overall_motion_energy": 1.0,
                                                       "arm_motion_energy": .5,
                                                       "leg_motion_energy": .5})
    candidates = _detector().detect_person(_person(rows, [window]))
    assert [candidate.behaviour for candidate in candidates] == ["cmai_29_general_restlessness"]


def test_invalid_landmarks_abstain_and_one_action_does_not_become_pacing():
    rows = [_row(i, pose=False) for i in range(8)]
    window = MotionWindow(start_timestamp=0, end_timestamp=3, valid_fraction=0,
                          low_quality=True, features={"overall_motion_energy": 100})
    assert DemoPhysicalBehaviourDetector().detect_person(_person(rows, [window])) == []


def test_detection_is_independent_of_track_identifier_and_source_path():
    rows = [_row(i, features={"right_wrist.velocity.speed": 3 if i in (1, 2, 3) else .1,
                              "right_wrist.acceleration.magnitude": 5 if i in (1, 2, 3) else .1},
                  points={"right_shoulder": (.2, .5), "left_shoulder": (.0, .5),
                          "right_wrist": ((.5 if i in (1, 2, 3) else .25), .5)}) for i in range(5)]
    first = DemoPhysicalBehaviourDetector().detect_person(_person(rows, person_id="one"))
    second = DemoPhysicalBehaviourDetector().detect_person(_person(rows, person_id="unseen_track"))
    assert [(c.behaviour, c.start_timestamp, c.end_timestamp) for c in first] == [(c.behaviour, c.start_timestamp, c.end_timestamp) for c in second]
