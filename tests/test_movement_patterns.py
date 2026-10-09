import pytest
from person1.contracts import BoundingBox, Landmark, ObservationQuality, PersonObservation, PoseData, TrackedPerson
from person2.movement_patterns import MovementPatternConfig, analyze_movement_patterns


def make_track(xs, *, scale=0.5, times=None, arm=None, pose_visible=True):
    times = times or [float(i) for i in range(len(xs))]
    rows=[]
    for i,(t,x) in enumerate(zip(times,xs)):
        bbox=BoundingBox(x_min=max(0,x-scale*.16),y_min=.1,x_max=min(1,x+scale*.16),y_max=.1+scale)
        lm={}
        shoulder_shift=0
        if arm:
            # Four short bursts separated by still intervals.
            phase=i % 13
            if phase in (0,1,2): shoulder_shift=arm[i] if i<len(arm) else 0
        for side,sign in (("left",-1),("right",1)):
            for name,point in ((f"{side}_shoulder",(sign*.3,-.3+shoulder_shift,0)),
                               (f"{side}_elbow",(sign*.45,-.1+shoulder_shift,0)),
                               (f"{side}_wrist",(sign*(.5+(arm[i] if arm else 0)),-.05+shoulder_shift,0)),
                               (f"{side}_hip",(sign*.2,.25,0)),
                               (f"{side}_knee",(sign*.2,.45,0)),
                               (f"{side}_ankle",(sign*.2,.65,0))):
                if pose_visible or "wrist" not in name:
                    lm[name]=Landmark(x=point[0],y=point[1],z=point[2],visibility=.95)
        quality=ObservationQuality(detection_confidence=.95,valid_landmarks=len(lm),pose_detected=pose_visible,
            landmark_validity={n:True for n in lm})
        rows.append(PersonObservation(timestamp=t,frame_index=i,bbox=bbox,detection_confidence=.95,
            pose=PoseData(landmarks=lm) if pose_visible else None,
            normalized_pose=PoseData(landmarks=lm) if pose_visible else None,quality=quality))
    return TrackedPerson(person_id="track-a",observations=rows)


def config(**changes):
    base=dict(min_track_seconds=6,min_valid_coverage=.5,min_pacing_path_body_scales=1.5,
              min_pacing_reversals=2,min_recurrence=.2,min_pose_frames=6,
              min_restlessness_bursts=3,min_moving_fraction=.08)
    base.update(changes)
    return MovementPatternConfig(**base)


def test_stationary_and_single_straight_walk_are_not_pacing():
    stationary=make_track([.5]*13)
    walk=make_track([.2+i*.05 for i in range(13)])
    for track in (stationary,walk):
        events,diag=analyze_movement_patterns(track,config())
        assert not any(e[0]=="cmai_01_pacing_aimless_wandering" for e in events)
    assert "insufficient_direction_reversals" in analyze_movement_patterns(walk,config())[1]["pacing"]["abstention_reasons"]


def test_repeated_back_and_forth_is_pacing_and_traceable():
    track=make_track([.2,.4,.6,.8,.6,.4,.2,.4,.6,.8,.6,.4,.2])
    events,diag=analyze_movement_patterns(track,config())
    event=next(e for e in events if e[0]=="cmai_01_pacing_aimless_wandering")
    assert event[4]["direction_reversals"]>=2
    assert len(event[4]["source_observation_ids"])==13
    assert event[4]["path_length_body_scales"]>event[4]["net_displacement_body_scales"]


def test_short_track_abstains_and_irregular_gaps_are_safe():
    short=make_track([.2,.6,.2],times=[0,.2,5])
    events,diag=analyze_movement_patterns(short,config())
    assert not events
    assert "track_duration_below_minimum" in diag["pacing"]["abstention_reasons"]


def test_pacing_is_normalized_for_person_scale():
    xs=[.2,.4,.6,.8,.6,.4,.2,.4,.6,.8,.6,.4,.2]
    small=make_track(xs,scale=.5)
    big=make_track([.5+(x-.5)*.5 for x in xs],scale=.25)
    a=analyze_movement_patterns(small,config())[0][0][4]["path_length_body_scales"]
    b=analyze_movement_patterns(big,config())[0][0][4]["path_length_body_scales"]
    assert a==pytest.approx(b,rel=.15)


def burst_track():
    times=[i*.25 for i in range(49)]
    values=[]
    for i in range(len(times)):
        cycle=i%13
        values.append(.22 if cycle in (1,2,3) else 0.0)
    return make_track([.5]*len(times),times=times,arm=values)


def test_stationary_and_isolated_arm_gesture_are_not_restlessness():
    stationary=make_track([.5]*49,times=[i*.25 for i in range(49)])
    one=[0.0]*49; one[4]=.25; one[5]=.25
    isolated=make_track([.5]*49,times=[i*.25 for i in range(49)],arm=one)
    for track in (stationary,isolated):
        events,_=analyze_movement_patterns(track,config())
        assert not any(e[0]=="cmai_29_general_restlessness" for e in events)


def test_repeated_stationary_pose_motion_generates_restlessness():
    events,diagnostics=analyze_movement_patterns(burst_track(),config())
    event=next(e for e in events if e[0]=="cmai_29_general_restlessness")
    assert event[4]["repeated_movement_burst_count"]>=3
    assert event[4]["upper_body_movement_frequency"]>0
    assert event[4]["source_observation_ids"]


def test_missing_pose_joints_and_low_coverage_abstain_without_fabrication():
    no_pose=make_track([.5]*12,pose_visible=False)
    events,diagnostics=analyze_movement_patterns(no_pose,config())
    assert not any(e[0]=="cmai_29_general_restlessness" for e in events)
    assert "valid_pose_frames_below_minimum" in diagnostics["restlessness"]["abstention_reasons"]


def test_clear_straight_locomotion_is_not_restlessness():
    track=make_track([.2+i*.018 for i in range(25)],times=[i*.5 for i in range(25)],arm=[.2 if i%2 else 0 for i in range(25)])
    events,diagnostics=analyze_movement_patterns(track,config())
    assert not any(e[0]=="cmai_29_general_restlessness" for e in events)
    assert "clear_locomotion_excluded_from_restlessness" in diagnostics["restlessness"]["abstention_reasons"]

def test_repeated_posture_shifts_count_as_repeated_pose_evidence():
    values=[]
    for i in range(49):
        values.append(.16 if i%13 in (1,2,3) else 0.0)
    track=make_track([.5]*49,times=[i*.25 for i in range(49)],arm=values)
    events,diag=analyze_movement_patterns(track,config())
    event=next(e for e in events if e[0]=="cmai_29_general_restlessness")
    assert event[4]["posture_change_count"]>=3


def test_p2_candidate_keeps_observation_ids_and_timestamps():
    from unittest.mock import Mock
    from person3.contracts import Verification
    from person2.config import Person2Config
    from person2.hitting import HittingConfig
    from person2.pipeline import process_perception
    from person3.pipeline import validate_p2_result
    from test_person2 import source
    from person3.evidence import candidates_from_p2
    from cmai.bundle import load_bundle
    from cmai.results import build_camera_result
    p1=source([])
    p1.video.fps=1
    p1.video.processed_fps=1
    p1.persons=[make_track([.2,.4,.6,.8,.6,.4,.2,.4,.6,.8,.6,.4,.2])]
    p2=process_perception(p1,Person2Config(),hitting_config=HittingConfig(enabled=False),movement_config=config())
    candidate=next(c for c in candidates_from_p2(p2,p1) if c.behaviour=="cmai_01_pacing_aimless_wandering")
    assert candidate.source_observation_ids
    assert candidate.start_timestamp==0
    assert candidate.end_timestamp==12
    assert candidate.source_window_ids
    verifier=Mock()
    verifier.validate.side_effect=lambda packet: Verification(decision="supported",reason="Synthetic source-linked movement evidence",
        evidence_segment_ids=[segment.evidence_id for segment in packet.segments])
    validated=validate_p2_result(p1,p2,verifier=verifier)
    assert any(event.behaviour=="cmai_01_pacing_aimless_wandering" and event.validation_status=="supported" for event in validated)
    camera=build_camera_result(p1,p2,load_bundle())
    camera_event=next(e for e in camera.events if e.cmai_item_id=="cmai_01_pacing_aimless_wandering")
    assert camera_event.evidence_check["candidate_source"]=="motion_baseline"
    assert camera_event.evidence_check["motion_features"]["detector"]=="pacing_trajectory_v1"
    assert camera_event.evidence.frame_indices



def test_one_posture_change_and_missing_wrist_or_elbow_do_not_trigger():
    one=[0.0]*49
    one[1]=.16; one[2]=.16
    posture=make_track([.5]*49,times=[i*.25 for i in range(49)],arm=one)
    for row in posture.observations:
        for name in ("left_wrist","right_wrist","left_elbow","right_elbow","left_ankle","right_ankle"):
            row.normalized_pose.landmarks.pop(name,None)
            row.quality.landmark_validity.pop(name,None)
    events,diagnostics=analyze_movement_patterns(posture,config())
    assert not any(e[0]=="cmai_29_general_restlessness" for e in events)
    assert diagnostics["restlessness"]["valid_landmark_coverage"]<.65


def test_restlessness_is_consistent_when_person_bbox_scale_changes():
    original=burst_track()
    smaller=make_track([.5]*49,scale=.25,times=[i*.25 for i in range(49)],
                       arm=[.22 if i%13 in (1,2,3) else 0 for i in range(49)])
    first=analyze_movement_patterns(original,config())[0]
    second=analyze_movement_patterns(smaller,config())[0]
    assert any(e[0]=="cmai_29_general_restlessness" for e in first)
    assert any(e[0]=="cmai_29_general_restlessness" for e in second)
