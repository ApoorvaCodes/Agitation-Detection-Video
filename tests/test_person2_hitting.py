"""Synthetic mechanics tests for the experimental, non-clinical punch baseline."""
import math

import pytest

from person1.contracts import Landmark, PoseData, TrackedPerson
from person2.config import Person2Config
from person2.hitting import HittingConfig, analyze_hitting
from person2.pipeline import process_perception
from test_person2 import observation, source


def punch_positions(kind, count=31):
    if kind == "punch":
        extension = [0.0] * 6 + [.03, .12, .38, .78, 1.08, 1.15, 1.10, .92, .68, .43, .23, .10] + [.08] * 14
    elif kind == "slow":
        extension = [0.0] * 6 + [min(.7, (i-5)*.07) for i in range(6,16)] + [.7]*15
    elif kind == "walking":
        extension = [.16*math.sin(2*math.pi*i/16) for i in range(count)]
    elif kind == "jitter":
        extension = [.025*((i%3)-1) for i in range(count)]
    else:
        extension = [0.0]*count
    return extension[:count]


def make_track(kind="punch", side="left", person_id="p1"):
    positions=punch_positions(kind)
    observations=[]
    for index,extension in enumerate(positions):
        landmarks={}
        for arm in ("left","right"):
            sign=-1 if arm=="left" else 1
            shoulder=(sign*.5,-.4,0.0)
            arm_extension=extension if arm==side else 0.0
            wrist=(sign*(.7+arm_extension),-.30,0.0)
            elbow=(sign*(.6+arm_extension*.45),-.35,0.0)
            for name,point in ((f"{arm}_shoulder",shoulder),(f"{arm}_elbow",elbow),(f"{arm}_wrist",wrist)):
                landmarks[name]=Landmark(x=point[0],y=point[1],z=point[2],visibility=.95)
        for hip,point in (("left_hip",(-.35,.6,0)),("right_hip",(.35,.6,0))):
            landmarks[hip]=Landmark(x=point[0],y=point[1],z=point[2],visibility=.95)
        sample=observation(index/10,index,good=True,x=0.0)
        quality=sample.quality.model_copy(update={"pose_detected":True,"bbox_interpolated":False,
            "detection_confidence":.95,"landmark_validity":{name:True for name in landmarks}})
        observations.append(sample.model_copy(update={"normalized_pose":PoseData(landmarks=landmarks),"quality":quality}))
    return TrackedPerson(person_id=person_id,observations=observations)


def source_for(kind="punch", side="left"):
    data=source([])
    data.video.fps=10
    data.video.processed_fps=10
    data.persons=[make_track(kind,side)]
    return data


def test_punch_pattern_generates_person2_candidate_and_preserves_arm_and_features():
    from person1.features import enrich_observations
    data=source_for("punch","right")
    track=data.persons[0]
    enrich_observations(track.observations,max_gap_seconds=.5,min_delta_time=.001)
    assert max(o.motion.feature_values["right_wrist.velocity.speed"] or 0 for o in track.observations)>1.25
    assert max(o.motion.feature_values["right_wrist.acceleration.magnitude"] or 0 for o in track.observations)>4.0
    found,debug=analyze_hitting(track,HittingConfig())
    assert found
    assert found[0]["arm_side"]=="right"
    assert found[0]["candidate_score"]>=HittingConfig().min_score
    assert found[0]["evidence"]["peak_wrist_velocity"]>=HittingConfig().min_velocity
    assert found[0]["evidence"]["extension_change"]>=HittingConfig().min_extension_change
    p2=process_perception(data,Person2Config(window_seconds=1,overlap=.5,min_frames=3,video_weight=0))
    candidate=next(e for e in p2.persons[0].events if e.behaviour=="cmai_07_hitting")
    assert candidate.candidate_source=="motion_baseline" and candidate.arm_side=="right"
    assert candidate.evidence["observation_features"]
    assert debug["max_right_wrist_velocity"]>debug["max_left_wrist_velocity"]


def test_contactless_punch_reaches_p3_and_camera_result_without_groq():
    from unittest.mock import Mock
    from person3.contracts import Verification
    from person3.pipeline import validate_p2_result
    from person3.evidence import candidates_from_p2
    from cmai.bundle import load_bundle
    from cmai.results import build_camera_result

    data=source_for("punch")
    p2=process_perception(data,Person2Config(window_seconds=1,overlap=.5,min_frames=3,video_weight=0))
    candidate=next(e for e in p2.persons[0].events if e.behaviour=="cmai_07_hitting")
    assert candidate.candidate_source=="motion_baseline"
    p3_candidates=candidates_from_p2(p2,data)
    assert p3_candidates[0].evidence["arm_side"]=="left"
    verifier=Mock()
    verifier.validate.side_effect=lambda packet: Verification(decision="supported",reason="Mock evidence review",
        evidence_segment_ids=[segment.evidence_id for segment in packet.segments])
    events=validate_p2_result(data,p2,verifier=verifier)
    supported=next(event for event in events if event.validation_status=="supported")
    assert supported.behaviour=="cmai_07_hitting"
    assert supported.start_timestamp==min(o.timestamp for o in data.persons[0].observations
        if f"p1:frame:{o.frame_index}" in supported.selected_evidence_ids)
    camera=build_camera_result(data,p2,load_bundle())
    assert camera.events and camera.events[0].cmai_item_id=="cmai_07_hitting"
    assert camera.events[0].score_semantics=="heuristic_motion_score"
    assert camera.events[0].evidence_check["arm_side"]=="left"
    assert camera.events[0].evidence_check["status"]=="reviewable"


@pytest.mark.parametrize("kind",["still","slow","walking","jitter"])
def test_non_punch_motion_does_not_generate_hitting(kind):
    data=source_for(kind)
    found,_=analyze_hitting(data.persons[0],HittingConfig())
    assert not found


def test_stationary_motion_diagnostics_have_zero_derivatives_and_real_coordinates():
    track=make_track("still")
    found,debug=analyze_hitting(track,HittingConfig())
    assert not found
    assert debug["motion_observations"] > 0
    assert debug["max_left_wrist_velocity"] == 0
    assert debug["max_left_acceleration"] == 0
    assert debug["first_left_wrist_x"] == debug["last_left_wrist_x"]


def test_body_scale_change_preserves_candidate_and_normalized_motion():
    original=make_track("punch")
    scaled=make_track("punch")
    for obs in scaled.observations:
        for point in obs.normalized_pose.landmarks.values():
            point.x *= 2.7; point.y *= 2.7; point.z *= 2.7
    original_events,original_debug=analyze_hitting(original,HittingConfig())
    scaled_events,scaled_debug=analyze_hitting(scaled,HittingConfig())
    assert original_events and scaled_events
    assert scaled_events[0]["arm_side"] == original_events[0]["arm_side"]
    assert scaled_debug["max_left_wrist_velocity"] == pytest.approx(original_debug["max_left_wrist_velocity"])
    assert scaled_debug["max_left_acceleration"] == pytest.approx(original_debug["max_left_acceleration"])


def test_two_people_remain_separate_and_missing_or_interpolated_samples_break_motion():
    data=source_for("punch")
    still=make_track("still","left","p2")
    data.persons.append(still)
    data.persons[0].observations[20].quality.bbox_interpolated=True
    p2=process_perception(data,Person2Config(window_seconds=1,overlap=.5,min_frames=3,video_weight=0))
    assert any(e.behaviour=="cmai_07_hitting" for e in next(p for p in p2.persons if p.person_id=="p1").events)
    assert not next(p for p in p2.persons if p.person_id=="p2").events
