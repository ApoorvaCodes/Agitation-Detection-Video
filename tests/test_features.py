from person1.contracts import BoundingBox, ObservationQuality, PersonObservation, PoseData, Landmark
from person1.features import aggregate_windows, enrich_observations, feature_names, joint_angle, normalize_landmarks, vector_features, window_feature_names
def test_normalization_is_translation_invariant():
    a={"left_hip":{"x":0,"y":0,"z":0},"right_hip":{"x":2,"y":0,"z":0},"left_wrist":{"x":1,"y":1,"z":0}}; b={k:{**v,"x":v["x"]+10,"y":v["y"]-4} for k,v in a.items()}; assert normalize_landmarks(a)["left_wrist"]==normalize_landmarks(b)["left_wrist"]
def test_vector_features_use_actual_delta_time():
    x=vector_features((2,0,0),(0,0,0),(1,0,0),2,.001); assert x["displacement"]["magnitude"]==2 and x["velocity"]["x"]==1 and x["acceleration"]["x"]==0
def test_joint_angle_and_invalid_geometry():
    assert joint_angle((1,0,0),(0,0,0),(0,1,0))==90 and joint_angle((0,0,0),(0,0,0),(1,0,0)) is None

def _obs(t, x, valid=True):
    landmarks={"left_hip":Landmark(x=x,y=0,z=0),"right_hip":Landmark(x=x+1,y=0,z=0),"left_shoulder":Landmark(x=x,y=1,z=0),"right_shoulder":Landmark(x=x+1,y=1,z=0),"left_wrist":Landmark(x=x+0.2,y=0.5,z=0)} if valid else {}
    return PersonObservation(timestamp=t,frame_index=round(t*10),bbox=BoundingBox(x_min=0,y_min=0,x_max=1,y_max=1),detection_confidence=.9,normalized_pose=PoseData(landmarks=landmarks),quality=ObservationQuality(detection_confidence=.9,valid_landmarks=len(landmarks),pose_detected=valid))
def test_enrichment_uses_irregular_time_and_resets_gap():
    observations=[_obs(0,0),_obs(1,1),_obs(3,3),_obs(3.1,3.1)]
    enrich_observations(observations,1.5,.001)
    assert observations[1].motion.feature_values["left_wrist.velocity.speed"] == 1
    assert observations[2].motion.feature_values["left_wrist.velocity.speed"] is None
    assert observations[3].motion.feature_values["left_wrist.velocity.speed"] is not None
def test_jerk_and_angular_velocity_are_validity_aware():
    observations=[_obs(0,0),_obs(1,1),_obs(2,3),_obs(3,6)]
    enrich_observations(observations,2,.001)
    assert observations[3].motion.feature_values["left_wrist.jerk"] is not None
    assert observations[3].motion.feature_values["left_elbow.angular_velocity"] is None
def test_window_aggregation_and_low_quality():
    observations=[_obs(0,0),_obs(1,1),_obs(2,2),_obs(3,3)]
    enrich_observations(observations,2,.001); windows=aggregate_windows(observations,2,.5,.6)
    assert windows and "mean_speed" in windows[0].features and "body_centroid.net_path_ratio" in windows[0].features
    invalid=[_obs(0,0,False),_obs(1,1,False)]; enrich_observations(invalid,2,.001); assert aggregate_windows(invalid,2,.5,.6)[0].low_quality
def test_feature_order_is_fixed():
    assert feature_names()==feature_names() and window_feature_names()==window_feature_names(); assert len(feature_names())>50 and len(window_feature_names())>10
