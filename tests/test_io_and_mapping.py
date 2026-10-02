import json
import numpy as np
from person1.io import load_perception, save_perception_npz
from person1.pipeline import _map_pose
from person1.contracts import Person1VideoResult, VideoMetadata, TrackedPerson, PersonObservation, BoundingBox, ObservationQuality, MotionFeatures
from person1.features import feature_names
def test_crop_full_frame_mapping_round_trip():
    mapped=_map_pose({'w':{'x':.25,'y':.5,'z':0}},(.2,.1,.6,.9),(100,200,3)); assert abs(mapped['w']['x']-.3)<1e-9 and abs(mapped['w']['y']-.5)<1e-9
def test_loader_rejects_nonfinite_json(tmp_path):
    path=tmp_path/'x.json'; path.write_text('{"schema_version":"1.0","video":{"video_id":"x","source_path":"x","fps":1,"width":1,"height":1},"persons":[]}')
    assert load_perception(path).schema_version=='1.0'
def test_npz_empty_contract_has_stable_arrays(tmp_path):
    result=Person1VideoResult(video=VideoMetadata(video_id='x',source_path='x',fps=1,width=1,height=1)); path=tmp_path/'x.npz'; save_perception_npz(result,path); arrays=np.load(path); assert arrays['landmarks_norm'].shape==(0,0,4)
def test_npz_feature_dimensions_match_contract(tmp_path):
    names=feature_names(); obs=PersonObservation(timestamp=0,frame_index=0,bbox=BoundingBox(x_min=0,y_min=0,x_max=1,y_max=1),detection_confidence=1,quality=ObservationQuality(detection_confidence=1,valid_landmarks=0),motion=MotionFeatures(feature_values={n:float(i) for i,n in enumerate(names)})); result=Person1VideoResult(video=VideoMetadata(video_id='x',source_path='x',fps=1,width=1,height=1,feature_names=names),persons=[TrackedPerson(person_id='p',observations=[obs])]); path=tmp_path/'x.npz'; save_perception_npz(result,path); arrays=np.load(path); assert arrays['features'].shape==(1,len(names)) and arrays['feature_valid_masks'].shape==(1,len(names))
