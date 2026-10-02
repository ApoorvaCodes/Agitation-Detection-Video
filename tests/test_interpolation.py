from person1.pipeline import _interpolate_observations
from test_features import _obs
def test_short_gap_interpolation_metadata():
    result=_interpolate_observations([_obs(0,0),_obs(.3,3)],1)
    assert len(result)==2
    result=_interpolate_observations([_obs(0,0),_obs(.3,3)],3)
    assert len(result)==4 and result[1].quality.bbox_interpolated and result[1].quality.pose_detected is False
