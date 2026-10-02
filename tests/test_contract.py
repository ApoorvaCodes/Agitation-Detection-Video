import pytest
from person1.contracts import BoundingBox, Person1VideoResult, TrackedPerson, VideoMetadata
def test_contract_round_trip():
    x=Person1VideoResult(video=VideoMetadata(video_id="x",source_path="x.mp4",fps=30,width=10,height=10),persons=[TrackedPerson(person_id="person_0001")]); assert Person1VideoResult.model_validate_json(x.model_dump_json())==x and x.schema_version=="1.0"
def test_bbox_rejects_pixel_coordinates():
    with pytest.raises(ValueError): BoundingBox(x_min=-1,y_min=0,x_max=1,y_max=1)
