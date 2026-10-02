from person1.perception import Detection
from person1.tracking import IoUTracker
def d(x): return Detection(x,0,x+.2,.2,.9)
def test_tracker_keeps_ids_and_expires():
    tracker=IoUTracker(iou_threshold=.1,max_missed=2); first=tracker.update([d(0),d(.5)]); assert [x[0] for x in first]==["person_0001","person_0002"]
    assert tracker.update([d(.02)])[0][0]=="person_0001"; tracker.update([]); assert tracker.update([d(.5)])[0][0]=="person_0002"
