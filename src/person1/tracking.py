from dataclasses import dataclass
from person1.perception import Detection
@dataclass
class _Track:
    track_id: str; detection: Detection; missed: int = 0
def _iou(a,b):
    x1=max(a.x_min,b.x_min); y1=max(a.y_min,b.y_min); x2=min(a.x_max,b.x_max); y2=min(a.y_max,b.y_max); inter=max(0,x2-x1)*max(0,y2-y1)
    area=lambda d:max(0,d.x_max-d.x_min)*max(0,d.y_max-d.y_min); union=area(a)+area(b)-inter; return inter/union if union else 0
class IoUTracker:
    """Deterministic fallback tracker; production deployments may replace it with ByteTrack."""
    def __init__(self,iou_threshold=.2,max_missed=5): self.iou_threshold=iou_threshold; self.max_missed=max_missed; self._tracks=[]; self._next=1
    def update(self,detections):
        unmatched=set(range(len(detections))); matches=[]
        for track in self._tracks:
            options=[(i,_iou(track.detection,detections[i])) for i in unmatched]
            if options:
                i,score=max(options,key=lambda x:x[1])
                if score>=self.iou_threshold: matches.append((track,i)); unmatched.remove(i)
        result=[]
        for track,i in matches: track.detection=detections[i]; track.missed=0; result.append((track.track_id,detections[i]))
        for track in self._tracks:
            if not any(track is item[0] for item in matches): track.missed+=1
        self._tracks=[t for t in self._tracks if t.missed<=self.max_missed]
        for i in sorted(unmatched):
            track=_Track(f"person_{self._next:04d}",detections[i]); self._next+=1; self._tracks.append(track); result.append((track.track_id,detections[i]))
        return sorted(result)
