from pathlib import Path
import cv2
from person1.contracts import Person1VideoResult
def render_overlay(video_path: str|Path,result: Person1VideoResult,output_path: str|Path) -> None:
    cap=cv2.VideoCapture(str(video_path)); fps=cap.get(cv2.CAP_PROP_FPS) or 30; w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); writer=cv2.VideoWriter(str(output_path),cv2.VideoWriter_fourcc(*"MJPG"),fps,(w,h)); by_frame={}
    for person in result.persons:
        for obs in person.observations: by_frame.setdefault(obs.frame_index,[]).append((person.person_id,obs))
    index=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        for person_id,obs in by_frame.get(index,[]):
            b=obs.bbox; p1=(int(b.x_min*w),int(b.y_min*h)); p2=(int(b.x_max*w),int(b.y_max*h)); cv2.rectangle(frame,p1,p2,(0,180,0),2); cv2.putText(frame,person_id,(p1[0],max(15,p1[1]-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,120,0),1)
            if obs.pose:
                for landmark in obs.pose.landmarks.values(): cv2.circle(frame,(int(landmark.x*w),int(landmark.y*h)),2,(0,0,220),-1)
            speed=obs.motion.body_center.magnitude if obs.motion and obs.motion.body_center else None
            if speed is not None: cv2.putText(frame,f"body_speed={speed:.3f}",(p1[0],p2[1]+15),cv2.FONT_HERSHEY_SIMPLEX,.45,(180,0,0),1)
        writer.write(frame); index+=1
    cap.release(); writer.release()
