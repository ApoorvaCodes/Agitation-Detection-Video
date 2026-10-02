"""Create a deterministic, non-human synthetic clip for mechanics testing."""
import argparse
import cv2
import numpy as np
def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",default="sample_synthetic.avi"); p.add_argument("--seconds",type=int,default=10); args=p.parse_args(); fps=10; size=(320,240)
    writer=cv2.VideoWriter(args.output,cv2.VideoWriter_fourcc(*"MJPG"),fps,size)
    for i in range(args.seconds*fps):
        frame=np.full((size[1],size[0],3),245,dtype=np.uint8); x1=40+(i*2)%180; x2=230-(i*2)%180
        cv2.rectangle(frame,(x1,70),(x1+35,190),(30,80,220),-1); cv2.circle(frame,(x1+18,50),18,(30,80,220),-1); cv2.rectangle(frame,(x2,80),(x2+35,200),(220,80,30),-1); cv2.circle(frame,(x2+18,60),18,(220,80,30),-1); writer.write(frame)
    writer.release(); print(args.output)
if __name__=="__main__": main()
