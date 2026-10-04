"""Optional local R3D-18 feature encoder. Never downloads or maps Kinetics to CMAI."""
from hashlib import sha256
from pathlib import Path
import cv2
import numpy as np
from person2.embeddings import packed, space_id


class LocalR3DEncoder:
    feature_names = [f"r3d18.feature.{i}" for i in range(512)]

    def __init__(self, video_path, checkpoint_path, specification, interactions=None, source=None):
        self.video_path = Path(video_path)
        if checkpoint_path is None:
            raise ValueError("Local R3D-18 checkpoint is missing; provide an explicit local path")
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise ValueError("Local R3D-18 checkpoint is missing; no checkpoint is downloaded automatically")
        if sha256(checkpoint_path.read_bytes()).hexdigest() != specification.checkpoint_sha256:
            raise ValueError("video checkpoint checksum mismatch")
        import torch
        import torchvision
        from torchvision.models.video import r3d_18, R3D_18_Weights
        self.torch = torch
        self.model = r3d_18(weights=None)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True), strict=True)
        self.model.fc = torch.nn.Identity()
        self.model.eval()
        self.transform = R3D_18_Weights.KINETICS400_V1.transforms()
        self.interactions = interactions
        self.actor_lookup = {id(o):p.person_id for p in source.persons for o in p.observations} if source else {}
        self.identity = {"model":specification.model,"version":"1","checkpoint_sha256":specification.checkpoint_sha256,
                         "source":specification.source,"torch":torch.__version__,"torchvision":torchvision.__version__,
                         "preprocessing":"actor_target_union_roi_rgb_official_transform_16_frames_15fps_v1"}
        self.space = space_id("local_r3d18_v1",self.feature_names,self.identity)

    def encode(self, observations):
        if not observations:
            return packed(self.space,[None]*512)
        cap = cv2.VideoCapture(str(self.video_path))
        frames = []
        try:
            fps,count = cap.get(cv2.CAP_PROP_FPS),cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if not cap.isOpened() or fps <= 0 or count <= 0:
                return packed(self.space,[None]*512)
            boxes = [o.bbox for o in observations if not o.quality.bbox_interpolated]
            frame_ids = {o.frame_index for o in observations}
            actor = self.actor_lookup.get(id(observations[0]))
            if actor is None:
                return packed(self.space,[None]*512)
            if self.interactions:
                boxes.extend(e.target_bbox for e in self.interactions.observations if e.person_id == actor and e.frame_index in frame_ids and e.target_visible)
            if not boxes:
                return packed(self.space,[None]*512)
            x0,y0,x1,y1 = min(b.x_min for b in boxes),min(b.y_min for b in boxes),max(b.x_max for b in boxes),max(b.y_max for b in boxes)
            for i in range(16):
                frame_index = round((observations[0].timestamp+i/15)*fps)
                if frame_index >= count:
                    return packed(self.space,[None]*512)
                cap.set(cv2.CAP_PROP_POS_FRAMES,frame_index)
                ok,frame = cap.read()
                if not ok:
                    return packed(self.space,[None]*512)
                h,w = frame.shape[:2]
                crop = frame[max(0,int(y0*h)):min(h,max(1,int(y1*h))),max(0,int(x0*w)):min(w,max(1,int(x1*w)))]
                if not crop.size:
                    return packed(self.space,[None]*512)
                frames.append(cv2.resize(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB),(171,128)))
            tensor = self.torch.from_numpy(np.stack(frames)).permute(0,3,1,2)
            with self.torch.inference_mode():
                vector = self.model(self.transform(tensor).unsqueeze(0)).squeeze(0).cpu().numpy()
            return packed(self.space,vector.tolist())
        finally:
            cap.release()
