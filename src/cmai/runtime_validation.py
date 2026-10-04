"""Run the actual local video worker and persist auditable runtime evidence."""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys

from cmai.bundle import load_bundle
from cmai.contracts import CameraResult
from cmai.action_detection import ActionEvidenceResult
from cmai.interactions import InteractionEvidence
from cmai.results import build_camera_result, create_evidence, export_archive
from cmai.worker import run
from person1.config import Person1Config
from person1.contracts import Person1VideoResult
from person2.contracts import Person2VideoResult
from person2.experiments import runtime_metadata


def validate_video(input_path, output_dir, config_path, bundle_path=None, interaction_path=None, input_kind='real_unlabelled'):
    video=Path(input_path).resolve()
    if not video.is_file():
        raise ValueError('source video does not exist')
    output=Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('use an empty output directory to preserve earlier validation evidence')
    config=Person1Config.from_yaml(config_path)
    model=Path(config.yolo_model).resolve()
    if not model.is_file():
        raise ValueError('provide an existing local YOLO checkpoint; validation does not download weights')
    bundle=load_bundle(bundle_path)
    contacts=InteractionEvidence.model_validate_json(Path(interaction_path).read_text()) if interaction_path else None
    recording_hash=sha256(video.read_bytes()).hexdigest()
    from dataclasses import replace
    config=replace(config,yolo_model=str(model),video_id='video-'+recording_hash[:16])
    output.mkdir(parents=True,exist_ok=True)
    request=dict(path=str(video),p1_configuration=asdict(config),bundle=bundle.metadata.model_dump(mode='json'),
                 bundle_sha256=bundle.bundle_sha256,bank=bundle.bank.model_dump() if bundle.bank else None,
                 action_model=bundle.action_model.model_dump() if bundle.action_model else None,
                 checkpoint_path=str(bundle.checkpoint_path) if bundle.checkpoint_path else None,
                 interactions=contacts.model_dump() if contacts else None)
    request_path=output/'request.json';request_path.write_text(json.dumps(request,indent=2)+'\n')
    report=dict(schema_version='cmai-runtime-validation-1.0',input_kind=input_kind,recording_sha256=recording_hash,
                runtime=runtime_metadata(),python_executable=sys.executable,detector_bundle_sha256=bundle.bundle_sha256,
                yolo_checkpoint_sha256=sha256(model.read_bytes()).hexdigest(),status='pending',
                limitations=['Runtime smoke validation, not labelled detector performance or clinical validation.',
                             'Human identity, contact assertions, review decisions and action accuracy require separate inspection.'])
    run(request_path)
    progress=json.loads((output/'progress.json').read_text())
    report['worker']=progress
    if progress['phase']!='complete':
        report['status']='failed'
    else:
        p1=Person1VideoResult.model_validate_json((output/'perception.json').read_text())
        p2=Person2VideoResult.model_validate_json((output/'candidates.json').read_text())
        sidecar=output/'action-assessments.json'
        checks=ActionEvidenceResult.model_validate_json(sidecar.read_text()) if sidecar.exists() else None
        camera=build_camera_result(p1,p2,bundle,recording_hash,checks)
        camera=create_evidence(camera,video,output/'evidence')
        (output/'camera-result.json').write_text(camera.model_dump_json(indent=2)+'\n')
        (output/'evidence.zip').write_bytes(export_archive(camera,output/'evidence'))
        CameraResult.model_validate_json((output/'camera-result.json').read_text())
        observations=[o for p in p1.persons for o in p.observations]
        report.update(status='completed',person_tracks=len(p1.persons),observations=len(observations),
                      pose_observations=sum(bool(o.pose and o.pose.landmarks) for o in observations),
                      valid_landmarks=sum(sum(o.quality.landmark_validity.values()) for o in observations),
                      candidates=len(camera.events),clip_statuses=[e.evidence.clip_status for e in camera.events],
                      enabled_items=[r.item_id for r in bundle.metadata.rules],
                      abstention_counts={s:sum(c.status==s for p in p2.persons for c in p.chunks)
                                         for s in ['scored','low_quality','no_prototypes','insufficient_evidence']})
    (output/'runtime-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--config',default='configs/default.yaml')
    parser.add_argument('--bundle')
    parser.add_argument('--interactions')
    parser.add_argument('--input-kind',choices=['real_unlabelled','permitted_labelled','synthetic_mechanics_only'],default='real_unlabelled')
    args=parser.parse_args()
    report=validate_video(args.input,args.output_dir,args.config,args.bundle,args.interactions,args.input_kind)
    print(json.dumps({k:report.get(k) for k in ['status','worker','person_tracks','pose_observations','candidates']},indent=2))
    if report['status']!='completed':
        raise SystemExit(1)


if __name__=='__main__':main()
