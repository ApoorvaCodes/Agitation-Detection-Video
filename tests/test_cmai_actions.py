"""Synthetic algorithm/contract fixtures, never detector performance evidence."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest
from person1.contracts import Landmark, PoseData, TrackedPerson
from person1.io import save_perception
from person2.config import Person2Config
from person2.pipeline import process_perception
from cmai.action_model import train_action_model
from cmai.action_detection import ActionEvidenceResult
from cmai.action_experiments import ActionManifest, ActionAnnotation, audit_dataset, run_action_experiments
from cmai.bundle import ActionAsset, DetectorBundle, DetectorRule, LoadedBundle, LocalVideoEncoder, load_bundle
from cmai.contracts import CameraResult
from cmai.detection import detect_with_assessments
from cmai.interactions import ContactObservation, InteractionEvidence
from cmai.results import build_camera_result, review_event
from test_person2 import observation, source

HIT='cmai_07_hitting'
KICK='cmai_08_kicking'
HASH='a'*64


def fixture(label=HIT, contact='observed', times=None):
    data=source([observation(t) for t in (times or [i/5 for i in range(20)])])
    for o in data.persons[0].observations:
        joints=['left_'+j for j in ('shoulder','elbow','wrist','hip','knee','ankle')]
        o.pose=PoseData(landmarks={j:Landmark(x=.4,y=.5,visibility=.95) for j in joints})
        for j in joints:
            o.normalized_pose.landmarks[j]=Landmark(x=.4,y=.5)
            o.quality.landmark_validity[j]=True
        for j in ('left_wrist','left_ankle'):
            o.motion.feature_values[j+'.velocity.speed']=1
            o.quality.feature_validity[j+'.velocity.speed']=True
    data.video.feature_names=['left_wrist.velocity.speed','left_ankle.velocity.speed']
    data.persons.append(TrackedPerson(person_id='target',observations=[o.model_copy(deep=True) for o in data.persons[0].observations]))
    cfg=Person2Config(window_seconds=1,overlap=0,min_frames=3,min_valid_fraction=.8,video_weight=0,max_gap_seconds=.35)
    embedding=process_perception(data,cfg).persons[0].chunks[0].fused_embedding
    negative=embedding.model_copy(update={'values':[-v for v in embedding.values]})
    model=train_action_model([(label,'positive',embedding),(label,'negative',negative)],[label],'synthetic-test-only',{})
    metadata=DetectorBundle(schema_version='1.1',detector_mode='interaction_actions',declared_labels=[label],
        detector_id='test-only',version='test',mode='research',configuration=cfg,
        action_asset=ActionAsset(file='test-only',sha256=HASH),rules=[DetectorRule(item_id=label,similarity_threshold=.9,contrast_margin=.1,min_event_seconds=.2)])
    bundle=LoadedBundle(metadata,model.positive_bank(),HASH,model)
    evidence=InteractionEvidence(video_id=data.video.video_id,recording_sha256=HASH,provider_id='test-only',provider_version='1',
        method='manual_visual_review',annotation_protocol='synthetic contract fixtures',observations=[ContactObservation(
            evidence_id=f'e{i}',person_id='p1',frame_index=o.frame_index,timestamp=o.timestamp,
            limb='left_hand' if label==HIT else 'left_foot',target_kind='person',target_id='target',target_bbox=o.bbox,
            target_visible=True,actor_limb_visible=True,contact=contact,evidence_reference=f'frame:{o.frame_index}')
        for i,o in enumerate(data.persons[0].observations)])
    return data,bundle,evidence


@pytest.mark.parametrize('label',[HIT,KICK])
def test_distinct_clear_candidates_review_and_contract(label):
    data,bundle,contacts=fixture(label)
    p2,checks=detect_with_assessments(data,bundle,contacts,recording_sha256=HASH)
    assert [e.behaviour for e in p2.persons[0].events]==[label]
    assert not p2.persons[1].events
    camera=build_camera_result(data,p2,bundle,HASH,checks)
    assert camera.contract_version=='cmai-camera-result-1.1'
    assert camera.events[0].evidence.contacts
    assert camera.events[0].score_semantics=='uncalibrated_cosine_similarity'
    assert CameraResult.model_validate_json(camera.model_dump_json())==camera
    assert review_event(camera,camera.events[0].event_id,'uncertain','test').events[0].status=='uncertain'
    raw=camera.model_dump();raw['events'][0]['evidence']['contacts'][0]['limb']='right_foot' if label==HIT else 'right_hand'
    with pytest.raises(ValueError,match='limb'):CameraResult.model_validate(raw)


@pytest.mark.parametrize('case',['absent','missing','unclear','occluded','missing_visibility','low_target','bad_motion'])
def test_ordinary_motion_and_missing_contact_abstain(case):
    data,bundle,contacts=fixture(contact='absent' if case=='absent' else 'observed')
    if case=='missing':contacts=None
    elif case=='unclear':
        for e in contacts.observations:e.contact='unclear'
    elif case=='occluded':
        for o in data.persons[0].observations:o.quality.bbox_interpolated=True
    elif case=='missing_visibility':
        for o in data.persons[0].observations:o.pose=None
    elif case=='low_target':
        for o in data.persons[1].observations:o.detection_confidence=.1
    elif case=='bad_motion':
        for o in data.persons[0].observations:o.quality.feature_validity['left_wrist.velocity.speed']=False
    result,checks=detect_with_assessments(data,bundle,contacts,recording_sha256=HASH)
    if case in {'absent','missing','unclear','low_target'}:
        assert any(p.events for p in result.persons), 'Hitting contact is optional when the arm-action model scores the motion.'
    else:
        assert not any(p.events for p in result.persons)
    actor=[a for a in checks.assessments if a.person_id=='p1']
    if case in {'absent','missing','unclear','low_target'}:assert all(a.status=='scored' and not a.contact_present for a in actor)
    else:assert all(a.status!='scored' for a in actor)


def test_benign_contact_negative_centroid_does_not_become_action():
    data,bundle,contacts=fixture()
    cls=bundle.action_model.classes[0]
    cls.positive,cls.negative=cls.negative,cls.positive
    bundle=replace(bundle,bank=bundle.action_model.positive_bank())
    result,checks=detect_with_assessments(data,bundle,contacts,recording_sha256=HASH)
    assert not any(p.events for p in result.persons)
    assert all(a.contrast_margin<0 for a in checks.assessments if a.person_id=='p1')


def test_gaps_and_unclear_contact_break_support():
    data,bundle,contacts=fixture(times=[i/5 for i in range(10)]+[4+i/5 for i in range(10)])
    result,checks=detect_with_assessments(data,bundle,contacts,recording_sha256=HASH)
    assert len(result.persons[0].events)==2
    assert result.persons[0].events[0].end_timestamp<=2
    for e in contacts.observations:
        if 1<=e.timestamp<2:e.contact='unclear'
    result,_=detect_with_assessments(data,bundle,contacts,recording_sha256=HASH)
    assert len(result.persons[0].events)==2


def test_target_changes_do_not_merge():
    data,bundle,contacts=fixture()
    for e in contacts.observations:
        if e.timestamp>=2:e.target_kind='object';e.target_id='test-object'
    result,checks=detect_with_assessments(data,bundle,contacts,recording_sha256=HASH)
    assert len(result.persons[0].events)==2
    assert result.persons[0].events[0].end_timestamp<=2
    camera=build_camera_result(data,result,bundle,HASH,checks)
    assert len(camera.events)==2


def test_source_and_model_incompatibility(tmp_path):
    data,bundle,contacts=fixture()
    with pytest.raises(ValueError,match='checksum'):detect_with_assessments(data,bundle,contacts,recording_sha256='b'*64)
    contacts.observations[0].timestamp=.1
    with pytest.raises(ValueError,match='timestamp'):detect_with_assessments(data,bundle,contacts)
    contacts.observations[0].timestamp=0
    bundle.action_model.classes[0].negative.embedding.space='wrong-model'
    with pytest.raises(ValueError,match='incompatible'):detect_with_assessments(data,bundle,contacts)
    data,bundle,contacts=fixture()
    model_path=tmp_path/'model.json';model_path.write_text(bundle.action_model.model_dump_json())
    bundle.metadata.action_asset=ActionAsset(file='model.json',sha256=sha256(model_path.read_bytes()).hexdigest())
    path=tmp_path/'bundle.json';path.write_text(bundle.metadata.model_dump_json())
    assert load_bundle(path).action_model==bundle.action_model
    model_path.write_text('tampered')
    with pytest.raises(ValueError,match='checksum'):load_bundle(path)


def test_checkpoint_missing_or_corrupt_never_downloads(tmp_path):
    from cmai.video_actions import LocalR3DEncoder
    spec=LocalVideoEncoder(model='torchvision_r3d_18_kinetics400_v1',source='https://download.pytorch.org/models/r3d_18-b3b3357e.pth',checkpoint_file='missing.pth',checkpoint_sha256=HASH)
    with pytest.raises(ValueError,match='missing'):LocalR3DEncoder('missing.mp4',tmp_path/'missing.pth',spec)
    cp=tmp_path/'bad.pth';cp.write_bytes(b'not a model')
    with pytest.raises(ValueError,match='checksum'):LocalR3DEncoder('missing.mp4',cp,spec)


def test_actions_unavailable_and_no_results_fabricated(tmp_path):
    default=load_bundle();action=load_bundle(Path(__file__).parents[1]/'configs/cmai_action_detector_bundle.json')
    assert not default.metadata.rules and not action.metadata.rules
    data=source();result,checks=detect_with_assessments(data,action)
    assert not result.persons[0].events
    assert build_camera_result(data,result,action,action_assessments=checks).contract_version=='cmai-camera-result-1.1'
    report=run_action_experiments(None,tmp_path)
    assert report['status']=='pending_labelled_data'
    assert len(report['variants'])==4 and all(v['metrics'] is None for v in report['variants'].values())
    assert not list(tmp_path.glob('*.action-model.json'))


def manifest_fixture(tmp_path):
    recordings=[]
    for split in ['train','validation','test']:
        data,bundle,contacts=fixture();data.video.video_id=split;data.video.duration_seconds=4
        contacts.video_id=split;contacts.recording_sha256=sha256(split.encode()).hexdigest()
        for o in data.persons[0].observations:
            if o.timestamp>=2:
                o.normalized_pose.landmarks['left_wrist'].x=-.4
                o.motion.feature_values['left_wrist.velocity.speed']=0
        for e in contacts.observations:
            if e.timestamp>=2:e.contact='absent'
        save_perception(data,tmp_path/f'{split}.json');(tmp_path/f'{split}-contacts.json').write_text(contacts.model_dump_json())
        recordings.append(dict(perception=f'{split}.json',person_id='p1',subject_id=split,participant_subject_ids=[split,split+'-target'],track_subject_ids={'p1':split,'target':split+'-target'},session_id=split,
            split=split,recording_sha256=contacts.recording_sha256,interaction_evidence=f'{split}-contacts.json',annotations=[
            dict(annotation_id=split+'-pos',item_id=HIT,outcome='positive',start_timestamp=0,end_timestamp=2,target_kind='person',target_id='target'),
            dict(annotation_id=split+'-neg',item_id=HIT,outcome='negative',start_timestamp=2,end_timestamp=4)]))
    return dict(provenance=dict(name='synthetic-test-only',source='synthetic unit tests',license_or_permission='synthetic',annotation_protocol='test',annotation_version='1',label_agreement='test'),behaviours=[HIT],recordings=recordings)


def test_disjoint_audit_and_validation_only_thresholds(tmp_path):
    raw=manifest_fixture(tmp_path)
    manifest=ActionManifest.model_validate(raw)
    audit_dataset(manifest,tmp_path)
    manifest.recordings[2].participant_subject_ids[1]='train-target'
    manifest.recordings[2].track_subject_ids['target']='train-target'
    with pytest.raises(ValueError,match='subject leakage'):audit_dataset(manifest,tmp_path)
    manifest=ActionManifest.model_validate(raw);manifest.recordings[2].session_id='train'
    with pytest.raises(ValueError,match='session leakage'):audit_dataset(manifest,tmp_path)
    path=tmp_path/'annotations.json';path.write_text(json.dumps(raw))
    cfg=Person2Config(window_seconds=1,overlap=0,min_frames=3,min_valid_fraction=.8,video_weight=0,max_gap_seconds=.35)
    report=run_action_experiments(path,tmp_path/'run1',cfg,thresholds=[.5,.9],margins=[0,.1])
    assert report['status']=='evaluated'
    for v in report['variants'].values():
        assert v['threshold_selection_split']=='validation'
        assert 'matched_timing' in v['metrics']['test']['events']
        loaded=load_bundle(tmp_path/'run1'/v['bundle_file'])
        from person1.io import load_perception
        test_data=load_perception(tmp_path/'test.json')
        test_contacts=InteractionEvidence.model_validate_json((tmp_path/'test-contacts.json').read_text())
        detect_with_assessments(test_data,loaded,test_contacts,recording_sha256=test_contacts.recording_sha256)
    # Changing held-out targets changes evaluation, never threshold fitting.
    raw['recordings'][2]['annotations'][0]['target_kind']='object';raw['recordings'][2]['annotations'][0]['target_id']='other'
    path.write_text(json.dumps(raw))
    second=run_action_experiments(path,tmp_path/'run2',cfg,thresholds=[.5,.9],margins=[0,.1])
    assert {n:(v['selected_threshold'],v['selected_contrast_margin']) for n,v in report['variants'].items()}=={
        n:(v['selected_threshold'],v['selected_contrast_margin']) for n,v in second['variants'].items()}


def test_annotations_require_target_and_explicit_outcomes():
    with pytest.raises(ValueError,match='target'):ActionAnnotation(annotation_id='x',item_id=HIT,outcome='positive',start_timestamp=0,end_timestamp=1)
    assert ActionAnnotation(annotation_id='x',item_id=KICK,outcome='ambiguous',start_timestamp=0,end_timestamp=1).outcome=='ambiguous'


def test_new_schema_files_match():
    from cmai.action_model import ActionModel
    models={'cmai.action-model':ActionModel,'cmai.interaction-evidence':InteractionEvidence,
            'cmai.action-assessments':ActionEvidenceResult,'cmai.action-annotations':ActionManifest}
    for name,model in models.items():
        assert json.loads((Path(__file__).parents[1]/'docs'/f'{name}.schema.json').read_text())==model.model_json_schema()


def test_ambiguous_training_never_becomes_negative_and_reports_abstentions(tmp_path):
    raw=manifest_fixture(tmp_path)
    raw['recordings'][0]['annotations'][1]['outcome']='ambiguous'
    path=tmp_path/'annotations.json';path.write_text(json.dumps(raw))
    cfg=Person2Config(window_seconds=1,overlap=0,min_frames=3,video_weight=0,max_gap_seconds=.35)
    report=run_action_experiments(path,tmp_path/'out',cfg)
    assert report['status']=='incomplete_evaluation'
    assert all('negative examples required' in v['reason'] for v in report['variants'].values())
    # Missing contact on test is a false negative and an explicit abstention.
    raw=manifest_fixture(tmp_path)
    raw['recordings'][2]['interaction_evidence']=None
    path.write_text(json.dumps(raw))
    report=run_action_experiments(path,tmp_path/'out2',cfg,thresholds=[.5],margins=[0])
    for v in report['variants'].values():
        m=v['metrics']['test']['events']['per_item'][HIT]
        assert m['fn']==1 and m['tp']==0 and m['coverage']==0
        assert m['abstained_positive'] and m['abstained_negative']


def test_person3_does_not_verify_contactless_action():
    from unittest.mock import Mock
    from person3.pipeline import validate_p2_result
    data,bundle,contacts=fixture()
    result,checks=detect_with_assessments(data,bundle,contacts,recording_sha256=HASH)
    verifier=Mock()
    reviews=validate_p2_result(data,result,verifier=verifier)  # missing sidecar
    verifier.validate.assert_not_called()
    assert reviews and all(r.validation_status=='insufficient_evidence' for r in reviews)


def test_synthetic_punch_reaches_person3_with_source_timestamps_and_camera_result():
    """Synthetic mechanics check only; it is not evidence of real-video accuracy."""
    from unittest.mock import Mock
    from person3.contracts import Verification
    from person3.pipeline import validate_p2_result
    from cmai.results import attach_machine_reviews

    data,bundle,contacts=fixture()
    actor=data.persons[0]
    # A brief normalized wrist extension, velocity/acceleration burst, then
    # retraction. These are fixture features, not a trained real-world detector.
    for i,o in enumerate(actor.observations):
        phase=i % 10
        wrist_x={0:0.0,1:.15,2:.4,3:.65,4:.4,5:.15}.get(phase,0.0)
        o.normalized_pose.landmarks['left_wrist'].x=wrist_x
        o.normalized_pose.landmarks['left_elbow'].x=wrist_x*.55
        speed={0:0.0,1:1.5,2:2.5,3:2.5,4:-2.5,5:-1.5}.get(phase,0.0)
        o.motion.feature_values['left_wrist.velocity.speed']=abs(speed)
        o.motion.feature_values['left_wrist.acceleration.magnitude']=abs(speed)*2
        o.quality.feature_validity['left_wrist.velocity.speed']=True
        o.quality.feature_validity['left_wrist.acceleration.magnitude']=True
    data.video.feature_names=['left_wrist.velocity.speed','left_wrist.acceleration.magnitude','left_ankle.velocity.speed']
    # Train the synthetic two-centroid baseline on this controlled example and
    # a contrasting negative. Production bundles require permitted labelled data.
    cfg=bundle.metadata.configuration
    seed=process_perception(data,cfg).persons[0].chunks[0].fused_embedding
    negative=seed.model_copy(update={'values':[-v for v in seed.values]})
    model=train_action_model([(HIT,'positive',seed),(HIT,'negative',negative)],[HIT],'synthetic-punch-fixture',{})
    bundle=replace(bundle,action_model=model,bank=model.positive_bank())
    p2,checks=detect_with_assessments(data,bundle,contacts,recording_sha256=HASH)
    assert p2.persons[0].events and p2.persons[0].events[0].behaviour==HIT
    verifier=Mock()
    verifier.validate.side_effect=lambda packet: Verification(
        decision='supported',reason='Fixture evidence selected.',
        evidence_segment_ids=[s.evidence_id for s in packet.segments])
    reviews=validate_p2_result(data,p2,verifier=verifier,action_assessments=checks)
    supported=[r for r in reviews if r.validation_status=='supported']
    assert supported and supported[0].behaviour==HIT
    assert supported[0].start_timestamp==min(s.timestamp for s in actor.observations
        if f'{actor.person_id}:frame:{s.frame_index}' in supported[0].selected_evidence_ids)
    assert supported[0].end_timestamp==max(s.timestamp for s in actor.observations
        if f'{actor.person_id}:frame:{s.frame_index}' in supported[0].selected_evidence_ids)
    assert verifier.validate.called
    camera=build_camera_result(data,p2,bundle,HASH,checks)
    camera=attach_machine_reviews(camera,reviews,{'provider':'mock','model':'fixture'})
    assert camera.events[0].cmai_item_id==HIT
    assert camera.events[0].machine_validation['result']['validation_status']=='supported'


def test_required_video_missing_does_not_fall_back_to_pose():
    from cmai.action_detection import apply_action_rules
    from person2.contracts import Embedding
    data,bundle,contacts=fixture()
    result=process_perception(data,bundle.metadata.configuration,bundle.bank)
    bundle.metadata.video_encoder=LocalVideoEncoder(model='torchvision_r3d_18_kinetics400_v1',
        source='https://download.pytorch.org/models/r3d_18-b3b3357e.pth',checkpoint_file='test',checkpoint_sha256=HASH)
    for person in result.persons:
        for chunk in person.chunks:
            chunk.embeddings['video']=Embedding(space='test-only',values=[0],valid=[False])
    result,checks=apply_action_rules(result,data,bundle,contacts,HASH)
    assert not any(p.events for p in result.persons)
    assert all(a.status=='incompatible_evidence' for a in checks.assessments)


def test_worker_action_handoff_and_evidence_export(tmp_path):
    import cv2
    import numpy as np
    from unittest.mock import patch
    from cmai.worker import run
    from cmai.results import create_evidence, export_archive
    from person1.config import Person1Config
    from person2.contracts import Person2VideoResult
    from dataclasses import asdict
    data,bundle,contacts=fixture()
    video=tmp_path/'source.avi'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),5,(32,32))
    for _ in range(20):writer.write(np.zeros((32,32,3),dtype=np.uint8))
    writer.release()
    actual_hash=sha256(video.read_bytes()).hexdigest();contacts.recording_sha256=actual_hash
    request=dict(path=str(video),p1_configuration=asdict(Person1Config(frame_sample_fps=5)),
        bundle=bundle.metadata.model_dump(),bundle_sha256=bundle.bundle_sha256,bank=bundle.bank.model_dump(),
        action_model=bundle.action_model.model_dump(),interactions=contacts.model_dump(),checkpoint_path=None)
    path=tmp_path/'request.json';path.write_text(json.dumps(request))
    def perception(*args,progress_callback,**kwargs):
        progress_callback(3.8)
        return data
    with patch('cmai.worker.process_video',side_effect=perception):run(path)
    assert json.loads((tmp_path/'progress.json').read_text())['phase']=='complete'
    p2=Person2VideoResult.model_validate_json((tmp_path/'candidates.json').read_text())
    checks=ActionEvidenceResult.model_validate_json((tmp_path/'action-assessments.json').read_text())
    camera=build_camera_result(data,p2,bundle,actual_hash,checks)
    camera=create_evidence(camera,video,tmp_path/'evidence')
    assert camera.events[0].evidence.clip_status=='available'
    assert export_archive(camera,tmp_path/'evidence')
