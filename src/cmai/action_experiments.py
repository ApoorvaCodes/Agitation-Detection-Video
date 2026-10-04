"""Permitted event annotations, train-only supervised centroids, held-out metrics."""
import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from cmai.action_detection import evidence_gate
from cmai.action_model import train_action_model
from cmai.bundle import (ActionAsset, ActionQuality, DetectorBundle, DetectorRule, LoadedBundle,
                         LocalVideoEncoder, ROOT, load_bundle)
from cmai.detection import detect_with_assessments
from cmai.interactions import InteractionEvidence
from cmai.taxonomy import VERSION, require_action_items
from cmai.experiments import overlap
from person1.io import load_perception
from person2.contracts import Contract
from person2.embeddings import TemporalPoseEncoder, encoder_metadata
from person2.experiments import DatasetProvenance, runtime_metadata
from person2.pipeline import process_perception


class ActionAnnotation(Contract):
    annotation_id: str = Field(min_length=1)
    item_id: str
    outcome: Literal['positive','negative','ambiguous']
    start_timestamp: float = Field(ge=0)
    end_timestamp: float = Field(gt=0)
    target_id: str | None = None
    target_kind: Literal['person','self','object'] | None = None

    @model_validator(mode='after')
    def check(self):
        require_action_items([self.item_id])
        if self.end_timestamp <= self.start_timestamp:
            raise ValueError('action annotation must have positive duration')
        if self.outcome == 'positive' and (not self.target_id or self.target_kind is None):
            raise ValueError('positive action requires a reviewed target identity/kind')
        return self


class ActionRecording(Contract):
    perception: str = Field(min_length=1)
    person_id: str = Field(min_length=1)
    subject_id: str = Field(min_length=1)
    participant_subject_ids: list[str] = Field(min_length=1)
    track_subject_ids: dict[str,str]
    session_id: str = Field(min_length=1)
    split: Literal['train','validation','test']
    recording_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    video: str | None = None
    interaction_evidence: str | None = None
    annotations: list[ActionAnnotation] = Field(min_length=1)

    @model_validator(mode='after')
    def check(self):
        if self.subject_id not in self.participant_subject_ids or len(set(self.participant_subject_ids)) != len(self.participant_subject_ids):
            raise ValueError('participants must include actor and all other people, with unique global IDs')
        if self.track_subject_ids.get(self.person_id) != self.subject_id or set(self.track_subject_ids.values()) != set(self.participant_subject_ids):
            raise ValueError('track/global subject mapping must agree with actor and participant identities')
        if any(not k or not v for k,v in self.track_subject_ids.items()):
            raise ValueError('track and global subject IDs must be nonempty')
        for item in {a.item_id for a in self.annotations}:
            spans = sorted((a.start_timestamp,a.end_timestamp) for a in self.annotations if a.item_id == item)
            if any(b0 < a1 for (a0,a1),(b0,b1) in zip(spans,spans[1:])):
                raise ValueError('conflicting overlapping action annotations for one item')
        return self


class ActionManifest(Contract):
    schema_version: Literal['cmai-action-annotations-1.0'] = 'cmai-action-annotations-1.0'
    taxonomy_version: Literal['cmai-long-form-camera-v1'] = VERSION
    detector_mode: Literal['interaction_actions'] = 'interaction_actions'
    provenance: DatasetProvenance
    behaviours: list[str] = Field(min_length=1)
    recordings: list[ActionRecording] = Field(min_length=1)

    @model_validator(mode='after')
    def check(self):
        require_action_items(self.behaviours)
        if len(set(self.behaviours)) != len(self.behaviours):
            raise ValueError('duplicate action labels')
        if {r.split for r in self.recordings} != {'train','validation','test'}:
            raise ValueError('action manifest requires train, validation, test partitions')
        ids = [a.annotation_id for r in self.recordings for a in r.annotations]
        if len(set(ids)) != len(ids):
            raise ValueError('duplicate annotation IDs')
        if any(a.item_id not in self.behaviours for r in self.recordings for a in r.annotations):
            raise ValueError('annotation item not declared')
        return self


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def audit_dataset(manifest, base):
    sources, contacts, audit, seen, subjects, sessions = {}, {}, [], {}, {}, {}
    partitions = {kind:{} for kind in ('subject','session','recording','perception','video_id')}
    for r in manifest.recordings:
        path = (base/r.perception).resolve()
        if path not in sources:
            sources[path] = load_perception(path)
        source = sources[path]
        if set(r.track_subject_ids) != {p.person_id for p in source.persons}:
            raise ValueError('map every visible person track to a global subject for disjoint splits')
        for person_id,subject in r.track_subject_ids.items():
            identity=(r.recording_sha256,person_id)
            if identity in subjects and subjects[identity]!=subject:
                raise ValueError('conflicting subject identity for a track')
            subjects[identity]=subject
        if r.person_id not in {p.person_id for p in source.persons}:
            raise ValueError('annotation actor track does not exist')
        track = (r.recording_sha256,r.person_id)
        if track in seen:
            raise ValueError('duplicate recording/actor annotations; combine intervals in one recording entry')
        seen[track] = True
        if track in subjects and subjects[track] != r.subject_id:
            raise ValueError('conflicting subject identity for a track')
        subjects[track] = r.subject_id
        if r.recording_sha256 in sessions and sessions[r.recording_sha256] != r.session_id:
            raise ValueError('conflicting session identity for recording')
        sessions[r.recording_sha256] = r.session_id
        values = [('subject',s) for s in r.participant_subject_ids] + [('session',r.session_id),('recording',r.recording_sha256),
                  ('perception',digest(path)),('video_id',source.video.video_id)]
        for kind,value in values:
            if value in partitions[kind] and partitions[kind][value] != r.split:
                raise ValueError(f'{kind} leakage across action splits: {value}')
            partitions[kind][value] = r.split
        for a in r.annotations:
            if source.video.duration_seconds is not None and a.end_timestamp > source.video.duration_seconds:
                raise ValueError('action annotation exceeds recording duration')
            if a.outcome == 'positive' and a.target_kind == 'person' and a.target_id not in {p.person_id for p in source.persons}:
                raise ValueError('annotated target track does not exist')
        video = (base/r.video).resolve() if r.video else None
        if video and digest(video) != r.recording_sha256:
            raise ValueError('source recording checksum mismatch')
        contact = InteractionEvidence.model_validate_json((base/r.interaction_evidence).read_text()) if r.interaction_evidence else None
        if contact:
            contact.validate_source(source,r.recording_sha256)
        contacts[(path,r.person_id)] = contact
        audit.append({**r.model_dump(), 'video_id':source.video.video_id,'perception_sha256':digest(path),
                      'interaction_evidence_sha256':digest(base/r.interaction_evidence) if r.interaction_evidence else None,
                      'video_checksum_verified':bool(video)})
    return sources,contacts,audit


def contained(start,end,annotations):
    spans = sorted((a.start_timestamp,a.end_timestamp) for a in annotations)
    merged = []
    for lo,hi in spans:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(hi,merged[-1][1])
        else:
            merged.append([lo,hi])
    return any(lo <= start and hi >= end for lo,hi in merged)


def event_metrics(rows, labels, iou_threshold=.5):
    per_item, timing = {}, []
    for label in labels:
        tp=fp=fn=excluded=ambiguous=assessed=total=abstained_pos=abstained_neg=0
        reasons = {}
        positive_count = negative_count = ambiguous_count = 0
        item_timing = []
        for r,result,evidence in rows:
            track = next(p for p in result.persons if p.person_id == r.person_id)
            truth = [a for a in r.annotations if a.item_id == label]
            positive = [a for a in truth if a.outcome == 'positive']
            uncertain = [a for a in truth if a.outcome == 'ambiguous']
            observed = [a for a in truth if a.outcome != 'ambiguous']
            positive_count += len(positive)
            negative_count += sum(a.outcome == 'negative' for a in truth)
            ambiguous_count += len(uncertain)
            used = set()
            for chunk in track.chunks:
                relevant = [a for a in truth if overlap(chunk.start_timestamp,chunk.end_timestamp,a.start_timestamp,a.end_timestamp)>0]
                if any(a.outcome == 'ambiguous' for a in relevant):
                    ambiguous += 1
                    continue
                if not contained(chunk.start_timestamp,chunk.end_timestamp,observed):
                    continue
                total += 1
                score = next((s for s in chunk.scores if s.behaviour == label),None)
                if chunk.status == 'scored' and score and score.similarity is not None:
                    assessed += 1
                else:
                    abstained_pos += int(any(a.outcome == 'positive' for a in relevant))
                    abstained_neg += int(not any(a.outcome == 'positive' for a in relevant))
                    check = next((a for a in evidence.assessments if a.person_id == r.person_id and a.chunk_id == chunk.chunk_id and a.item_id == label),None)
                    reason = check.status if check else chunk.status
                    reasons[reason] = reasons.get(reason,0)+1
            for event in track.events:
                if event.behaviour != label:
                    continue
                if any(overlap(event.start_timestamp,event.end_timestamp,a.start_timestamp,a.end_timestamp)>0 for a in uncertain) or not contained(event.start_timestamp,event.end_timestamp,observed):
                    excluded += 1
                    continue
                target_ids = {(c.target_kind,c.target_id) for check in evidence.assessments
                              if check.person_id == r.person_id and check.item_id == label and check.chunk_id in event.chunk_ids
                              for c in check.contacts if c.contact == 'observed' and event.start_timestamp <= c.timestamp < event.end_timestamp}
                matches = []
                for index,a in enumerate(positive):
                    inter = overlap(event.start_timestamp,event.end_timestamp,a.start_timestamp,a.end_timestamp)
                    union = event.end_timestamp-event.start_timestamp+a.end_timestamp-a.start_timestamp-inter
                    if index not in used and (a.target_kind,a.target_id) in target_ids and inter/union >= iou_threshold:
                        matches.append((inter/union,index,a))
                if matches:
                    iou,index,a = max(matches,key=lambda m:(m[0],-m[1]))
                    used.add(index);tp+=1
                    item_timing.append(dict(item_id=label, annotation_id=a.annotation_id, person_id=r.person_id,
                                       iou=iou,onset_error_seconds=event.start_timestamp-a.start_timestamp,
                                       offset_error_seconds=event.end_timestamp-a.end_timestamp))
                else:
                    fp+=1
            fn += len(positive)-len(used)
        timing.extend(item_timing)
        per_item[label] = dict(positive_events=positive_count,negative_intervals=negative_count,ambiguous_intervals=ambiguous_count,
                              mean_absolute_timing_error_seconds=sum(max(abs(t["onset_error_seconds"]),abs(t["offset_error_seconds"])) for t in item_timing)/len(item_timing) if item_timing else None,
                              tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else None,
                              recall=tp/(tp+fn) if tp+fn else None,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                              coverage=assessed/total if total else None,assessed_windows=assessed,evaluable_windows=total,
                              abstained_positive=abstained_pos,abstained_negative=abstained_neg,
                              abstention_reasons=reasons,ambiguous_windows=ambiguous,excluded_predictions=excluded)
    tp,fp,fn = (sum(m[k] for m in per_item.values()) for k in ('tp','fp','fn'))
    return dict(per_item=per_item,micro_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,iou_threshold=iou_threshold,
                matched_timing=timing,
                mean_absolute_onset_error_seconds=sum(abs(t['onset_error_seconds']) for t in timing)/len(timing) if timing else None,
                mean_absolute_offset_error_seconds=sum(abs(t['offset_error_seconds']) for t in timing)/len(timing) if timing else None)


def run_action_experiments(manifest_path,output_dir,config=None,thresholds=(.6,.8,.9),margins=(0,.1,.2),video_spec=None,checkpoint_path=None):
    config = config or load_bundle(ROOT/'configs/cmai_action_detector_bundle.json').metadata.configuration
    if not thresholds or not margins or any(not -1<=t<=1 for t in thresholds) or any(not 0<=m<=2 for m in margins):
        raise ValueError('finite threshold/margin grids required')
    if video_spec and abs(config.window_seconds-16/15)>1e-6:
        raise ValueError('video comparison requires the same 16/15-second windows for all variants')
    if video_spec:
        if checkpoint_path is None:
            raise ValueError('an explicit local checkpoint path is required')
        video_spec = video_spec.model_copy(update={'checkpoint_file':str(Path(checkpoint_path).resolve())})
    variants = [(name,replace(config,pose_weight=p,motion_weight=m,video_weight=0),pose,None)
                for name,p,m,pose in [('pose_only',1,0,'stats'),('motion_only',0,1,'stats'),
                                     ('pose_plus_motion',1,1,'stats'),('temporal_pose_motion',1,1,'time-bins')]]
    if video_spec:
        variants.append(('temporal_pose_motion_r3d18',replace(config,pose_weight=1,motion_weight=1,video_weight=1),'time-bins',video_spec))
    quality = ActionQuality()
    report = dict(schema_version='cmai-action-experiment-1.0',taxonomy_version=VERSION,status='pending_labelled_data',runtime=runtime_metadata(),
                  threshold_grid=sorted(set(thresholds)),contrast_margin_grid=sorted(set(margins)),variants={},selection=None,
                  limitations=['No clinical validity, calibration, intent, injury or two-week frequency inference.',
                               'Contact evidence is independently reviewed/external; P1 does not detect contact or objects.',
                               'Two-centroid classifier is an engineering baseline, not an evaluated best representation.',
                               'Coverage counts overlapping annotated windows; no independent-window confidence interval is inferred.',
                               'Participant identities, permissions and contact provenance require dataset-owner audit.',
                               'Test is evaluated once per representation per command; freeze reports and avoid repeated test-guided development.'])
    if not video_spec:
        report['video_comparison'] = dict(status='pending_local_checkpoint',model='torchvision_r3d_18_kinetics400_v1',metrics=None)
    output = Path(output_dir)
    if manifest_path is None:
        for name,cfg,pose,video in variants:
            report['variants'][name]=dict(status='pending_labelled_data',configuration=asdict(cfg),metrics=None)
    else:
        path = Path(manifest_path)
        manifest = ActionManifest.model_validate_json(path.read_text())
        sources,contacts,audit = audit_dataset(manifest,path.parent)
        report.update(status='evaluated',dataset_provenance=manifest.provenance.model_dump(),manifest_sha256=digest(path),split_audit=audit,
                      behaviours=manifest.behaviours)
        output.mkdir(parents=True,exist_ok=True)
        for name,cfg,pose,video in variants:
            enc = TemporalPoseEncoder(cfg.window_seconds,4) if pose == 'time-bins' else None
            extracted,video_encoders = {},{}
            for r in manifest.recordings:
                source_path = (path.parent/r.perception).resolve()
                key = (source_path,r.person_id)
                vencoder = None
                if video:
                    if not r.video:
                        raise ValueError('video comparison requires original local video paths for all recordings')
                    from cmai.video_actions import LocalR3DEncoder
                    vencoder = LocalR3DEncoder(path.parent/r.video,checkpoint_path,video,contacts[key],sources[source_path])
                video_encoders[key]=vencoder
                extracted[key]=process_perception(sources[source_path],cfg,pose_encoder=enc,video_encoder=vencoder)
            examples,unusable = [],[]
            for r in manifest.recordings:
                if r.split != 'train':continue
                source_path=(path.parent/r.perception).resolve();key=(source_path,r.person_id)
                person=next(p for p in sources[source_path].persons if p.person_id == r.person_id)
                for chunk in next(p for p in extracted[key].persons if p.person_id == r.person_id).chunks:
                    for label in manifest.behaviours:
                        rows=[a for a in r.annotations if a.item_id == label and overlap(chunk.start_timestamp,chunk.end_timestamp,a.start_timestamp,a.end_timestamp)>0]
                        if any(a.outcome=='ambiguous' for a in rows):continue
                        annotation=next((a for a in rows if a.start_timestamp<=chunk.start_timestamp and a.end_timestamp>=chunk.end_timestamp),None)
                        if annotation is None:continue
                        rule=DetectorRule(item_id=label,similarity_threshold=cfg.similarity_threshold,min_event_seconds=.2,min_evidence_fraction=.8)
                        status,reason,contact,evidence=evidence_gate(sources[source_path],person,chunk,rule,quality,contacts[key])
                        correct_target=annotation.outcome!='positive' or any(e.target_id==annotation.target_id and e.target_kind==annotation.target_kind for e in evidence)
                        if status!='scored' or not correct_target or annotation.outcome=='positive' and not contact:
                            unusable.append(dict(annotation_id=annotation.annotation_id,chunk_id=chunk.chunk_id,reason=reason));continue
                        examples.append((label,annotation.outcome,chunk.fused_embedding))
            first_key=next(iter(extracted))
            metadata=encoder_metadata(extracted[first_key],enc,video_encoders[first_key])
            try:
                model=train_action_model(examples,manifest.behaviours,f'{digest(path)[:16]}:{name}',metadata)
            except ValueError as exc:
                report['variants'][name]=dict(status='pending_usable_training_examples',reason=str(exc),configuration=asdict(cfg),metrics=None,
                                              unusable_training=unusable);continue
            asset_path=output/f'{name}.action-model.json';asset_path.write_text(model.model_dump_json(indent=2)+'\n')
            asset=ActionAsset(file=str(asset_path.resolve()),sha256=digest(asset_path))
            def infer(split,threshold,margin):
                rules=[DetectorRule(item_id=b,similarity_threshold=threshold,contrast_margin=margin,
                                    min_event_seconds=.2,min_evidence_fraction=.8) for b in manifest.behaviours]
                bundle_config=DetectorBundle(schema_version='1.1',detector_mode='interaction_actions',declared_labels=manifest.behaviours,
                              detector_id=f'action-centroids:{name}',version=model.version,mode='research',configuration=replace(cfg,similarity_threshold=threshold),
                              pose_representation=pose,temporal_bins=4,action_asset=asset,rules=rules,action_quality=quality,video_encoder=video)
                bundle=LoadedBundle(bundle_config,model.positive_bank(),sha256(bundle_config.model_dump_json().encode()).hexdigest(),model,
                                    Path(checkpoint_path) if checkpoint_path else None)
                results=[]
                for r in manifest.recordings:
                    if r.split!=split:continue
                    source_path=(path.parent/r.perception).resolve();key=(source_path,r.person_id)
                    result,assessments=detect_with_assessments(sources[source_path],bundle,contacts[key],path.parent/r.video if r.video else None,r.recording_sha256,
                                                             extracted_result=extracted[key])
                    results.append((r,result,assessments))
                metrics=event_metrics(results,manifest.behaviours)
                return metrics,bundle_config,[dict(person_id=r.person_id,recording_sha256=r.recording_sha256,
                     events=[e.model_dump() for p in result.persons if p.person_id==r.person_id for e in p.events],assessments=checks.model_dump())
                     for r,result,checks in results]
            grid=[]
            for threshold in sorted(set(thresholds)):
                for margin in sorted(set(margins)):
                    metrics,bundle_cfg,predictions=infer('validation',threshold,margin)
                    grid.append((threshold,margin,metrics,bundle_cfg,predictions))
            eligible=[g for g in grid if g[2]['micro_f1'] is not None and any((m['coverage'] or 0)>0 for m in g[2]['per_item'].values())]
            if not eligible:
                report['variants'][name]=dict(status='pending_usable_validation_evidence',metrics=None,
                                              validation_grid=[dict(threshold=t,margin=m,metrics=v) for t,m,v,b,p in grid]);continue
            threshold,margin,val_metrics,bundle_cfg,val_predictions=max(eligible,key=lambda g:(g[2]['micro_f1'],g[0],g[1]))
            test_metrics,_,test_predictions=infer('test',threshold,margin)
            # Use report-relative asset references when packaging/releasing.
            portable=bundle_cfg.model_copy(deep=True);portable.action_asset.file=asset_path.name
            bundle_path=output/f'{name}.bundle.json';bundle_path.write_text(portable.model_dump_json(indent=2)+'\n')
            report['variants'][name]=dict(status='evaluated',configuration=asdict(bundle_cfg.configuration),encoder_metadata=metadata,
                       action_quality=quality.model_dump(),video_encoder=video.model_dump() if video else None,
                       detector_rules=[r.model_dump() for r in bundle_cfg.rules],action_model_file=asset_path.name,
                       action_model_sha256=digest(asset_path),bundle_file=bundle_path.name,threshold_selection_split='validation',
                       selected_threshold=threshold,selected_contrast_margin=margin,unusable_training=unusable,
                       validation_grid=[dict(threshold=t,margin=m,metrics=v) for t,m,v,b,p in grid],
                       metrics=dict(validation=dict(events=val_metrics),test=dict(events=test_metrics)),
                       predictions=dict(validation=val_predictions,test=test_predictions))
        if any(v['status']!='evaluated' for v in report['variants'].values()):report['status']='incomplete_evaluation'
        report['selection']='Comparison only. No automatic best-model or release claim.'
    output.mkdir(parents=True,exist_ok=True)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest');parser.add_argument('--output-dir',required=True)
    parser.add_argument('--config',help='Person2Config JSON, same window settings for all comparisons')
    parser.add_argument('--thresholds',type=float,nargs='+',default=[.6,.8,.9])
    parser.add_argument('--margins',type=float,nargs='+',default=[0,.1,.2])
    parser.add_argument('--video-encoder',help='LocalVideoEncoder JSON; includes explicit checkpoint path/hash/source')
    args=parser.parse_args()
    from person2.config import Person2Config
    cfg=Person2Config(**json.loads(Path(args.config).read_text())) if args.config else None
    spec=LocalVideoEncoder.model_validate_json(Path(args.video_encoder).read_text()) if args.video_encoder else None
    checkpoint=(Path(args.video_encoder).parent/spec.checkpoint_file).resolve() if spec else None
    report=run_action_experiments(args.manifest,args.output_dir,cfg,args.thresholds,args.margins,spec,checkpoint)
    print(f"action experiment status={report['status']}")


if __name__=='__main__':main()
