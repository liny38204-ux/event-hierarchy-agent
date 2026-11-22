#!/usr/bin/env python3
"""Replay causal small-event inputs through buffered organization and collapse."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from event_memory.phase_end_collapse import PhaseEndCollapseManager
from event_memory.buffered_phase import (
    BufferedPhaseStore, BufferedPhaseManager,
    DEFAULT_EVENT_TRIGGER, DEFAULT_SECONDS_TRIGGER,
)
from event_memory.direct_frame_memory import frame_order, verify_evidence
from event_memory.backend_qwen38 import Qwen38Backend
from event_memory.export import save, markdown

SCHEMA='buffered_phase_memory_v10'


def state_hash(state):
    s=deepcopy(state)
    for f in s['frames'].values():f.pop('path')
    return hashlib.sha256(json.dumps(s,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def ingest(store,step,frames,replay_ids):
    """Source IDs belong only to the offline adapter, never to model/runtime state."""
    s=store.snapshot();events=[];new=[];changed=[];counter=s['next_event_id']
    for raw in step['events']:
        key=raw['source_event_id']
        if raw['op']=='new':
            if key in replay_ids:raise ValueError('source event created twice')
            eid=f'E{counter}';counter+=1;replay_ids[key]=eid;new.append(eid)
        elif raw['op']=='continue':
            eid=replay_ids[key]
            if eid not in s['events']:raise ValueError('source attempted to continue a collapsed event')
        else:raise ValueError('unsupported source E action')
        fs=frame_order(frames,raw['frame_ids'])
        event={'id':eid,'caption':raw['caption'],'frame_ids':fs,'start':raw['start'],'end':raw['end']}
        events.append(event);changed.append(eid);s['events'][eid]=event
    phases=[];parents=[]
    for ph in s['phases'].values():
        if set(ph['event_ids'])&set(changed):
            ph['start']=min(s['events'][e]['start'] for e in ph['event_ids'])
            ph['end']=max(s['events'][e]['end'] for e in ph['event_ids'])
            phases.append(ph)
    for p in s['episodes'].values():
        if set(p['phase_ids'])&{ph['id'] for ph in phases}:
            p['start']=min(s['phases'][x]['start'] for x in p['phase_ids'])
            p['end']=max(s['phases'][x]['end'] for x in p['phase_ids'])
            parents.append(p)
    store.write(now=step['now'],frames=[frames[f] for f in step['frame_ids']],events=events,phases=phases,episodes=parents)
    return new,changed


def build_video(video,frame_root,output,backend):
    vid=video['video_id'];folder=output/vid;(folder/'frames').mkdir(parents=True)
    frames={}
    for fid,f in video['frames'].items():
        src=frame_root/vid/'frames'/(fid+'.jpg');dst=folder/'frames'/src.name
        if hashlib.sha256(src.read_bytes()).hexdigest()!=f['sha256']:raise ValueError('source frame hash mismatch')
        shutil.copy2(src,dst);frames[fid]={**f,'path':str(dst.resolve())}
    verify_evidence(list(frames.values()))
    store=BufferedPhaseStore();manager=BufferedPhaseManager(backend);collapser=PhaseEndCollapseManager(backend)
    replay_ids={};trace=[];first=len(backend.calls)
    def checkpoint(status):
        s=store.snapshot()
        bundle={'schema':SCHEMA,'schema_version':10,'video_id':vid,'observed_until':s['observed_until'],
                'store':s,'status':status,'manager_state':deepcopy(manager.state),
                'provenance':{'kind':'buffered_causal_frozen_event_replay_new_model_organization',
                              'note_zh':'小事件按原前沿重放；缓存后重新组织PH/P，阶段结束后整体合并。没有使用旧PH/P或问答。'}}
        save(folder/'memory.json',bundle);(folder/'memory_tree.md').write_text(markdown(bundle))
        return bundle
    def record(step,review,collapsed):
        s=store.snapshot()
        entry={'segment_id':step['segment_id'],'now':step['now'],'review':review,'collapsed':collapsed,
               'state_sha256':state_hash(s),'events':len(s['events']),'phases':len(s['phases']),
               'episodes':len(s['episodes']),'pending_event_ids':s['roots'],'frames':len(s['frames'])}
        trace.append(entry)
        with (folder/'trace.jsonl').open('a') as f:f.write(json.dumps(entry,ensure_ascii=False)+'\n')
        print(json.dumps({'video':vid,**{k:entry[k] for k in ['segment_id','now','events','phases','episodes','pending_event_ids','frames']}},ensure_ascii=False),flush=True)
        checkpoint('running')
    try:
        for step in video['steps']:
            new,changed=ingest(store,step,frames,replay_ids)
            review=manager.maybe_review(store,step['now'],new_event_ids=new,changed_event_ids=changed)
            collapsed=collapser.collapse_closed(store,step['now'])
            record(step,review,collapsed)
        now=video['steps'][-1]['now']
        # EOF may drain work which the bounded decision view has already
        # accepted, but may not coerce a legitimate WAIT root into a phase.
        # Continue only when a review actually consumes roots; record why any
        # remaining tail is intentionally still unassigned.
        finalization=[]
        while True:
            before=list(store.snapshot()['roots'])
            review=manager.maybe_review(store,now,force=True)
            collapsed=collapser.collapse_closed(store,now)
            after=list(store.snapshot()['roots'])
            assigned=list(review.get('assigned_event_ids', [])) if review.get('reviewed') else []
            progress=len(after) < len(before)
            if not after:
                stop_reason='no_remaining_roots'
            elif not progress:
                stop_reason='model_deferred_remaining_roots'
            else:
                stop_reason='draining_visible_prefix'
            finalization.append({'review_index':len(finalization)+1,'roots_before':before,
                                 'assigned_event_ids':assigned,'remaining_root_ids':after,
                                 'progress':progress,'stop_reason':stop_reason,
                                 'organizer_reason':(review.get('decision') or {}).get('reason')})
            record({'segment_id':f'FINAL_REVIEW_{len(finalization):04d}','now':now},review,collapsed)
            if not after or not progress:
                break
        s=store.snapshot()
        if set(s['frames'])!=set(frames):raise ValueError('input coverage incomplete')
        if {f for e in s['events'].values() for f in e['frame_ids']}!=set(frames):raise ValueError('event evidence lost')
        if store.reviewable_phases():raise ValueError('closed phases remain unprocessed')
        checkpoint('complete')
        stats={'events':len(s['events']),'phases':len(s['phases']),'episodes':len(s['episodes']),
               'pending_events':len(s['roots']),'frames':len(s['frames']),'steps':len(video['steps']),
               'closed_phases':len(s['completed_phase_ids']),'actual_model_calls':len(backend.calls)-first,
               'open_phases':sum(p['status']=='open' for p in s['phases'].values()),
               'finalization':finalization}
        save(folder/'verification.json',stats)
        return stats
    except Exception:
        checkpoint('failed');raise


def build(input_path,frame_root,output,backend):
    data=json.loads(input_path.read_text())
    result={'schema':SCHEMA,'schema_version':10,'status':'running','videos':[]}
    save(output/'index.json',result);save(output/'input_manifest.json',data)
    save(output/'config.json',{'schema':SCHEMA,'schema_version':10,
         'review_new_events':DEFAULT_EVENT_TRIGGER,'review_video_seconds':DEFAULT_SECONDS_TRIGGER,
         'model_receipt':getattr(backend,'receipt',None),'input_sha256':hashlib.sha256(input_path.read_bytes()).hexdigest(),
         'training':False,'qa':False,'fixed_capacity':False})
    try:
        for v in data['videos']:
            stats=build_video(v,frame_root,output,backend)
            result['videos'].append({'video_id':v['video_id'],'statistics':stats});save(output/'index.json',result)
        result['status']='complete';save(output/'index.json',result)
        return result
    except Exception as e:
        result.update(status='failed',error=repr(e));save(output/'index.json',result);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--frames-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--model',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('output exists')
    backend=Qwen38Backend(a.model,a.output/'calls.jsonl')
    build(a.input,a.frames_run,a.output,backend)
