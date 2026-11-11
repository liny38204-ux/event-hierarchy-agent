"""Current captions + direct visual evidence, without source-event storage.

Closed phases are consolidated once. Consumed events are physically removed
from runtime state; their complete, time-ordered frame union belongs to M.
"""
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import re
from threading import RLock

from .phase_lifecycle import reconcile_lifecycle

SCHEMA = 'direct_frame_hierarchy_v7'
POLICY = 'on_phase_close_once_v1'
STATE_KEYS = {'frames', 'events', 'phases', 'episodes', 'roots',
              'completed_phase_ids', 'observed_until', 'next_merge_id', 'next_event_id'}
EVENT_KEYS = {'id', 'caption', 'frame_ids', 'start', 'end'}
NODE_KEYS = {'id', 'title', 'summary', 'status', 'start', 'end',
             'closed_at', 'close_reason', 'close_trigger_id'}


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('time must be finite')
    return float(value)


def ids(value):
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value) or len(set(value)) != len(value):
        raise ValueError('IDs must be a unique string list')
    return value


def text(value, limit=None):
    if not isinstance(value, str) or not value.strip() or (limit and len(value) > limit):
        raise ValueError('invalid caption or summary')
    return value


def frame_order(frames, values):
    return sorted(set(values), key=lambda fid: (frames[fid]['time'], fid))


class DirectFrameMemoryStore:
    """Serializable active state; no hidden source/observation/archive bank."""
    def __init__(self, state=None):
        self._lock = RLock()
        self._state = deepcopy(state) if state is not None else {
            'frames': {}, 'events': {}, 'phases': {}, 'episodes': {}, 'roots': [],
            'completed_phase_ids': [], 'observed_until': 0.,
            'next_merge_id': 1, 'next_event_id': 1}
        self.validate(self._state)

    @staticmethod
    def validate(s):
        if set(s) != STATE_KEYS:
            raise ValueError('active state has unknown or missing fields')
        now = finite(s['observed_until'])
        for counter in ('next_merge_id', 'next_event_id'):
            if type(s[counter]) is not int or s[counter] < 1:
                raise ValueError('invalid event counter')
        for fid, f in s['frames'].items():
            if set(f) != {'id', 'time', 'path', 'sha256'} or f['id'] != fid:
                raise ValueError('invalid direct frame reference')
            if not 0 <= finite(f['time']) <= now:
                raise ValueError('future or negative frame')
            text(f['path'])
            if not re.fullmatch('[0-9a-f]{64}', f['sha256']):
                raise ValueError('frame requires SHA256')
        for eid, e in s['events'].items():
            if set(e) != EVENT_KEYS or e['id'] != eid or not re.fullmatch('[EM][1-9][0-9]*', eid):
                raise ValueError('invalid active event')
            text(e['caption'])
            fs = ids(e['frame_ids'])
            if not fs or any(f not in s['frames'] for f in fs):
                raise ValueError('event needs known frames')
            if fs != frame_order(s['frames'], fs):
                raise ValueError('frames must be in time order')
            times = [s['frames'][f]['time'] for f in fs]
            if (finite(e['start']), finite(e['end'])) != (min(times), max(times)):
                raise ValueError('event bounds must be frame envelope')
            counter = 'next_merge_id' if eid.startswith('M') else 'next_event_id'
            if int(eid[1:]) >= s[counter]:
                raise ValueError('event counter must exceed active IDs')
        assigned = []
        for name, child_key, children in [('phases', 'event_ids', s['events']),
                                          ('episodes', 'phase_ids', s['phases'])]:
            members = []
            for nid, n in s[name].items():
                required = {'id', 'title', 'summary', 'status', 'start', 'end', child_key}
                if not required <= set(n) or set(n) - (NODE_KEYS | {child_key}) or n['id'] != nid:
                    raise ValueError('invalid hierarchy node')
                text(n['title'], 32); text(n['summary'], 120)
                ci = ids(n[child_key])
                if not ci or any(c not in children for c in ci):
                    raise ValueError('unknown or missing child')
                if ci != sorted(ci, key=lambda c: children[c]['start']):
                    raise ValueError('children must be chronological')
                if (finite(n['start']), finite(n['end'])) != (min(children[c]['start'] for c in ci), max(children[c]['end'] for c in ci)):
                    raise ValueError('parent bounds must be child envelope')
                if n['end'] > now or n['status'] not in ('open', 'closed'):
                    raise ValueError('invalid node time/status')
                if n['status'] == 'closed':
                    if not n['end'] <= finite(n.get('closed_at')) <= now:
                        raise ValueError('invalid close confirmation time')
                elif any(k in n for k in ('closed_at', 'close_reason', 'close_trigger_id')):
                    raise ValueError('open node has closure metadata')
                members.extend(ci)
            if len(members) != len(set(members)):
                raise ValueError('child has multiple parents')
            if name == 'episodes' and set(members) != set(children):
                raise ValueError('every phase needs one parent')
            if name == 'phases': assigned = members
        if set(ids(s['roots'])) != set(s['events']) - set(assigned):
            raise ValueError('roots do not match unassigned events')
        complete = set(ids(s['completed_phase_ids']))
        if any(p not in s['phases'] or s['phases'][p]['status'] != 'closed' for p in complete):
            raise ValueError('only closed phases may be complete')
        ps = sorted(s['episodes'].values(), key=lambda p: p['start'])
        if any(p['status'] != 'closed' for p in ps[:-1]):
            raise ValueError('previous episode left open')
        for p in ps:
            for i, phid in enumerate(p['phase_ids']):
                ph = s['phases'][phid]
                if (p['status'] == 'closed' or i < len(p['phase_ids']) - 1) and ph['status'] != 'closed':
                    raise ValueError('previous/child phase left open')
                if any(e.startswith('M') for e in ph['event_ids']) and phid not in complete:
                    raise ValueError('merged event in unfinished phase')

    def snapshot(self):
        with self._lock: return deepcopy(self._state)

    def write(self, *, now, frames=(), events=(), phases=(), episodes=()):
        """Atomic patches from writer/organizer, storing only current captions.

        New E IDs are monotonic; a continued E supplies its complete current
        caption and frame IDs. Closed events cannot be continued or reopened.
        """
        now = finite(now)
        with self._lock:
            old = self._state; s = deepcopy(old)
            if now < old['observed_until']: raise ValueError('time moved backwards')
            s['observed_until'] = now
            for f in frames:
                if f['id'] in s['frames'] and s['frames'][f['id']] != f:
                    raise ValueError('immutable frame changed')
                s['frames'][f['id']] = deepcopy(f)
            frozen = {e for p in old['phases'].values() if p['status'] == 'closed' for e in p['event_ids']}
            for e in events:
                eid = e['id']
                if not re.fullmatch('E[1-9][0-9]*', eid): raise ValueError('writer only writes E IDs')
                if eid in frozen and old['events'][eid] != e: raise ValueError('closed event changed')
                if eid not in old['events'] and int(eid[1:]) < old['next_event_id']:
                    raise ValueError('retired or reused E ID')
                if eid in old['events'] and not set(old['events'][eid]['frame_ids']) <= set(e['frame_ids']):
                    raise ValueError('continued event lost frames')
                s['events'][eid] = deepcopy(e)
                s['next_event_id'] = max(s['next_event_id'], int(eid[1:]) + 1)
            for name, nodes, key in [('phases', phases, 'event_ids'), ('episodes', episodes, 'phase_ids')]:
                for n in nodes:
                    previous = old[name].get(n['id'])
                    if previous and n[key][:len(previous[key])] != previous[key]:
                        raise ValueError('organizer cannot rewrite existing membership')
                    s[name][n['id']] = deepcopy(n)
            s['phases'], s['episodes'], transitions = reconcile_lifecycle(
                s['phases'], s['episodes'], now, old['phases'], old['episodes'])
            assigned = {e for p in s['phases'].values() for e in p['event_ids']}
            s['roots'] = sorted(set(s['events']) - assigned, key=lambda e: s['events'][e]['start'])
            for pid, p in s['phases'].items():
                if p['status'] == 'closed' and len(p['event_ids']) == 1 and pid not in s['completed_phase_ids']:
                    s['completed_phase_ids'].append(pid)
            self.validate(s)
            self._state = s
            return transitions

    def reviewable_phases(self):
        s = self.snapshot()
        return [p for p, n in s['phases'].items() if n['status'] == 'closed'
                and p not in s['completed_phase_ids'] and len(n['event_ids']) > 1]

    def apply_consolidation(self, decision, now, phase_ids):
        """Delete consumed E rows, replace by M with the full direct-frame union."""
        now = finite(now); ids(phase_ids)
        with self._lock:
            s = deepcopy(self._state)
            if now < s['observed_until']: raise ValueError('time moved backwards')
            if not phase_ids or not set(phase_ids) <= set(self.reviewable_phases()):
                raise ValueError('only unfinished closed phases may be consolidated')
            keys = {'merges', 'updates', 'kept_separate', 'phases', 'episodes'}
            if not isinstance(decision, dict) or set(decision) != keys or any(not isinstance(decision[k], list) for k in keys) or decision['updates']:
                raise ValueError('invalid close-only decision')
            pairs = {(p, a, b) for p in phase_ids for a, b in zip(s['phases'][p]['event_ids'], s['phases'][p]['event_ids'][1:])}
            covered = set(); consumed = set(); affected = set(); new_ids = []
            for m in decision['merges']:
                if set(m) != {'phase_id', 'event_ids', 'summary', 'reason', 'support_evidence_ids'}:
                    raise ValueError('invalid merge fields')
                p = m['phase_id']; es = ids(m['event_ids'])
                if p not in phase_ids or len(es) < 2 or set(es) & consumed:
                    raise ValueError('invalid or reused merge member')
                sequence = s['phases'][p]['event_ids']
                if es[0] not in sequence: raise ValueError('unknown active event')
                at = sequence.index(es[0])
                if sequence[at:at+len(es)] != es: raise ValueError('merge must be same-phase adjacent')
                fs = frame_order(s['frames'], [f for e in es for f in s['events'][e]['frame_ids']])
                support = ids(m['support_evidence_ids'])
                if not set(support) <= set(fs) or any(not set(support) & set(s['events'][e]['frame_ids']) for e in es):
                    raise ValueError('support must cover each merge member')
                text(m['reason']); text(m['summary'], 600)
                covered.update((p, a, b) for a, b in zip(es, es[1:]))
                mid = 'M' + str(s['next_merge_id']); s['next_merge_id'] += 1
                s['events'][mid] = {'id': mid, 'caption': m['summary'], 'frame_ids': fs,
                    'start': min(s['events'][e]['start'] for e in es), 'end': max(s['events'][e]['end'] for e in es)}
                for e in es: del s['events'][e]
                s['phases'][p]['event_ids'] = sequence[:at] + [mid] + sequence[at+len(es):]
                consumed.update(es); affected.add(p); new_ids.append(mid)
            for k in decision['kept_separate']:
                if set(k) != {'phase_id', 'event_ids', 'reason'} or len(ids(k['event_ids'])) != 2:
                    raise ValueError('invalid kept pair')
                pair = (k['phase_id'], *k['event_ids'])
                if pair not in pairs or pair in covered: raise ValueError('unknown or duplicate pair')
                text(k['reason']); covered.add(pair)
            if covered != pairs: raise ValueError('every adjacent pair needs a decision')
            parents = {p for p, n in s['episodes'].items() if set(n['phase_ids']) & affected}
            for name, required in [('phases', affected), ('episodes', parents)]:
                updates = decision[name]
                if len(updates) != len(required) or {n['id'] for n in updates} != required:
                    raise ValueError('all and only affected parents need summary refresh')
                for n in updates:
                    if set(n) != {'id', 'summary'}: raise ValueError('invalid parent refresh')
                    s[name][n['id']]['summary'] = text(n['summary'], 120)
            s['completed_phase_ids'].extend(phase_ids); s['observed_until'] = now
            self.validate(s)
            self._state = s
            return {'new_event_ids': new_ids, 'completed_phase_ids': list(phase_ids)}


def direct_review_input(store, phase_ids, now, image_limit=6):
    """Read only active captions and their directly referenced frames."""
    s = store.snapshot(); now = finite(now)
    if now < s['observed_until'] or not set(ids(phase_ids)) <= set(store.reviewable_phases()):
        raise ValueError('invalid review time/phases')
    event_ids = [e for p in phase_ids for e in s['phases'][p]['event_ids']]
    chosen = list(dict.fromkeys(s['events'][e]['frame_ids'][0] for e in event_ids))
    if type(image_limit) is not int or image_limit < len(chosen):
        raise ValueError('image budget must cover every candidate event')
    chosen += [s['events'][e]['frame_ids'][-1] for e in reversed(event_ids)]
    chosen = list(dict.fromkeys(chosen))[:image_limit]
    evidence = [deepcopy(s['frames'][f]) for f in frame_order(s['frames'], chosen)]
    payload = {'observed_until': now, 'policy': POLICY, 'review_phase_ids': phase_ids,
        'events': {e: deepcopy(s['events'][e]) for e in event_ids},
        'phases': {p: deepcopy(s['phases'][p]) for p in phase_ids},
        'episodes': {p: deepcopy(n) for p, n in s['episodes'].items() if set(n['phase_ids']) & set(phase_ids)},
        'candidate_pairs': [{'phase_id': p, 'event_ids': [a, b]} for p in phase_ids
                            for a, b in zip(s['phases'][p]['event_ids'], s['phases'][p]['event_ids'][1:])],
        'frame_times': {f: s['frames'][f]['time'] for e in event_ids for f in s['events'][e]['frame_ids']},
        'visible_support_frame_ids': [f['id'] for f in evidence]}
    return payload, evidence


def verify_evidence(evidence):
    for f in evidence:
        if hashlib.sha256(Path(f['path']).read_bytes()).hexdigest() != f['sha256']:
            raise ValueError('frame content changed: ' + f['id'])
