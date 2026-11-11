"""Collect small events while a phase is open; replace them once it closes.

The replacement owns the complete frame union. Retired event records, aliases
and source captions are absent from serialized and in-memory runtime state.
"""
import json

from .direct_frame_memory import DirectFrameMemoryStore, finite, frame_order, ids, text, verify_evidence
from .protocol import parse_manager_json

SCHEMA = 'phase_end_collapse_memory_v9'
POLICY = 'all_events_once_after_phase_close'


class PhaseEndCollapseStore(DirectFrameMemoryStore):
    @staticmethod
    def validate(state):
        DirectFrameMemoryStore.validate(state)
        if any(not eid.startswith('E') for eid in state['events']) or state['next_merge_id'] != 1:
            raise ValueError('v9 uses E IDs only')
        if any(len(state['phases'][pid]['event_ids']) != 1 for pid in state['completed_phase_ids']):
            raise ValueError('a consolidated phase must have exactly one event')

    def write(self, **kwargs):
        # Inherited writes allow separate E records and sequential closure.
        # Undo v7's automatic completion of singleton phases: completion always
        # means that the phase-end transaction has actually succeeded.
        with self._lock:
            old = self.snapshot()
            try:
                transitions = super().write(**kwargs)
                self._state['completed_phase_ids'] = list(old['completed_phase_ids'])
                self.validate(self._state)
                return transitions
            except Exception:
                self._state = old
                raise

    def reviewable_phases(self):
        s = self.snapshot()
        return [pid for pid, phase in s['phases'].items()
                if phase['status'] == 'closed' and pid not in s['completed_phase_ids']]

    def apply_consolidation(self, *args, **kwargs):
        raise ValueError('v9 requires whole-phase collapse, not pairwise merge/keep')

    def collapse_phase(self, decision, now):
        now = finite(now)
        with self._lock:
            s = self.snapshot()
            required = {'phase_id', 'event_ids', 'caption', 'phase_summary', 'support_frame_ids'}
            if not isinstance(decision, dict) or set(decision) != required:
                raise ValueError('invalid whole-phase collapse fields')
            pid = decision['phase_id']
            if now < s['observed_until'] or pid not in self.reviewable_phases():
                raise ValueError('only an unfinished closed phase may be collapsed')
            members = ids(decision['event_ids'])
            if members != s['phases'][pid]['event_ids']:
                raise ValueError('must replace every event in this phase in time order')
            fs = frame_order(s['frames'], [f for eid in members for f in s['events'][eid]['frame_ids']])
            support = ids(decision['support_frame_ids'])
            if not support or not set(support) <= set(fs):
                raise ValueError('support must reference this phase frames')
            caption = text(decision['caption'], 3000)
            summary = text(decision['phase_summary'], 80)
            eid = 'E' + str(s['next_event_id'])
            s['next_event_id'] += 1
            replacement = {'id': eid, 'caption': caption, 'frame_ids': fs,
                           'start': s['frames'][fs[0]]['time'], 'end': s['frames'][fs[-1]]['time']}
            for old in members:
                del s['events'][old]
            s['events'][eid] = replacement
            s['phases'][pid]['event_ids'] = [eid]
            s['phases'][pid]['summary'] = summary
            s['completed_phase_ids'].append(pid)
            s['observed_until'] = now
            self.validate(s)
            self._state = s
            return eid


COLLAPSER = '''你是流式视频的阶段结束整理器。阶段进行中已经积累多个独立小事件；现在该阶段已经结束，将这个阶段下全部小事件合并为一个完整事件。
输入events按观察时间排序。新caption要按顺序描述这一阶段，保留有证据的不同动作、重复次数、物体变化、文字和不确定性；多个步骤仍是多个步骤，不要编成一次原子动作。不要推测未见行为。
phase_summary只写一句简短概括，最多80字符；caption最多3000字符。所有event_ids必须完整按输入顺序列出，不得只合并部分、不输出keep、不改变阶段边界。
程序会删除原小事件记录，新事件直接持有全部帧的有序并集，不保存source、旧caption、别名或历史。support_frame_ids只是本次依据，不是筛帧结果；只可引用visible_frame_ids中实际展示的图片。
只返回JSON：{"phase_id":"PH1","event_ids":["E1","E2"],"phase_summary":"简短阶段描述。","caption":"合并后的完整描述。","support_frame_ids":["f000001"]}。'''


class PhaseEndCollapseManager:
    def __init__(self, backend, image_limit=8):
        if type(image_limit) is not int or image_limit < 2:
            raise ValueError('at least two image slots required')
        self.backend = backend
        self.image_limit = image_limit

    def collapse_closed(self, store, now):
        now = finite(now)
        if now < store.snapshot()['observed_until']:
            raise ValueError('time moved backwards')
        results = []
        for pid in store.reviewable_phases():
            s = store.snapshot()
            phase = s['phases'][pid]
            members = [s['events'][eid] for eid in phase['event_ids']]
            fs = frame_order(s['frames'], [f for e in members for f in e['frame_ids']])
            chosen = fs if len(fs) <= self.image_limit else [
                fs[round(i * (len(fs)-1) / (self.image_limit-1))] for i in range(self.image_limit)]
            evidence = [s['frames'][fid] for fid in chosen]
            verify_evidence(evidence)
            payload = {'observed_until': now, 'policy': POLICY, 'phase': phase, 'events': members,
                       'frame_times': {f: s['frames'][f]['time'] for f in fs}, 'visible_frame_ids': chosen}
            base = json.dumps(payload, ensure_ascii=False)
            repair = ''; rejected = []
            for attempt in range(3):
                raw = self.backend.generate(COLLAPSER, base+repair, evidence,
                    stage=f'collapse:{pid}:{attempt}', max_tokens=2048)
                try:
                    if getattr(self.backend, 'calls', []) and self.backend.calls[-1].get('finish_reason') == 'length':
                        raise ValueError('truncated consolidated caption')
                    decision, _ = parse_manager_json(raw)
                    if decision.get('phase_id') != pid:
                        raise ValueError('wrong phase')
                    if not set(ids(decision.get('support_frame_ids'))) <= set(chosen):
                        raise ValueError('support must use actually displayed frames')
                    eid = store.collapse_phase(decision, now)
                    results.append({'phase_id': pid, 'event_id': eid, 'accepted_attempt': attempt,
                                    'rejected_attempts': rejected, 'visible_frame_ids': chosen})
                    break
                except ValueError as exc:
                    rejected.append(str(exc))
                    repair = '\n修复校验错误：' + str(exc) + '\n上次回复：' + raw
            else:
                raise ValueError(f'phase {pid} collapse failed: {rejected}')
        return results
