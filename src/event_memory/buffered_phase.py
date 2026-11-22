"""Delayed P/PH assignment over the direct-frame phase-end memory.

The buffer is the ordered ``roots`` list already present in runtime memory.
Only small event IDs and review scheduling live in the manager checkpoint; no
captions, frames, old events, aliases or archives are duplicated there.
"""
from copy import deepcopy
import json
import re

from .direct_frame_memory import finite, frame_order, ids, text, verify_evidence
from .phase_end_collapse import PhaseEndCollapseStore
from .protocol import parse_manager_json

SCHEMA = 'buffered_phase_memory_v10'
POLICY = 'buffered_prefix_assignment_v1'
DEFAULT_EVENT_TRIGGER = 5
DEFAULT_SECONDS_TRIGGER = 30.
DEFAULT_REVIEW_EVENT_LIMIT = 8
DEFAULT_ACTIVE_EVENT_LIMIT = 8
ORGANIZER_CONTEXT_LIMIT = 10000
REPAIR_RAW_LIMIT = 4096


class BufferedPhaseStore(PhaseEndCollapseStore):
    """Schema10 runtime state has exactly the schema9 nine store fields."""


ORGANIZER = '''你管理流式视频记忆。小事件E已保存原始caption和直接帧；未归入阶段的E按时间组成待组织缓存。你需要区分大事件P与内部阶段PH的粒度，既不把所有内容归成宽泛的一段，也不因一个新画面就新建阶段。

一个E的caption可以包含多行观察，每行是该事件内部的一句话，不是独立小事件或阶段。请结合整条caption判断活动与边界，不能因为换行就拆阶段；P/PH继续只写简短概括，不复述逐行细节。

这次只根据目前可见证据，决定缓存最早的连续前缀是否已经可以归入当前阶段或一个新阶段；无法确认用途、边界或段落角色时保留在缓存。画面风格变化、标题、图形、人物或镜头切换只是证据，不是绝对规则；不要猜测未来内容或使用问题、答案。

层级尺度：P概括一个较宽的、持续的已观察活动或过程，可以包含不同的局部步骤与观察焦点。PH概括其中时间连续、局部焦点或具体步骤一致的一段。PH标题和短摘要应说清这一段的主要对象/子区域及观察或操作内容，不能只复述P的整体主题。静态观察也可以构成阶段，不必编造动作或意图。不得为了凑层级强行制造多个PH，也不预设阶段数量或长度。

阶段边界：当已观察证据表明主要关注对象/子区域、具体步骤或交互活动发生实质转换，先前PH的具体描述已不能准确概括后续内容时，应在同一P内建立后续PH；只有更宽的活动/过程也改变时才考虑新P。同一地点、同一人物、时间相邻或共享宽泛主题，都不足以证明仍是同一PH。不要通过扩大旧PH的含义来吸收已经不同的局部活动。

连续性保护：同一具体活动中的视角变化、短暂遮挡、人物进出、顺带扫到物体、动作的连续细节或短暂移动，不单独构成阶段边界。要结合相邻E和可见帧判断局部活动是否真的转换。若只有孤立或含糊线索，保留尚不能判定的后缀等待；已有充分证据的边界不必等到整个活动结束才确认。短暂过渡可以随连续活动组织，有独立观察内容且边界明确的过渡才单列。

缓存判读：按时间检查缓存内部是否存在多个局部活动，不要把整批缓存默认当成一个组。一次可以先把连续前缀追加当前PH，再把后续已确认的连续组分别归入新PH，最后保留不确定后缀。后面的内容又接近早先主题，不能据此抹去中间已观察到的独立活动或边界；已结束阶段也不能重新追加。reason简短说明延续的是哪项具体活动，或边界前后发生了什么变化，避免只用“主题一致”解释所有归组。

输出groups按时间顺序消费缓存前缀。每组的event_ids必须是连续E；剩余event必须全部出现在deferred_event_ids，且只能是缓存后缀。对已有open phase只能追加成员；新phase表示确认进入新的已观察阶段，程序会据此关闭前一个phase。P/PH摘要只概括已观察内容。EOF只是最后一次复核，不强制归组或结束阶段。

若输入含deferred_tail_event_ids，它们是仍保存在缓存、但本次未展开caption的更晚后缀；绝不能写入groups或deferred_event_ids，程序会自动保留它们供下一次审查。只处理pending_event_ids及其pending_events。

只输出JSON：
{"groups":[{"event_ids":["E1","E2"],"phase_id":"PH1","phase_title":"简短阶段名","phase_summary":"不超过80字的已观察阶段概括","episode_id":"P1","episode_title":"简短大事件名","episode_summary":"不超过120字的已观察概括"}],"deferred_event_ids":["E3"],"refreshes":[{"phase_id":"PH1","phase_title":"简短阶段名","phase_summary":"不超过80字","episode_id":"P1","episode_title":"简短大事件名","episode_summary":"不超过120字"}],"reason":"为何已归组或继续等待"}

输入中的next_phase_id/next_episode_id是新节点必须使用的下一个ID；同次新建多个节点时依次递增。已有节点只能使用已经存在且仍open的ID。
dirty_event_ids中若有已属于当前阶段的事件被续写，需要在refreshes更新该阶段及大事件短摘要；若groups已经更新同一阶段可不重复。没有摘要更新时refreshes为[]。需要继续等待时groups为[]、所有pending_event_ids原样放入deferred_event_ids；等待本身不会关闭阶段。
title最多32字；不得删除、移动、重写已有阶段/大事件成员；不要输出caption、帧、source、merge或keep。'''


def _ordered_roots(state):
    return sorted(state['roots'], key=lambda eid: (state['events'][eid]['start'], eid))


def _bounded_organizer_text(value, limit, path):
    """Validate a bounded manager field without hiding repairable length errors."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{path} must be a nonempty string')
    actual = len(value)
    if actual > limit:
        raise ValueError(f'{path} length {actual} exceeds maximum {limit}; shorten it')
    return value


def _token_count(backend, value):
    """Production uses the frozen backend tokenizer; fixtures need no model."""
    if hasattr(backend, 'token_count'):
        return backend.token_count(value)
    return len(value.split())


def _node(identifier, title, summary, child_key, children, state):
    title = text(title, 32); summary = text(summary, 120); children = ids(children)
    records = state['events'] if child_key == 'event_ids' else state['phases']
    return {'id': identifier, 'title': title, 'summary': summary,
            child_key: children, 'status': 'open',
            'start': min(records[x]['start'] for x in children),
            'end': max(records[x]['end'] for x in children)}


class BufferedPhaseManager:
    """Review roots after a small delay; the caller persists ``state`` beside logs."""
    def __init__(self, backend, *, event_trigger=DEFAULT_EVENT_TRIGGER,
                 seconds_trigger=DEFAULT_SECONDS_TRIGGER, image_limit=8, state=None,
                 review_event_limit=DEFAULT_REVIEW_EVENT_LIMIT,
                 active_event_limit=DEFAULT_ACTIVE_EVENT_LIMIT):
        if type(event_trigger) is not int or event_trigger < 1:
            raise ValueError('event trigger must be positive')
        if type(image_limit) is not int or not 2 <= image_limit <= 8:
            raise ValueError('image limit must be 2..8')
        if type(review_event_limit) is not int or review_event_limit < 1:
            raise ValueError('review event limit must be positive')
        if type(active_event_limit) is not int or active_event_limit < 1:
            raise ValueError('active event limit must be positive')
        self.backend = backend; self.event_trigger = event_trigger
        self.seconds_trigger = finite(seconds_trigger); self.image_limit = image_limit
        self.review_event_limit = review_event_limit
        self.active_event_limit = active_event_limit
        self.state = deepcopy(state) if state is not None else {
            'last_review_at': None, 'new_event_ids_since_review': [], 'dirty_event_ids_since_review': []}
        if set(self.state) != {'last_review_at', 'new_event_ids_since_review', 'dirty_event_ids_since_review'}:
            raise ValueError('invalid buffer checkpoint')

    def checkpoint(self): return deepcopy(self.state)

    def _due(self, now, roots, new_ids, changed_ids, force):
        dirty = list(dict.fromkeys(self.state['dirty_event_ids_since_review'] + list(changed_ids)))
        if not roots and not dirty: return False, 'no_pending_or_dirty_events'
        queued = list(dict.fromkeys(self.state['new_event_ids_since_review'] + list(new_ids)))
        if force: return True, 'forced_final_review'
        if len(queued) >= self.event_trigger: return True, 'event_trigger'
        last = self.state['last_review_at']
        if last is not None and now - last >= self.seconds_trigger: return True, 'time_trigger'
        return False, 'buffering'

    def _input(self, store, now, dirty_event_ids, *, review_limit=None, active_limit=None):
        state = store.snapshot(); roots = _ordered_roots(state)
        review_limit = self.review_event_limit if review_limit is None else review_limit
        active_limit = self.active_event_limit if active_limit is None else active_limit
        reviewed_roots, deferred_tail = roots[:review_limit], roots[review_limit:]
        open_phases = [p for p in state['phases'].values() if p['status'] == 'open']
        open_phases.sort(key=lambda p: (p['start'], p['id']))
        active = open_phases[-1] if open_phases else None
        active_episode = None
        if active:
            active_episode = next(p for p in state['episodes'].values() if active['id'] in p['phase_ids'])
        # All captions are textual. Images only provide bounded visual anchors.
        events = [{k: deepcopy(state['events'][eid][k]) for k in ('id','caption','frame_ids','start','end')}
                  for eid in reviewed_roots]
        active_payload = None
        anchor_ids = []
        for eid in reviewed_roots:
            fs = state['events'][eid]['frame_ids']; anchor_ids.extend((fs[0], fs[-1]))
        if active:
            active_ids = list(active['event_ids'])
            # A continued source E must be visible to the organizer even when
            # it lies outside a long phase's recent suffix: otherwise a PH/P
            # refresh would be asked to summarize a caption it cannot inspect.
            recent_active_ids = active_ids[-active_limit:]
            dirty_active_ids = [eid for eid in dirty_event_ids if eid in active_ids]
            shown_active_ids = list(dict.fromkeys(recent_active_ids + dirty_active_ids))
            active_events = [{k: deepcopy(state['events'][eid][k]) for k in ('id','caption','frame_ids','start','end')}
                             for eid in shown_active_ids]
            active_payload = {'phase': deepcopy(active), 'episode': deepcopy(active_episode),
                              'events': active_events,
                              'omitted_event_ids':[eid for eid in active_ids if eid not in shown_active_ids]}
            for eid in shown_active_ids:
                anchor_ids.append(state['events'][eid]['frame_ids'][-1])
        anchor_ids = list(dict.fromkeys(anchor_ids))
        if len(anchor_ids) > self.image_limit:
            anchor_ids = [anchor_ids[round(i*(len(anchor_ids)-1)/(self.image_limit-1))]
                          for i in range(self.image_limit)]
        evidence = [state['frames'][fid] for fid in frame_order(state['frames'], anchor_ids)]
        verify_evidence(evidence)
        def next_id(prefix, values):
            return prefix + str(1 + max([int(x[len(prefix):]) for x in values] or [0]))
        return {'observed_until': now, 'policy': POLICY, 'pending_event_ids': reviewed_roots,
                'pending_events': events, 'active_context': active_payload,
                'deferred_tail_event_ids': deferred_tail,
                'dirty_event_ids': list(dirty_event_ids),
                'next_phase_id': next_id('PH', state['phases']),
                'next_episode_id': next_id('P', state['episodes']),
                'visible_support_frame_ids': [f['id'] for f in evidence]}, evidence

    def _apply(self, store, decision, now, changed_ids, *, reviewed_roots=None):
        required = {'groups','deferred_event_ids','refreshes','reason'}
        if not isinstance(decision, dict) or set(decision) != required or not isinstance(decision['groups'], list) or not isinstance(decision['refreshes'], list):
            raise ValueError('invalid buffered organizer fields')
        text(decision['reason'])
        s = store.snapshot(); roots = _ordered_roots(s); cursor = 0
        reviewed_roots = roots if reviewed_roots is None else ids(list(reviewed_roots))
        if reviewed_roots != roots[:len(reviewed_roots)]:
            raise ValueError('reviewed roots must be a consecutive pending prefix')
        phases = {}; episodes = {}; phase_work = deepcopy(s['phases']); episode_work = deepcopy(s['episodes'])
        active_phases=sorted((p for p in s['phases'].values() if p['status']=='open'),key=lambda p:(p['start'],p['id']))
        active_phase_id=active_phases[-1]['id'] if active_phases else None
        active_episode_id=None
        if active_phase_id: active_episode_id=next(p['id'] for p in s['episodes'].values() if active_phase_id in p['phase_ids'])
        def phase_value(phid, title, summary, event_ids):
            text(summary, 80)
            return _node(phid, title, summary, 'event_ids', event_ids, {**s, 'phases': phase_work})
        def episode_value(pid, title, summary, phase_ids):
            return _node(pid, title, summary, 'phase_ids', phase_ids, {**s, 'phases': phase_work})
        def owner_of(phid):
            return next(p for p in episode_work.values() if phid in p['phase_ids'])
        for group_index, group in enumerate(decision['groups']):
            keys = {'event_ids','phase_id','phase_title','phase_summary','episode_id','episode_title','episode_summary'}
            if not isinstance(group, dict) or set(group) != keys: raise ValueError('invalid group fields')
            _bounded_organizer_text(group['phase_title'], 32, f'groups[{group_index}].phase_title')
            _bounded_organizer_text(group['phase_summary'], 80, f'groups[{group_index}].phase_summary')
            _bounded_organizer_text(group['episode_title'], 32, f'groups[{group_index}].episode_title')
            _bounded_organizer_text(group['episode_summary'], 120, f'groups[{group_index}].episode_summary')
            member = ids(group['event_ids'])
            if not member or reviewed_roots[cursor:cursor+len(member)] != member:
                raise ValueError('groups must consume an exact consecutive root prefix')
            cursor += len(member)
            phid, pid = text(group['phase_id']), text(group['episode_id'])
            if not re.fullmatch(r'PH[1-9][0-9]*', phid) or not re.fullmatch(r'P[1-9][0-9]*', pid):
                raise ValueError('invalid phase or episode ID')
            existing_ph = phase_work.get(phid)
            existing_p = episode_work.get(pid)
            if existing_ph:
                if existing_ph['status'] != 'open' or phid != active_phase_id:
                    raise ValueError('only current open phase may receive buffered events')
                owner = owner_of(phid)
                if owner['id'] != pid or pid != active_episode_id or owner['status'] != 'open': raise ValueError('open phase must keep its current open episode')
                combined = existing_ph['event_ids'] + member
                phases[phid] = phase_value(phid, group['phase_title'], group['phase_summary'], combined)
                phase_work[phid] = phases[phid]
                episodes[pid] = episode_value(pid, group['episode_title'], group['episode_summary'], owner['phase_ids'])
                episode_work[pid] = episodes[pid]
            else:
                if phid in phases: raise ValueError('phase repeated')
                expected_ph='PH'+str(1+max([int(x[2:]) for x in phase_work] or [0]))
                if phid != expected_ph: raise ValueError('new phase must use next_phase_id')
                phases[phid] = phase_value(phid, group['phase_title'], group['phase_summary'], member)
                phase_work[phid] = phases[phid]
                if existing_p:
                    if existing_p['status'] != 'open' or pid != active_episode_id: raise ValueError('cannot append a noncurrent/closed episode')
                    combined = existing_p['phase_ids'] + [phid]
                    episodes[pid] = episode_value(pid, group['episode_title'], group['episode_summary'], combined)
                    episode_work[pid] = episodes[pid]
                else:
                    expected_p='P'+str(1+max([int(x[1:]) for x in episode_work] or [0]))
                    if pid != expected_p: raise ValueError('new episode must use next_episode_id')
                    episodes[pid] = episode_value(pid, group['episode_title'], group['episode_summary'], [phid])
                    episode_work[pid] = episodes[pid]
                active_phase_id=phid;active_episode_id=pid
        deferred = ids(decision['deferred_event_ids'])
        if deferred != reviewed_roots[cursor:]: raise ValueError('deferred events must be exactly the remaining reviewed-root suffix')
        refresh_seen=set()
        for refresh_index, value in enumerate(decision['refreshes']):
            keys={'phase_id','phase_title','phase_summary','episode_id','episode_title','episode_summary'}
            if not isinstance(value,dict) or set(value)!=keys: raise ValueError('invalid refresh fields')
            _bounded_organizer_text(value['phase_title'], 32, f'refreshes[{refresh_index}].phase_title')
            _bounded_organizer_text(value['phase_summary'], 80, f'refreshes[{refresh_index}].phase_summary')
            _bounded_organizer_text(value['episode_title'], 32, f'refreshes[{refresh_index}].episode_title')
            _bounded_organizer_text(value['episode_summary'], 120, f'refreshes[{refresh_index}].episode_summary')
            phid,pid=text(value['phase_id']),text(value['episode_id'])
            if phid in refresh_seen or phid not in phase_work: raise ValueError('unknown or repeated refresh phase')
            phase=phase_work[phid]; owner=owner_of(phid)
            if phase['status']!='open' or owner['status']!='open' or owner['id']!=pid: raise ValueError('refresh requires current open phase and episode')
            phases[phid]=phase_value(phid,value['phase_title'],value['phase_summary'],phase['event_ids']);phase_work[phid]=phases[phid]
            episodes[pid]=episode_value(pid,value['episode_title'],value['episode_summary'],owner['phase_ids']);episode_work[pid]=episodes[pid]
            refresh_seen.add(phid)
        required_refresh={next((phid for phid,ph in s['phases'].items() if eid in ph['event_ids']),None)
                          for eid in changed_ids}
        required_refresh.discard(None)
        grouped_existing={g['phase_id'] for g in decision['groups'] if g['phase_id'] in s['phases']}
        if not required_refresh.issubset(refresh_seen | grouped_existing):
            raise ValueError('changed assigned event requires phase/episode refresh')
        before = store.snapshot()
        try:
            store.write(now=now, phases=list(phases.values()), episodes=list(episodes.values()))
        except Exception:
            if store.snapshot() != before: raise RuntimeError('buffered hierarchy write was not atomic')
            raise
        return {'assigned_event_ids': reviewed_roots[:cursor],
                'deferred_event_ids': deferred + roots[len(reviewed_roots):],
                'phase_ids': list(phases), 'episode_ids': list(episodes)}

    def maybe_review(self, store, now, *, new_event_ids=(), changed_event_ids=(), force=False):
        now = finite(now); state = store.snapshot()
        if now < state['observed_until']: raise ValueError('time moved backwards')
        new_ids = ids(list(new_event_ids)); changed_ids = ids(list(changed_event_ids))
        if any(e not in state['events'] for e in new_ids): raise ValueError('unknown new event')
        roots = _ordered_roots(state)
        all_dirty = list(dict.fromkeys(self.state['dirty_event_ids_since_review'] + changed_ids))
        due, trigger = self._due(now, roots, new_ids, all_dirty, force)
        if not due:
            self.state['new_event_ids_since_review'] = list(dict.fromkeys(self.state['new_event_ids_since_review'] + new_ids))
            self.state['dirty_event_ids_since_review'] = list(dict.fromkeys(self.state['dirty_event_ids_since_review'] + changed_ids))
            # Before the first review this timestamp is the start of the
            # current buffer interval; after a review it is that review time.
            if self.state['last_review_at'] is None and self.state['new_event_ids_since_review']:
                self.state['last_review_at'] = now
            return {'reviewed': False, 'trigger': trigger, 'pending_event_ids': roots}
        review_limit, active_limit = self.review_event_limit, self.active_event_limit
        payload, evidence = self._input(store, now, all_dirty, review_limit=review_limit, active_limit=active_limit)
        # This bounds only the decision view.  Unseen suffix events remain in
        # the store and are deterministically deferred, never deleted.
        while _token_count(self.backend, json.dumps(payload, ensure_ascii=False)) > ORGANIZER_CONTEXT_LIMIT:
            if review_limit > 1:
                review_limit -= 1
            elif active_limit > 1:
                active_limit -= 1
            else:
                raise ValueError('buffered organizer context exceeds cap even for one pending and one active event; history preserved')
            payload, evidence = self._input(store, now, all_dirty,
                                            review_limit=review_limit, active_limit=active_limit)
        raw=''; errors=[]
        for attempt in range(3):
            prompt=json.dumps(payload, ensure_ascii=False)
            if attempt:
                previous = raw if len(raw) <= REPAIR_RAW_LIMIT else '[省略：上次回复超过修复传输预算，请重新生成完整JSON]'
                prompt+='\n只返回完整JSON。上次校验错误：'+errors[-1]+'\n上次回复：'+previous
            raw = self.backend.generate(ORGANIZER, prompt, evidence, stage=f'buffered-organize:{now:.6f}:{attempt}', max_tokens=1536)
            try:
                # A length marker is insufficient to reject a reply: accept it
                # only if the strict JSON and store transaction both succeed.
                decision, tail = parse_manager_json(raw)
                applied = self._apply(store, decision, now, all_dirty,
                                      reviewed_roots=payload['pending_event_ids']); break
            except ValueError as exc: errors.append(str(exc))
        else: raise ValueError('buffered organizer failed: '+repr(errors))
        self.state = {'last_review_at': now, 'new_event_ids_since_review': [], 'dirty_event_ids_since_review': []}
        return {'reviewed': True, 'trigger': trigger, 'payload': payload, 'decision': decision,
                'tail': tail, 'raw': raw, 'attempts': attempt+1, 'rejected_attempts':errors,
                'visible_support_frame_ids': payload['visible_support_frame_ids'],
                'decision_view': {'reviewed_event_ids': payload['pending_event_ids'],
                                  'deferred_tail_event_ids': payload['deferred_tail_event_ids'],
                                  'active_omitted_event_ids': (payload['active_context'] or {}).get('omitted_event_ids', [])},
                **applied}
