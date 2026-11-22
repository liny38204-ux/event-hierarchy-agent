"""Bounded working buffer with immediate, atomic within-phase compression.

The working buffer is the current open phase's E plus unassigned roots. The
carry E remains a child of its PH; it is referenced, never copied into roots.
Closed PH/P history and direct frame storage are not bounded by this policy.
"""
from copy import deepcopy
import json
import re

from .buffered_phase import BufferedPhaseManager
from .direct_frame_memory import finite, frame_order, ids, text, verify_evidence
from .phase_end_collapse import PhaseEndCollapseStore, _token_count
from .protocol import parse_manager_json
from .timed_caption import caption_blocks, evidence_caption, render_blocks, time_header

SCHEMA = 'rolling_phase_memory_v11'
LEGACY_POLICY = 'rolling_phase_timed_actions_v9'
REFERENCE_POLICY_V1 = 'rolling_phase_event_refs_v1'
POLICY = 'rolling_phase_event_refs_v2'
# Request more concise wording for a long carry while accepting its complete text.
COMPACT_CARRY_AT = 2600
BUFFER_CAPACITY = 5
REVIEW_SECONDS = 30.0
CONTEXT_LIMIT = 9000
OUTPUT_TOKENS = 4096


# Split explicit clause boundaries while leaving alternatives, quoted speech
# and shared uncertainty intact. Identical action lines may be separate acts.
_ACTION_BOUNDARY = re.compile(r"[，,](?=(?:随后|接着|继而|然后)(?!的))|[，,]?并(?=将|把|放回|递回|收回|送回|归还)")
_SHARED_SCOPE = re.compile(r'或|可能|似乎|好像|疑似|不清|不确定|无法|没有|尚未|同时|一边|[“”「」《》"]')


def separate_action_lines(caption):
    lines = []
    for line in caption.splitlines():
        if _SHARED_SCOPE.search(line):
            lines.append(line)
            continue
        parts = _ACTION_BOUNDARY.split(line)
        if not all(part.strip() for part in parts):
            lines.append(line)
            continue
        for part in parts[:-1]:
            lines.append(part.rstrip('，,；; 。') + '。')
        lines.append(parts[-1])
    return '\n'.join(lines)


def working_buffer(state):
    active = [p for p in state['phases'].values() if p['status'] == 'open']
    if len(active) > 1:
        raise ValueError('only one open phase may be carried')
    carry = active[0]['event_ids'] if active else []
    roots = sorted(state['roots'], key=lambda e: (state['events'][e]['start'], e))
    return list(carry) + roots


class RollingPhaseStore(PhaseEndCollapseStore):
    @staticmethod
    def validate(state):
        PhaseEndCollapseStore.validate(state)
        if len(working_buffer(state)) > BUFFER_CAPACITY:
            raise ValueError('working buffer exceeds five events; review before another arrival')

    @staticmethod
    def validate_reviewed(state):
        RollingPhaseStore.validate(state)
        if state['roots'] or any(len(p['event_ids']) != 1 for p in state['phases'].values()):
            raise ValueError('review must assign all roots and leave one event per phase')
        closed = {p for p, node in state['phases'].items() if node['status'] == 'closed'}
        if set(state['completed_phase_ids']) != closed:
            raise ValueError('every closed phase must be finalized')

    def compress_groups(self, captions, now):
        """Used on a private transaction copy after validated grouping.

        A singleton retains its ID and frames. A merged group receives a new E
        owning the whole frame union. Both use the reviewed caption.
        """
        with self._lock:
            s = self.snapshot()
            if finite(now) != s['observed_until']:
                raise ValueError('compression must use the grouping frontier')
            replacements = []
            for pid, caption in captions.items():
                if pid not in s['phases'] or pid in s['completed_phase_ids']:
                    raise ValueError('cannot recompress a finalized phase')
                text(caption)
                phase = s['phases'][pid]
                members = list(phase['event_ids'])
                if len(members) == 1:
                    singleton = s['events'][members[0]]
                    changed = singleton['caption'] != caption
                    singleton['caption'] = caption
                    replacements.append({'phase_id': pid, 'consumed_event_ids': [],
                                         'event_id': members[0], 'rewritten': changed})
                    continue
                fs = frame_order(s['frames'], [f for e in members for f in s['events'][e]['frame_ids']])
                eid = f"E{s['next_event_id']}"
                s['next_event_id'] += 1
                replacement = {'id': eid, 'caption': caption, 'frame_ids': fs,
                               'start': s['frames'][fs[0]]['time'], 'end': s['frames'][fs[-1]]['time']}
                for old in members:
                    del s['events'][old]
                s['events'][eid] = replacement
                phase['event_ids'] = [eid]
                replacements.append({'phase_id': pid, 'consumed_event_ids': members,
                                     'event_id': eid, 'rewritten': True})
            # A previously compressed carry E is already final when a later
            # phase closes it. No redundant close-time inference is needed.
            s['completed_phase_ids'] = [p for p, node in s['phases'].items() if node['status'] == 'closed']
            self.validate_reviewed(s)
            self._state = s
            return replacements

    def commit_review(self, before, after):
        with self._lock:
            if self._state != before:
                raise ValueError('store changed during review; refusing stale decision')
            self.validate_reviewed(after)
            self._state = deepcopy(after)


_GROUPING = '''你管理流式视频的滚动层级记忆。工作缓存最多5个事件：当前进行中阶段的一个已压缩事件carry，加上本轮新到的pending_events。没有carry时只有pending_events。每条caption是逐行观察记录，不是问答，也不使用未来视频、问题或答案。

本次必须按时间顺序处理全部pending_event_ids，不允许WAIT、留空groups或deferred。不确定的是画面含义时，用可观察内容和不确定措辞组织；不要为满足规则编造人物身份、动作、次数或因果。不能因为没看到阶段结束就拒绝归组：可以先建立进行中的P和PH，再继续扩充。

P是较宽的活动过程，PH是其中局部对象/步骤/观察焦点一致的一段。若新事件延续当前具体活动，将其加入当前PH；若具体步骤或焦点实质转换，在同一P内建后续PH；整体活动也转换时才建新P和新PH。同一地点、同一人物、宽泛主题相同不足以证明同一PH；不能不断扩大旧PH含义。视角变化、短暂遮挡或同一动作细节也不自动形成新PH。

groups完整按顺序分组全部pending_event_ids，每个event只出现一次，每组必须是连续前缀且非空，同一phase_id最多一组。当前carry已经属于当前PH，不要把carry ID写进groups.event_ids，也不能移动、拆分carry或修改已结束阶段。新PH自动关闭前一PH；最后PH保持进行中。问答截点/EOF不意味着实际活动结束。

'''

LEGACY_ORGANIZER = _GROUPING + '''每组的 E.caption 按观察时间窗写。输入 pending_events 的 caption 带有经过帧时间校验的时间标题。先确定该组有哪些事件。逐段复制它们的原始标题（例如 <54s - 63s>，单帧可以是 <54s - 54s>），标题独占一行；同一时间窗内，每行用一个完整中文句子写一次有意义的动作或状态变化。时间是观察范围；不能估计动作精确开始/结束，不能扩大、拼接或拆分输入时间窗。只在对应时间窗内整理观察；不要把前段的动作归到后段。段内动作无法区分先后或主体时保留不确定性。行数由实际变化决定。

在同一阶段内，背景、未变物体和未间断的姿势首次交代即可；后续只有这些重复内容的时间窗可以省略。若换手、放回、取出、整理、又做一次同样的动作，分别逐行写出，不靠重复出现的同一画面增加次数。使用稳定称呼区分不同人的手。每个阶段只有一个E。

新建阶段返回完整时间块；只有静物也应至少返回一个时间块，描述可确认的状态。续接当前阶段遵守 carry_caption_mode：
* append：已有时间块由程序保留，只返回本批新增的时间块；新画面全是已记录的状态时返回空字符串。
* compact：给出精简的完整正文。保留已有的每个时间标题及其独立动作、重复次数与不确定性，在各自标题下缩短措辞，再加入有新变化的时间块。只有新增画面全是已记录的状态时可以返回空字符串。
P短语最多120字符，PH短语最多80字符，title最多32字符。E.caption保留各时间窗内有意义的观察。最后一次与中途采用相同写法。

程序直接转移所有成员帧的有序并集，删除被压缩事件；不保存source、旧caption、别名或隐藏档案。只看到了visible_support_frame_ids里的图片，不能声称看过其它帧。摘要是有损的，不得为了显得完整而补剧情。

使用next_phase_id/next_episode_id作为新节点起始编号，多组依次递增；已有ID只能使用当前进行中的PH/P。只输出JSON。下面时间标题和动作仅示范序列化格式；真实输出只用输入提供的标题与观察：
{"groups":[{"event_ids":["E2","E3"],"phase_id":"PH1","phase_title":"具体局部活动","phase_summary":"一句话概括本阶段。","episode_id":"P1","episode_title":"整体活动","episode_summary":"一句话概括整个活动。","caption":"<4s - 6s>\\n她把一本书放上架。\\n她整理手里余下的书。\\n<8s - 9s>\\n她又放上一本书。"}],"reason":"说明具体延续内容或阶段边界。"}
'''

ORGANIZER = _GROUPING + r'''每条输入 caption 按观察块标记，例如 [E2]。同一事件有多个原始观察窗时，各块标为 [E6:1]、[E6:2]。每组输出的 caption 用对应标记作为独占一行的标题，其后每行用一个完整中文句子描述一次有意义的动作或状态变化。直接引用输入列出的标记，时间窗由程序填写，不要写时间标题、不要自行创造标记。组内按原顺序使用，每个标记最多一次，只在对应标记下整理该段的观察。段内先后或主体不清时保留不确定性。
在同一阶段内，背景、未变物体和未间断的姿势首次交代即可；后续只有这些重复内容的新增观察块可省略。换手、放回、取出、整理、又做一次同样动作，分别写出；不能仅凭重复画面增加次数。每阶段一个E。
新PH至少写一个本组标记和观察，即使只有静物。续接当前PH时，append模式只输出有变化的新增标记，旧正文由程序保留；compact模式按原顺序保留全部carry标记、各自独立动作、次数和不确定性，再加入有变化的新块。续接时仅有重复状态可返回空字符串。不可跨阶段借用标记。
P短语最多120字符，PH短语最多80字符，title最多32字符。帧由程序转移，只看到了visible_support_frame_ids里的图片。使用next_phase_id/next_episode_id作为新节点起始编号，多组依次递增；已有ID只能使用当前进行中的PH/P。
只输出JSON。格式示例（实际标记取自该组输入）：
{"groups":[{"event_ids":["E2","E3"],"phase_id":"PH1","phase_title":"具体局部活动","phase_summary":"一句话概括本阶段。","episode_id":"P1","episode_title":"整体活动","episode_summary":"一句话概括整个活动。","caption":"[E2]\n她把一本书放上架。\n她整理手里余下的书。\n[E3]\n她又放上一本书。"}],"reason":"说明具体延续内容或阶段边界。"}
'''


def observation_refs(event):
    blocks = caption_blocks(event['caption'])
    for index, (bounds, actions) in enumerate(blocks, 1):
        ref = event['id'] if len(blocks) == 1 else f"{event['id']}:{index}"
        yield f'[{ref}]', bounds, actions


def reference_payload(payload):
    view = deepcopy(payload)
    events = list(view['pending_events'])
    if view['active_context']:
        events.extend(view['active_context']['events'])
    for event in events:
        event['caption'] = '\n'.join(line for ref, _, actions in observation_refs(event)
                                     for line in [ref, *actions])
    return view


class EventReferenceError(ValueError):
    """A reference in the generated caption does not belong to its event group."""


def resolve_caption(caption, events):
    if not isinstance(caption, str):
        raise ValueError('caption must be a string with event reference headers')
    windows = {ref: bounds for event in events for ref, bounds, _ in observation_refs(event)}
    converted = []
    for raw in caption.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith('['):
            if line not in windows:
                raise EventReferenceError(f'unknown or misplaced reference {line!r}; allowed: {", ".join(windows)}')
            converted.append(time_header(*windows[line]))
        elif not converted or line.startswith('<') or re.search(r'<\s*\d[^<>]*s\s*>|\[E\d+(?::\d+)?\]', line):
            raise EventReferenceError('use an allowed event reference on its own line; the program supplies timestamps')
        else:
            converted.append(line)
    return '\n'.join(converted)


class PhaseGroupingError(ValueError):
    """The phase assignment is ambiguous and needs a model decision."""


class RollingReviewFailure(ValueError):
    def __init__(self, now, errors, issues):
        super().__init__(f'rolling manager failed without changing memory: {errors}')
        self.diagnostic = {'stage': 'rolling-organize', 'time': now, 'attempt_errors': issues}


class RollingPhaseManager:
    def __init__(self, backend, *, seconds_trigger=REVIEW_SECONDS, image_limit=8, state=None, policy=POLICY):
        if policy not in (POLICY, REFERENCE_POLICY_V1, LEGACY_POLICY):
            raise ValueError('unknown rolling manager policy')
        self.policy = policy
        if finite(seconds_trigger) <= 0 or type(image_limit) is not int or image_limit < BUFFER_CAPACITY:
            raise ValueError('positive interval and at least five image slots required')
        self.backend, self.seconds_trigger, self.image_limit = backend, seconds_trigger, image_limit
        self.state = deepcopy(state) if state is not None else {'last_review_at': 0.0}
        if set(self.state) != {'last_review_at'} or finite(self.state['last_review_at']) < 0:
            raise ValueError('invalid rolling manager checkpoint')

    def checkpoint(self):
        return deepcopy(self.state)

    def _input(self, store, now):
        s = store.snapshot()
        # Reuse chronological grouping context, without v10's truncated tail.
        helper = BufferedPhaseManager(self.backend, review_event_limit=BUFFER_CAPACITY,
                                      active_event_limit=1, image_limit=self.image_limit)
        payload, _ = helper._input(store, now, [], review_limit=BUFFER_CAPACITY, active_limit=1)
        if payload['deferred_tail_event_ids']:
            raise ValueError('rolling review may not hide pending captions')
        buffer = working_buffer(s)
        if payload['active_context'] and len(payload['active_context']['phase']['event_ids']) != 1:
            raise ValueError('open phase must have one compressed carry event before review')
        # At least one actually displayed frame for every buffer event, then
        # add last frames. Caption/frame references themselves are never cut.
        chosen = list(dict.fromkeys(s['events'][e]['frame_ids'][0] for e in buffer))
        for eid in reversed(buffer):
            last = s['events'][eid]['frame_ids'][-1]
            if last not in chosen and len(chosen) < self.image_limit:
                chosen.append(last)
        chosen = frame_order(s['frames'], chosen)
        evidence = [s['frames'][f] for f in chosen]
        verify_evidence(evidence)
        payload.update(policy=self.policy, buffer_event_ids=buffer,
                       visible_support_frame_ids=chosen,
                       timer_origin=self.state['last_review_at'])
        # Present the prior account before the new observations in model input.
        active = payload.pop('active_context')
        caption_mode = None
        for event in [*payload['pending_events'], *(active['events'] if active else [])]:
            event['caption'] = evidence_caption(event, s['frames'])
        if active:
            previous = active['events'][0]['caption']
            caption_mode = 'append' if len(previous) < COMPACT_CARRY_AT else 'compact'
        return {'active_context': active, 'carry_caption_mode': caption_mode, **payload}, evidence

    def _apply(self, store, decision, payload, now):
        if not isinstance(decision, dict) or set(decision) != {'groups', 'reason'}:
            raise ValueError('rolling reply requires groups and reason only; no deferred/WAIT')
        text(decision['reason'])
        groups = decision['groups']
        if not isinstance(groups, list) or not groups:
            raise ValueError('all pending events must be assigned; WAIT is not allowed')
        active = payload['active_context']
        pending = {e['id']: e for e in payload['pending_events']}
        captions, plain = {}, []
        for group in groups:
            if not isinstance(group, dict) or 'caption' not in group:
                raise ValueError('every group needs a compressed caption')
            g = dict(group)
            caption = g.pop('caption')
            pid = g.get('phase_id')
            if not isinstance(pid, str):
                raise PhaseGroupingError('phase_id must be a string such as PH1')
            repeated = pid in captions
            member_ids = ids(g.get('event_ids'))
            if not member_ids or any(e not in pending for e in member_ids):
                raise ValueError('group may only reference its pending events')
            continuing = active and pid == active['phase']['id']
            previous = active['events'][0]['caption'] if continuing else ''
            compact = continuing and payload['carry_caption_mode'] == 'compact'
            if repeated:
                last = plain[-1]
                if self.policy != POLICY:
                    raise ValueError('each phase must be grouped once')
                if last['phase_id'] != pid:
                    raise PhaseGroupingError('cannot reopen a phase across an intervening phase')
                same_description = (g.keys() == last.keys() and
                                    all(g[k] == last[k] for k in g if k != 'event_ids'))
                if compact or not same_description:
                    raise PhaseGroupingError('same phase appears twice; return one group with all consecutive '
                                             'event_ids and one consistent phase and episode description')
                # This fragment still resolves references against its own event_ids.
                previous = captions[pid]
            if previous and caption == '':
                caption = previous
            else:
                # A valid boundary must be an entire observed window from this
                # group, not just an arbitrary pair of frame times in the phase.
                allowed = {bounds for eid in member_ids
                           for bounds, _ in caption_blocks(pending[eid]['caption'])}
                old = caption_blocks(previous) if previous else []
                old_ranges = {bounds for bounds, _ in old}
                if compact:
                    allowed |= old_ranges
                if self.policy != LEGACY_POLICY:
                    sources = [pending[eid] for eid in member_ids]
                    if compact:
                        sources.extend(active['events'])
                    caption = resolve_caption(caption, sources)
                blocks = caption_blocks(caption)
                for bounds, _ in blocks:
                    if bounds not in allowed:
                        raise ValueError(f'{time_header(*bounds)} is not an input window of this group')
                if compact and not old_ranges.issubset(bounds for bounds, _ in blocks):
                    raise ValueError('compaction must retain each existing observation window')
                updated = [(bounds, separate_action_lines('\n'.join(lines)).splitlines())
                           for bounds, lines in blocks]
                if previous and not compact:
                    updated = old + updated
                caption = render_blocks(updated)
                # Includes the boundary between the carry and the new blocks.
                caption_blocks(caption)
            captions[pid] = caption
            if repeated:
                plain[-1]['event_ids'] = plain[-1]['event_ids'] + member_ids
            else:
                plain.append(g)
        before = store.snapshot()
        trial = RollingPhaseStore(before)
        # Group and compress on a private copy. A bad late group, caption or
        # stale commit cannot leave half-mutated hierarchy or deleted events.
        grouping = {'groups': plain, 'reason': decision['reason'],
                    'deferred_event_ids': [], 'refreshes': []}
        helper = BufferedPhaseManager(self.backend)
        applied = helper._apply(trial, grouping, now, [], reviewed_roots=payload['pending_event_ids'])
        replacements = trial.compress_groups(captions, now)
        after = trial.snapshot()
        if set(before['frames']) != set(after['frames']) or {
                f for e in before['events'].values() for f in e['frame_ids']} != {
                f for e in after['events'].values() for f in e['frame_ids']}:
            raise ValueError('rolling review lost frame coverage')
        store.commit_review(before, after)
        return {**applied, 'replacements': replacements,
                'buffer_event_ids_before': working_buffer(before),
                'buffer_event_ids_after': working_buffer(after)}

    def maybe_review(self, store, now, *, force=False):
        now = finite(now)
        s = store.snapshot()
        if now != s['observed_until'] or now < self.state['last_review_at']:
            raise ValueError('review must use current causal frontier')
        RollingPhaseStore.validate(s)
        buffer = working_buffer(s)
        if not s['roots']:
            return {'reviewed': False, 'trigger': 'no_new_events', 'buffer_event_ids': buffer}
        if force:
            trigger = 'forced_final_review'
        elif len(buffer) >= BUFFER_CAPACITY:
            trigger = 'buffer_capacity'
        elif now - self.state['last_review_at'] >= self.seconds_trigger:
            trigger = 'time_trigger'
        else:
            return {'reviewed': False, 'trigger': 'buffering', 'buffer_event_ids': buffer}
        payload, evidence = self._input(store, now)
        view = reference_payload(payload) if self.policy != LEGACY_POLICY else payload
        prompt = ORGANIZER if self.policy != LEGACY_POLICY else LEGACY_ORGANIZER
        base = json.dumps(view, ensure_ascii=False)
        if _token_count(self.backend, base) > CONTEXT_LIMIT:
            raise ValueError('rolling context exceeds cap; no captions silently omitted')
        repair, errors, issues = '', [], []
        for attempt in range(3):
            raw = self.backend.generate(prompt, base + repair, evidence,
                                        stage=f'rolling-organize:{now:.6f}:{attempt}', max_tokens=OUTPUT_TOKENS)
            try:
                if getattr(self.backend, 'calls', []) and self.backend.calls[-1].get('finish_reason') == 'length':
                    raise ValueError('truncated rolling decision')
                decision, tail = parse_manager_json(raw)
                applied = self._apply(store, decision, payload, now)
                self.state = {'last_review_at': now}
                return {'reviewed': True, 'trigger': trigger, 'payload': view,
                        'decision': decision, 'raw': raw, 'tail': tail, 'attempts': attempt + 1,
                        'rejected_attempts': errors, **applied}
            except ValueError as exc:
                errors.append(str(exc))
                if isinstance(exc, json.JSONDecodeError):
                    issues.append('invalid_json')
                elif isinstance(exc, EventReferenceError):
                    issues.append('invalid_event_reference')
                elif isinstance(exc, PhaseGroupingError):
                    issues.append('invalid_phase_grouping')
                elif str(exc) == 'truncated rolling decision':
                    issues.append('output_truncated')
                else:
                    issues.append('invalid_decision')
                if attempt < 2:
                    feedback = '\n修复错误：' + str(exc)
                    repair = feedback + '\n上次输出：' + raw if len(raw) <= 4096 else feedback
                    if _token_count(self.backend, base + repair) > CONTEXT_LIMIT:
                        repair = feedback
                    if _token_count(self.backend, base + repair) > CONTEXT_LIMIT:
                        repair = '\n修复错误：请根据输入重新输出完整合法JSON。'
                    if _token_count(self.backend, base + repair) > CONTEXT_LIMIT:
                        repair = ''
        raise RollingReviewFailure(now, errors, issues)
