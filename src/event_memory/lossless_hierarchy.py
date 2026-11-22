"""Lossless streaming hierarchy (schema v12): P -> PH -> Step -> immutable E.

Design
------
* The writer produces one immutable E per candidate segment: an English timed
  caption, a short English summary, visible state changes and retrieval
  keywords.  Nothing downstream rewrites, merges or deletes an E.
* The manager only places events: which PH and P they belong to, which Step
  inside the PH, and short titles for newly opened nodes.  It reads E summaries
  plus anchor frames, never full captions, so its input is bounded by the
  working buffer instead of by phase length.
* PH/P summaries are written once, by a separate text-only call, when a phase
  closes or a query-time snapshot is taken.  Captions are not involved.
* Steps are optional: procedural videos get an ordered step chain per PH,
  walk-throughs may have none.

Compatibility
-------------
`project_v11_bundle` renders the v12 state as a completed ``rolling_phase_memory_v11``
bundle (one E per PH whose caption concatenates the member captions in time order,
each block headed by its own observation window).  The existing answering agent
loop reads this projection unchanged.  The v12 state itself is the source of truth
and is saved beside it.
"""
from __future__ import annotations

from copy import deepcopy
import json
import re

from .direct_frame_memory import finite, frame_order, ids, text, verify_evidence
from .protocol import parse_manager_json
from .timed_caption import caption_blocks, render_blocks, time_header

SCHEMA = 'lossless_step_memory_v12'
SCHEMA_VERSION = 12
POLICY = 'lossless_place_only_v1'
BUFFER_CAPACITY = 5
REVIEW_SECONDS = 30.0
CONTEXT_LIMIT = 9000
PLACE_OUTPUT_TOKENS = 1024
SUMMARY_OUTPUT_TOKENS = 512
ANCHOR_IMAGES = 8
RECENT_STEPS_SHOWN = 4

SUMMARY_LIMIT = 300          # characters, English
CHANGE_LIMIT = 8
CHANGE_FIELD_LIMIT = 80
KEYWORD_LIMIT = 12
KEYWORD_CHARS = 40
TITLE_LIMIT = 60
STEP_LABEL_LIMIT = 80
PHASE_SUMMARY_LIMIT = 240
EPISODE_SUMMARY_LIMIT = 300

E_KEYS = {'id', 'caption', 'summary', 'changes', 'keywords', 'frame_ids', 'start', 'end'}
CHANGE_KEYS = {'object', 'change', 'result'}
STATE_KEYS = {'frames', 'events', 'steps', 'phases', 'episodes', 'roots', 'observed_until',
              'next_event_id', 'next_step_id', 'next_phase_id', 'next_episode_id'}
STEP_KEYS = {'id', 'label', 'phase_id', 'event_ids', 'start', 'end'}
NODE_KEYS = {'id', 'title', 'summary', 'status', 'start', 'end', 'closed_at'}


# --------------------------------------------------------------------------
# writer contract
# --------------------------------------------------------------------------

WRITER = '''You record one causal segment of a video stream as English memory for later question answering.
The frames are shown in chronological order. Describe only what they show; the program assigns the observation window.

Return exactly one JSON object:
{"caption":"line one\\nline two","summary":"...","changes":[{"object":"...","change":"...","result":"..."}],"keywords":["..."]}

caption: chronological English sentences, one observation per line, separated by \\n.
- Make people, objects and places the subjects. Record each meaningful action or state change in order, and write repetitions as separate lines (a hand puts down one card, arranges cards, then puts down another).
- Locate objects by physical relations: what holds it, what it rests on, what it is next to, which room or fixture. Use left/right only relative to a person or a named object. Omit screen positions and camera commentary (panning, zooming, entering the frame); when a new view reveals an object, say what became visible and where it is.
- Copy legible text exactly in quotes. Give counts when they are visible.
- Name an object only when its identity is visually clear. When its shape is broken, blurred, partly occluded or looks like a reconstruction artifact, describe visible features instead ("an irregular pale object on the side table"), never a guessed identity.
- Do not invent movement, identity, intention, cause, speech or text. A single pose does not prove a movement. Do not claim persistence through gaps.
- Mention the same unchanged scene detail once. Prefer at most 150 words.

summary: one self-contained English sentence (at most 40 words) naming where this happens, the main person or object, and the concrete activity or viewing focus, with the key result. Another program groups events using only this summary, so be specific (use "slices onions on a wooden board", not "cooking activity").

changes: visible state changes only, in order. object = the thing that changed; change = verb phrase (picked up, placed, opened, poured into, turned on, appeared, removed); result = its resulting location or state. At most 8. Use [] for a static segment.

keywords: 3 to 12 lowercase English nouns or short noun phrases for retrieval: distinctive objects, fixtures, rooms, legible text. No verbs, no sentences.

No other fields, no time headers, no markdown, no explanations.'''

WRITER_REPAIR = ('Return only one JSON object with exactly the keys caption, summary, changes and keywords. '
                 'Fix this validation error: ')

_TIME_HEADER = re.compile(r'<\s*\d+(?:\.\d+)?\s*s\s*-\s*\d+(?:\.\d+)?\s*s\s*>')


def _clean_text(value, field, limit=None):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field} must be a nonempty string')
    value = ' '.join(value.split()) if '\n' not in value else value.strip()
    if limit and len(value) > limit:
        raise ValueError(f'{field} length {len(value)} exceeds {limit}; shorten it')
    return value


def parse_writer_output(raw):
    """Validate the writer JSON. Returns caption body, summary, changes and keywords."""
    parsed, _ = parse_manager_json(raw)
    if not isinstance(parsed, dict) or set(parsed) != {'caption', 'summary', 'changes', 'keywords'}:
        raise ValueError('writer must return exactly caption, summary, changes and keywords')
    body = parsed['caption']
    if not isinstance(body, str) or not body.strip():
        raise ValueError('caption must be a nonempty string')
    if _TIME_HEADER.search(body):
        raise ValueError('caption must not contain time headers; the program adds them')
    lines = [line.strip() for line in body.strip().splitlines() if line.strip()]
    if not lines:
        raise ValueError('caption needs at least one observation line')
    summary = _clean_text(parsed['summary'], 'summary', SUMMARY_LIMIT)
    changes = parsed['changes']
    if not isinstance(changes, list) or len(changes) > CHANGE_LIMIT:
        raise ValueError(f'changes must be a list of at most {CHANGE_LIMIT} items')
    clean_changes = []
    for index, item in enumerate(changes):
        if not isinstance(item, dict) or set(item) != CHANGE_KEYS:
            raise ValueError(f'changes[{index}] needs exactly object, change and result')
        clean_changes.append({k: _clean_text(item[k], f'changes[{index}].{k}', CHANGE_FIELD_LIMIT)
                              for k in ('object', 'change', 'result')})
    keywords = parsed['keywords']
    if not isinstance(keywords, list) or not 1 <= len(keywords) <= KEYWORD_LIMIT:
        raise ValueError(f'keywords must be a list of 1 to {KEYWORD_LIMIT} short strings')
    clean_keywords = []
    for index, word in enumerate(keywords):
        word = _clean_text(word, f'keywords[{index}]', KEYWORD_CHARS).lower()
        if word not in clean_keywords:
            clean_keywords.append(word)
    return '\n'.join(lines), summary, clean_changes, clean_keywords


def writer_event(event_id, body, summary, changes, keywords, evidence):
    """Attach the program-owned observation window to a validated writer record."""
    times = [finite(frame['time']) for frame in evidence]
    if not times or times != sorted(times):
        raise ValueError('writer evidence must be nonempty and chronological')
    caption = time_header(times[0], times[-1]) + '\n' + body
    caption_blocks(caption)
    return {'id': event_id, 'caption': caption, 'summary': summary, 'changes': changes,
            'keywords': keywords, 'frame_ids': [frame['id'] for frame in evidence],
            'start': times[0], 'end': times[-1]}


# --------------------------------------------------------------------------
# store
# --------------------------------------------------------------------------

def _new_state():
    return {'frames': {}, 'events': {}, 'steps': {}, 'phases': {}, 'episodes': {}, 'roots': [],
            'observed_until': 0.0, 'next_event_id': 1, 'next_step_id': 1,
            'next_phase_id': 1, 'next_episode_id': 1}


def _span(records, child_ids):
    return min(records[c]['start'] for c in child_ids), max(records[c]['end'] for c in child_ids)


class LosslessStore:
    """Single-writer state. Every mutation goes through a validated copy."""

    def __init__(self, state=None):
        self._state = deepcopy(state) if state is not None else _new_state()
        self.validate(self._state)

    def snapshot(self):
        return deepcopy(self._state)

    @staticmethod
    def validate(s):
        if set(s) != STATE_KEYS:
            raise ValueError('v12 state has unknown or missing fields')
        now = finite(s['observed_until'])
        for counter in ('next_event_id', 'next_step_id', 'next_phase_id', 'next_episode_id'):
            if type(s[counter]) is not int or s[counter] < 1:
                raise ValueError('invalid id counter')
        for fid, f in s['frames'].items():
            if set(f) != {'id', 'time', 'path', 'sha256'} or f['id'] != fid:
                raise ValueError('invalid frame reference')
            if not 0 <= finite(f['time']) <= now:
                raise ValueError('future or negative frame')
            if not re.fullmatch('[0-9a-f]{64}', f['sha256']):
                raise ValueError('frame requires SHA256')
        owner = {}
        for eid, e in s['events'].items():
            if set(e) != E_KEYS or e['id'] != eid or not re.fullmatch(r'E[1-9][0-9]*', eid):
                raise ValueError('invalid event record')
            if int(eid[1:]) >= s['next_event_id']:
                raise ValueError('event id counter must exceed existing ids')
            caption_blocks(e['caption'])
            text(e['summary'], SUMMARY_LIMIT)
            fs = ids(e['frame_ids'])
            if not fs or any(f not in s['frames'] for f in fs) or fs != frame_order(s['frames'], fs):
                raise ValueError('event needs known, chronological frames')
            for f in fs:
                if f in owner:
                    raise ValueError('a frame belongs to exactly one event')
                owner[f] = eid
            times = [s['frames'][f]['time'] for f in fs]
            if (finite(e['start']), finite(e['end'])) != (min(times), max(times)):
                raise ValueError('event bounds must equal its frame envelope')
        if set(owner) != set(s['frames']):
            raise ValueError('every frame must belong to an event')
        # phases / episodes
        phase_members, step_members = {}, {}
        for pid, p in s['phases'].items():
            if not {'id', 'title', 'summary', 'status', 'start', 'end', 'event_ids', 'step_ids', 'episode_id'} <= set(p) \
                    or set(p) - (NODE_KEYS | {'event_ids', 'step_ids', 'episode_id'}) or p['id'] != pid:
                raise ValueError('invalid phase node')
            text(p['title'], TITLE_LIMIT)
            if p['summary'] is not None:
                text(p['summary'], PHASE_SUMMARY_LIMIT)
            members = ids(p['event_ids'])
            if not members or any(e not in s['events'] for e in members):
                raise ValueError('phase needs known events')
            if members != sorted(members, key=lambda e: (s['events'][e]['start'], int(e[1:]))):
                raise ValueError('phase events must be chronological')
            for e in members:
                if e in phase_members:
                    raise ValueError('event belongs to more than one phase')
                phase_members[e] = pid
            if (finite(p['start']), finite(p['end'])) != _span(s['events'], members):
                raise ValueError('phase bounds must equal its events')
            if p['status'] not in ('open', 'closed'):
                raise ValueError('invalid phase status')
            if p['status'] == 'closed' and not p['end'] <= finite(p.get('closed_at')) <= now:
                raise ValueError('invalid phase close time')
            if p['status'] == 'open' and 'closed_at' in p:
                raise ValueError('open phase has a close time')
            for sid in ids(p['step_ids']):
                if sid not in s['steps'] or s['steps'][sid]['phase_id'] != pid:
                    raise ValueError('phase lists an unknown step')
        for sid, st in s['steps'].items():
            if set(st) != STEP_KEYS or st['id'] != sid or st['phase_id'] not in s['phases']:
                raise ValueError('invalid step node')
            text(st['label'], STEP_LABEL_LIMIT)
            members = ids(st['event_ids'])
            if not members or any(phase_members.get(e) != st['phase_id'] for e in members):
                raise ValueError('step events must belong to its phase')
            for e in members:
                if e in step_members:
                    raise ValueError('event belongs to more than one step')
                step_members[e] = sid
            if (finite(st['start']), finite(st['end'])) != _span(s['events'], members):
                raise ValueError('step bounds must equal its events')
            if sid not in s['phases'][st['phase_id']]['step_ids']:
                raise ValueError('step missing from its phase')
        for pid, p in s['phases'].items():
            order = [s['steps'][sid]['event_ids'] for sid in p['step_ids']]
            flat = [e for group in order for e in group]
            if flat != [e for e in p['event_ids'] if e in step_members]:
                raise ValueError('steps must partition a chronological subsequence of phase events')
        ep_members = {}
        for qid, q in s['episodes'].items():
            if not {'id', 'title', 'summary', 'status', 'start', 'end', 'phase_ids'} <= set(q) \
                    or set(q) - (NODE_KEYS | {'phase_ids'}) or q['id'] != qid:
                raise ValueError('invalid episode node')
            text(q['title'], TITLE_LIMIT)
            if q['summary'] is not None:
                text(q['summary'], EPISODE_SUMMARY_LIMIT)
            members = ids(q['phase_ids'])
            if not members or any(p not in s['phases'] for p in members):
                raise ValueError('episode needs known phases')
            for p in members:
                if p in ep_members or s['phases'][p]['episode_id'] != qid:
                    raise ValueError('phase/episode ownership mismatch')
                ep_members[p] = qid
            if (finite(q['start']), finite(q['end'])) != _span(s['phases'], members):
                raise ValueError('episode bounds must equal its phases')
            if q['status'] == 'closed' and any(s['phases'][p]['status'] != 'closed' for p in members):
                raise ValueError('closed episode has an open phase')
        if set(ep_members) != set(s['phases']):
            raise ValueError('every phase needs an episode')
        if len([p for p in s['phases'].values() if p['status'] == 'open']) > 1:
            raise ValueError('at most one open phase')
        if set(ids(s['roots'])) != set(s['events']) - set(phase_members):
            raise ValueError('roots must be exactly the unplaced events')
        if s['roots'] != sorted(s['roots'], key=lambda e: (s['events'][e]['start'], int(e[1:]))):
            raise ValueError('roots must be chronological')
        if len(s['roots']) > BUFFER_CAPACITY:
            raise ValueError('working buffer exceeds capacity; review before another arrival')

    def add_event(self, event, frames, now):
        s = self.snapshot()
        now = finite(now)
        if now < s['observed_until']:
            raise ValueError('time moved backwards')
        if event['id'] != f"E{s['next_event_id']}":
            raise ValueError('event id must be the next counter value')
        for frame in frames:
            if frame['id'] in s['frames'] and s['frames'][frame['id']] != frame:
                raise ValueError('immutable frame changed')
            s['frames'][frame['id']] = deepcopy(frame)
        s['events'][event['id']] = deepcopy(event)
        s['next_event_id'] += 1
        s['roots'] = sorted(s['roots'] + [event['id']], key=lambda e: (s['events'][e]['start'], int(e[1:])))
        s['observed_until'] = now
        self.validate(s)
        self._state = s
        return event['id']

    def advance(self, now):
        s = self.snapshot()
        now = finite(now)
        if now < s['observed_until']:
            raise ValueError('time moved backwards')
        s['observed_until'] = now
        self.validate(s)
        self._state = s

    def commit(self, before, after):
        if self._state != before:
            raise ValueError('store changed during review; refusing stale decision')
        self.validate(after)
        self._state = deepcopy(after)


def open_phase(state):
    opened = [p for p in state['phases'].values() if p['status'] == 'open']
    return opened[0] if opened else None


def last_episode(state):
    if not state['episodes']:
        return None
    return max(state['episodes'].values(), key=lambda q: (q['start'], int(q['id'][1:])))


def neighbor_index(state):
    """Global chronological order of placed and pending events, for cross-phase neighbors."""
    return sorted(state['events'], key=lambda e: (state['events'][e]['start'], int(e[1:])))


# --------------------------------------------------------------------------
# manager: placement only
# --------------------------------------------------------------------------

PLACER = '''You organize a streaming video memory. Each event E is an immutable record of one short observed segment; you see its English summary and a few anchor frames. You never rewrite captions.

Hierarchy:
- P (episode): a broad observed activity or visit (cooking a dish, touring a home).
- PH (phase): a contiguous part of P with one local focus, area or sub-task (preparing vegetables, the kitchen area).
- Step (inside a PH, optional): one procedural action unit, labelled as a short verb phrase in the style of a recipe or tutorial step ("add ham", "cut the beef into thin slices", "spread glue on the face"). Use steps when people perform an ordered task; a step may span several consecutive E. Leave steps empty for pure observation or walk-throughs.

This review must place EVERY pending event, in time order. You may not wait.
For each pending event choose:
- phase: "current" (continue the open PH, only if it exists), or "new" (open a new PH; the previous PH closes).
- episode: only for phase "new": "current" (same P) or "new" (a new broad activity begins).
- step: "current" (same procedural step as the previous placed event in this PH), "new" (a new step starts), or "none" (no procedural step).
Open a new PH when the concrete focus, area or sub-task really changes; viewpoint changes, short occlusions or repeated detail do not. Same room or same person alone does not prove the same PH. Do not stretch an old PH title to cover a different activity.
Open a new step when the action unit changes (adding a different ingredient, switching from cutting to stirring). Repetitions of the same action unit stay in one step.

Write titles only when you open nodes: phase_title (at most 8 words), episode_title (at most 8 words), step_label (verb phrase, at most 10 words). Leave them "" otherwise.

Return JSON only:
{"placements":[{"event_id":"E5","phase":"new","episode":"current","phase_title":"kitchen counter preparation","episode_title":"","step":"new","step_label":"chop the onion"}]}'''

PHASE_SUMMARIZER = '''Write short English summaries for a video memory hierarchy from the given event summaries and step labels only.
phase_summary: one sentence (at most 35 words) describing what happens in this phase and in what order.
episode_summary: one sentence (at most 45 words) describing the broader activity so far.
Do not add facts that are not in the inputs. Return JSON only: {"phase_summary":"...","episode_summary":"..."}'''


class PlacementError(ValueError):
    """Manager output violated the placement contract."""


class ReviewFailure(ValueError):
    def __init__(self, now, errors):
        super().__init__(f'placement review failed without changing memory: {errors}')
        self.diagnostic = {'stage': 'place', 'time': now, 'attempt_errors': errors}


def _anchor_frames(state, event_ids, limit=ANCHOR_IMAGES):
    chosen = []
    for eid in event_ids:
        fs = state['events'][eid]['frame_ids']
        chosen.append(fs[0])
    for eid in reversed(event_ids):
        last = state['events'][eid]['frame_ids'][-1]
        if last not in chosen and len(chosen) < limit:
            chosen.append(last)
    chosen = frame_order(state['frames'], list(dict.fromkeys(chosen)))[:limit]
    return [state['frames'][f] for f in chosen]


def placement_input(state, now):
    roots = list(state['roots'])
    current = open_phase(state)
    context = None
    shown_context = []
    if current:
        ep = state['episodes'][current['episode_id']]
        steps = [state['steps'][sid] for sid in current['step_ids']][-RECENT_STEPS_SHOWN:]
        last_eid = current['event_ids'][-1]
        shown_context = [last_eid]
        context = {'episode': {'id': ep['id'], 'title': ep['title']},
                   'phase': {'id': current['id'], 'title': current['title'],
                             'start': current['start'], 'end': current['end'],
                             'event_count': len(current['event_ids'])},
                   'recent_steps': [{'label': st['label'], 'start': st['start'], 'end': st['end']} for st in steps],
                   'last_event': {'id': last_eid, 'summary': state['events'][last_eid]['summary'],
                                  'start': state['events'][last_eid]['start'],
                                  'end': state['events'][last_eid]['end'],
                                  'step': next((state['steps'][sid]['label'] for sid in current['step_ids']
                                                if last_eid in state['steps'][sid]['event_ids']), None)}}
    pending = [{'id': e, 'start': state['events'][e]['start'], 'end': state['events'][e]['end'],
                'summary': state['events'][e]['summary'],
                'changes': state['events'][e]['changes']} for e in roots]
    evidence = _anchor_frames(state, shown_context + roots)
    payload = {'observed_until': now, 'current': context, 'pending_events': pending,
               'pending_event_ids': roots, 'visible_frame_ids': [f['id'] for f in evidence]}
    return payload, evidence


def apply_placements(state, decision, now):
    """Return a new state with the pending events placed. Never edits events."""
    if not isinstance(decision, dict) or set(decision) != {'placements'}:
        raise PlacementError('reply must be {"placements":[...]} only')
    items = decision['placements']
    roots = list(state['roots'])
    if not isinstance(items, list) or [p.get('event_id') if isinstance(p, dict) else None for p in items] != roots:
        raise PlacementError(f'placements must list every pending event exactly once in order: {roots}')
    s = deepcopy(state)
    keys = {'event_id', 'phase', 'episode', 'phase_title', 'episode_title', 'step', 'step_label'}
    for index, item in enumerate(items):
        if set(item) != keys:
            raise PlacementError(f'placements[{index}] needs exactly {sorted(keys)}')
        eid = item['event_id']
        event = s['events'][eid]
        current = open_phase(s)
        if item['phase'] not in ('current', 'new'):
            raise PlacementError(f'placements[{index}].phase must be current or new')
        if item['step'] not in ('current', 'new', 'none'):
            raise PlacementError(f'placements[{index}].step must be current, new or none')
        if item['phase'] == 'current':
            if current is None:
                raise PlacementError(f'placements[{index}]: there is no open phase; use phase "new"')
            if item['phase_title'] or item['episode_title'] or item['episode'] not in ('', 'current'):
                raise PlacementError(f'placements[{index}]: continuing a phase takes no new titles and episode ""')
            phase = current
        else:
            title = _title(item['phase_title'], f'placements[{index}].phase_title')
            episode = last_episode(s)
            if item['episode'] == 'current':
                if episode is None or episode['status'] != 'open':
                    raise PlacementError(f'placements[{index}]: no open episode; use episode "new"')
                if item['episode_title']:
                    raise PlacementError(f'placements[{index}]: episode_title only when opening an episode')
            elif item['episode'] == 'new':
                if episode is not None:
                    episode['status'] = 'closed'; episode['closed_at'] = float(now)
                qid = f"P{s['next_episode_id']}"; s['next_episode_id'] += 1
                episode = {'id': qid, 'title': _title(item['episode_title'], f'placements[{index}].episode_title'),
                           'summary': None, 'status': 'open', 'start': event['start'], 'end': event['end'],
                           'phase_ids': []}
                s['episodes'][qid] = episode
            else:
                raise PlacementError(f'placements[{index}].episode must be current or new when opening a phase')
            if current is not None:
                current['status'] = 'closed'; current['closed_at'] = float(now)
            pid = f"PH{s['next_phase_id']}"; s['next_phase_id'] += 1
            phase = {'id': pid, 'title': title, 'summary': None, 'status': 'open',
                     'start': event['start'], 'end': event['end'], 'event_ids': [], 'step_ids': [],
                     'episode_id': episode['id']}
            s['phases'][pid] = phase
            episode['phase_ids'].append(pid)
        phase['event_ids'].append(eid)
        phase['start'], phase['end'] = _span(s['events'], phase['event_ids'])
        ep = s['episodes'][phase['episode_id']]
        ep['start'], ep['end'] = _span(s['phases'], ep['phase_ids'])
        if item['step'] == 'none':
            if item['step_label']:
                raise PlacementError(f'placements[{index}]: step_label only for a new step')
        elif item['step'] == 'current':
            if item['step_label']:
                raise PlacementError(f'placements[{index}]: step_label only for a new step')
            previous = phase['event_ids'][-2] if len(phase['event_ids']) > 1 else None
            owner = next((sid for sid in phase['step_ids'] if previous in s['steps'][sid]['event_ids']), None)
            if owner is None or phase['step_ids'][-1] != owner:
                raise PlacementError(f'placements[{index}]: step "current" needs the previous event of this phase in the latest step; use "new" or "none"')
            st = s['steps'][owner]
            st['event_ids'].append(eid)
            st['start'], st['end'] = _span(s['events'], st['event_ids'])
        else:
            label = _clean_text(item['step_label'], f'placements[{index}].step_label', STEP_LABEL_LIMIT)
            sid = f"S{s['next_step_id']}"; s['next_step_id'] += 1
            s['steps'][sid] = {'id': sid, 'label': label, 'phase_id': phase['id'], 'event_ids': [eid],
                               'start': event['start'], 'end': event['end']}
            phase['step_ids'].append(sid)
        s['roots'].remove(eid)
    LosslessStore.validate(s)
    return s


def _title(value, field):
    value = _clean_text(value, field, TITLE_LIMIT)
    if len(value.split()) > 12:
        raise PlacementError(f'{field} must be a short phrase')
    return value


class LosslessManager:
    """Placement reviews on a 5-event buffer; summaries are filled separately."""

    def __init__(self, backend, *, seconds_trigger=REVIEW_SECONDS, state=None):
        self.backend = backend
        self.seconds_trigger = finite(seconds_trigger)
        self.state = deepcopy(state) if state is not None else {'last_review_at': 0.0}
        self.calls = {'place': 0, 'place_retries': 0, 'summarize': 0}

    def checkpoint(self):
        return deepcopy(self.state)

    def due(self, store_state, now, force=False):
        if not store_state['roots']:
            return None
        if force:
            return 'forced'
        if len(store_state['roots']) >= BUFFER_CAPACITY:
            return 'buffer_capacity'
        if finite(now) - self.state['last_review_at'] >= self.seconds_trigger:
            return 'time_trigger'
        return None

    def maybe_review(self, store, now, *, force=False):
        now = finite(now)
        before = store.snapshot()
        if now != before['observed_until']:
            raise ValueError('review must use the current causal frontier')
        trigger = self.due(before, now, force)
        if trigger is None:
            return {'reviewed': False}
        payload, evidence = placement_input(before, now)
        verify_evidence(evidence)
        base = json.dumps(payload, ensure_ascii=False)
        count = getattr(self.backend, 'token_count', None)
        if count is not None and count(base) > CONTEXT_LIMIT:
            raise ReviewFailure(now, ['placement context exceeds cap'])
        repair, errors = '', []
        for attempt in range(3):
            raw = self.backend.generate(PLACER, base + repair, evidence,
                                        stage=f'place:{now:.3f}:{attempt}', max_tokens=PLACE_OUTPUT_TOKENS)
            self.calls['place'] += 1
            self.calls['place_retries'] += int(attempt > 0)
            try:
                if getattr(self.backend, 'calls', None) and self.backend.calls[-1].get('finish_reason') == 'length':
                    raise PlacementError('placement output truncated')
                decision, _ = parse_manager_json(raw)
                after = apply_placements(before, decision, now)
                closed = [pid for pid, p in after['phases'].items()
                          if p['status'] == 'closed' and before['phases'].get(pid, {}).get('status') != 'closed']
                after = self.summarize(after, closed, now)
                store.commit(before, after)
                self.state = {'last_review_at': now}
                return {'reviewed': True, 'trigger': trigger, 'attempts': attempt + 1,
                        'rejected': errors, 'closed_phase_ids': closed}
            except ValueError as exc:
                errors.append(str(exc))
                repair = '\nFix this error and return the complete JSON again: ' + str(exc)[:400]
        raise ReviewFailure(now, errors)

    def summarize(self, state, phase_ids, now):
        """Text-only summaries for the given phases and their episodes."""
        s = deepcopy(state)
        for pid in phase_ids:
            phase = s['phases'][pid]
            ep = s['episodes'][phase['episode_id']]
            payload = {'phase_title': phase['title'], 'episode_title': ep['title'],
                       'steps': [s['steps'][sid]['label'] for sid in phase['step_ids']],
                       'events': [{'start': s['events'][e]['start'], 'end': s['events'][e]['end'],
                                   'summary': s['events'][e]['summary']} for e in phase['event_ids']],
                       'earlier_phases': [s['phases'][p]['title'] for p in ep['phase_ids'] if p != pid]}
            raw = ''
            for attempt in range(2):
                raw = self.backend.generate(PHASE_SUMMARIZER, json.dumps(payload, ensure_ascii=False), (),
                                            stage=f'summarize:{pid}:{attempt}', max_tokens=SUMMARY_OUTPUT_TOKENS)
                self.calls['summarize'] += 1
                try:
                    parsed, _ = parse_manager_json(raw)
                    if set(parsed) != {'phase_summary', 'episode_summary'}:
                        raise ValueError('summary reply needs phase_summary and episode_summary')
                    phase['summary'] = _clean_text(parsed['phase_summary'], 'phase_summary', PHASE_SUMMARY_LIMIT)
                    ep['summary'] = _clean_text(parsed['episode_summary'], 'episode_summary', EPISODE_SUMMARY_LIMIT)
                    break
                except ValueError:
                    continue
            else:
                # A summary is a convenience for the catalog; never fail the memory on it.
                phase['summary'] = _fallback_summary(s, phase)
                if ep['summary'] is None:
                    ep['summary'] = ep['title']
        LosslessStore.validate(s)
        return s

    def finalize_snapshot(self, store, now):
        """Fill missing summaries (open phase/episode) for a query-time snapshot."""
        before = store.snapshot()
        missing = [pid for pid, p in before['phases'].items() if p['summary'] is None]
        if not missing:
            return before
        after = self.summarize(before, missing, now)
        store.commit(before, after)
        return store.snapshot()


def _fallback_summary(state, phase):
    first = state['events'][phase['event_ids'][0]]['summary']
    return first[:PHASE_SUMMARY_LIMIT]


# --------------------------------------------------------------------------
# projection for the existing v11 agent loop
# --------------------------------------------------------------------------

V11_SCHEMA = 'rolling_phase_memory_v11'


def _event_block(event, steps_by_event):
    blocks = caption_blocks(event['caption'])
    rendered = []
    for bounds, lines in blocks:
        extra = []
        step = steps_by_event.get(event['id'])
        if step is not None:
            extra.append(f'Step: {step}')
        if event['keywords']:
            extra.append('Keywords: ' + ', '.join(event['keywords']))
        for change in event['changes']:
            extra.append(f"Change: {change['object']} {change['change']} -> {change['result']}")
        rendered.append((bounds, [*lines, *extra]))
        steps_by_event = {}  # annotate only the first block
    return rendered


def project_v11_bundle(state, video_id, *, provenance=None):
    """Render a v12 state as a completed rolling v11 bundle for the unchanged reader.

    Each PH becomes one E whose caption is the ordered concatenation of its
    member blocks, each block keeping its own observation window.  Unplaced
    roots (none after a forced review) become singleton phases.  Steps, change
    records and keywords are appended as plain lines inside their block, so line-
    level retrieval can hit them.  No content is summarized away.
    """
    s = deepcopy(state)
    if s['roots']:
        raise ValueError('project after a forced review: every event must be placed')
    steps_by_event = {e: st['label'] for st in s['steps'].values() for e in st['event_ids']}
    events, phases, episodes = {}, {}, {}
    order = sorted(s['phases'].values(), key=lambda p: (p['start'], int(p['id'][2:])))
    for index, phase in enumerate(order, 1):
        blocks = []
        frame_ids = []
        for eid in phase['event_ids']:
            blocks.extend(_event_block(s['events'][eid], steps_by_event))
            frame_ids.extend(s['events'][eid]['frame_ids'])
        merged = _merge_same_window(blocks)
        caption = render_blocks(merged)
        frame_ids = frame_order(s['frames'], frame_ids)
        eid = f'E{index}'
        events[eid] = {'id': eid, 'caption': caption, 'frame_ids': frame_ids,
                       'start': s['frames'][frame_ids[0]]['time'], 'end': s['frames'][frame_ids[-1]]['time']}
        node = {'id': phase['id'], 'title': _cut(phase['title'], 32),
                'summary': _cut(phase['summary'] or _fallback_summary(s, phase), 80),
                'status': phase['status'], 'start': events[eid]['start'], 'end': events[eid]['end'],
                'event_ids': [eid]}
        if phase['status'] == 'closed':
            node.update(closed_at=max(phase['closed_at'], node['end']), close_reason='next_phase',
                        close_trigger_id=phase['id'])
        phases[phase['id']] = node
    for q in s['episodes'].values():
        members = sorted(q['phase_ids'], key=lambda p: (phases[p]['start'], int(p[2:])))
        node = {'id': q['id'], 'title': _cut(q['title'], 32), 'summary': _cut(q['summary'] or q['title'], 120),
                'status': q['status'], 'start': min(phases[p]['start'] for p in members),
                'end': max(phases[p]['end'] for p in members), 'phase_ids': members}
        if q['status'] == 'closed':
            node.update(closed_at=max(q['closed_at'], node['end']), close_reason='next_episode',
                        close_trigger_id=q['id'])
        episodes[q['id']] = node
    store = {'frames': s['frames'], 'events': events, 'phases': phases, 'episodes': episodes,
             'roots': [], 'completed_phase_ids': [p for p, n in phases.items() if n['status'] == 'closed'],
             'observed_until': s['observed_until'], 'next_merge_id': 1,
             'next_event_id': len(events) + 1}
    return {'schema': V11_SCHEMA, 'schema_version': 11, 'video_id': video_id,
            'observed_until': s['observed_until'], 'store': store, 'status': 'complete',
            'manager_state': {'last_review_at': s['observed_until']},
            'working_buffer_event_ids': [e for e, n in events.items()
                                         if phases[next(p for p in phases if e in phases[p]['event_ids'])]['status'] == 'open'],
            'provenance': {'kind': 'v12_lossless_projection', 'source_schema': SCHEMA,
                           **(provenance or {})}}


def _merge_same_window(blocks):
    """Adjacent events may share a sampled window only if identical; keep order, merge text."""
    out = []
    for bounds, lines in blocks:
        if out and out[-1][0] == bounds:
            out[-1][1].extend(lines)
        elif out and bounds[0] < out[-1][0][1]:
            raise ValueError('member observation windows overlap out of order')
        else:
            out.append((bounds, list(lines)))
    return out


def _cut(value, limit):
    value = ' '.join(str(value).split())
    return value if len(value) <= limit else value[:limit - 1].rstrip() + '…'
