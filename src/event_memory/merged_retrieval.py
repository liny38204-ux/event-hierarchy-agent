"""Retrieve only finalized rolling-memory events and render their own frames.

The runtime state is the source of truth.  In particular this module never
loads writer traces, source captions, or a post-hoc subevent cache.  Text
ranking is delegated to :class:`CaptionRetriever`, which scores one final
event caption by ``max(full caption, best action line)``.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from .contact_sheet import write_contact_sheet
from .direct_frame_memory import finite, verify_evidence
from .rolling_phase import RollingPhaseStore, SCHEMA
from .timed_caption import header_range


MAX_PAGE_CELLS = 6


def _number(value, field):
    try:
        return finite(value)
    except ValueError as error:
        raise ValueError(f'{field} must be finite seconds') from error


def _window(value, *, event, field):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f'{field} must contain [start, end]')
    start, end = (_number(value[0], f'{field} start'), _number(value[1], f'{field} end'))
    if not event['start'] <= start <= end <= event['end']:
        raise ValueError(f'{field} lies outside the merged event')
    return start, end


def validate_merged_memory(memory, query_time):
    """Read a completed v11 state at a causal query time without modifying it.

    A caller gets no path to writer-level observations from this function.
    Every retained frame has to be owned by one final merged event, and every
    final event has to be the single child of a phase.
    """
    if not isinstance(memory, dict):
        raise ValueError('merged retrieval requires a rolling memory bundle')
    if (memory.get('schema'), memory.get('schema_version')) != (SCHEMA, 11):
        raise ValueError('merged retrieval requires rolling_phase_memory_v11')
    if memory.get('status') != 'complete':
        raise ValueError('merged retrieval requires a completed memory bundle')
    if 'store' not in memory or 'observed_until' not in memory:
        raise ValueError('merged memory bundle is incomplete')
    state = memory['store']
    if not isinstance(state, dict):
        raise ValueError('merged memory store must be a mapping')
    observed = _number(memory['observed_until'], 'bundle observed_until')
    if observed != _number(state.get('observed_until'), 'store observed_until'):
        raise ValueError('bundle/store frontiers differ')
    if _number(query_time, 'query time') < observed:
        raise ValueError('query precedes this causal memory snapshot')
    RollingPhaseStore.validate_reviewed(state)
    owned = [fid for event in state['events'].values() for fid in event['frame_ids']]
    if len(owned) != len(state['frames']) or set(owned) != set(state['frames']):
        raise ValueError('each retained frame must belong to exactly one merged event')
    return state


def _sample_chronological(frames, maximum=MAX_PAGE_CELLS):
    if not frames:
        raise ValueError('matched merged-event window has no retained frame')
    if len(frames) <= maximum:
        return frames
    last = len(frames) - 1
    positions = [index * last // (maximum - 1) for index in range(maximum)]
    if len(set(positions)) != maximum:
        raise AssertionError('chronological sampling duplicated a frame')
    return [frames[index] for index in positions]


def _anchor_window(hit, event):
    """Choose evidence time bounds without treating text scores as timestamps."""
    if not isinstance(hit, dict) or hit.get('event_id') != event['id']:
        raise ValueError('hit does not belong to this merged event')
    if hit.get('caption') != event['caption']:
        raise ValueError('hit caption differs from the retained merged caption')
    matched = hit.get('matched')
    if not isinstance(matched, dict):
        raise ValueError('hit has no match record')
    if matched.get('kind') == 'line' and 'time_range' in matched:
        return _window(matched['time_range'], event=event, field='matched line range'), 'matched_line'
    # When the whole caption wins, the best action line is a localization aid
    # only.  Ranking still came from the complete final event caption.
    anchor = hit.get('best_action_line')
    if isinstance(anchor, dict) and 'time_range' in anchor:
        return _window(anchor['time_range'], event=event, field='best action-line range'), 'best_action_line'
    return (event['start'], event['end']), 'event_envelope'


def _write_page(selected, folder):
    verify_evidence(selected)
    identity = '\0'.join(f"{f['id']}:{f['time']}:{f['sha256']}" for f in selected)
    key = hashlib.sha256(identity.encode()).hexdigest()[:16]
    destination = Path(folder).resolve() / f'page-{key}.png'
    rendered = write_contact_sheet(selected, destination, rows=2, columns=3)
    return {**rendered, 'image': destination.name, 'path': str(destination),
            'cells': [{'frame_id': f['id'], 'time_s': f['time']} for f in selected]}


def persist_event_pages(memory, frame_groups, folder):
    """Keep the writer's visual groups under their final E, with no source text or IDs."""
    state = validate_merged_memory(memory, memory['observed_until'])
    owners = {fid: eid for eid, event in state['events'].items() for fid in event['frame_ids']}
    pages = {eid: [] for eid in state['events']}
    for group in frame_groups:
        if not group or len({owners[fid] for fid in group}) != 1:
            raise ValueError('a visual group must belong to one merged event')
        frames = [state['frames'][fid] for fid in group]
        # Up to six pages per original observation, with no per-merged-E limit.
        selected = _sample_chronological(frames, 6 * MAX_PAGE_CELLS)
        for start in range(0, len(selected), MAX_PAGE_CELLS):
            pages[owners[group[0]]].append(_write_page(selected[start:start + MAX_PAGE_CELLS], folder))
    if any(not group for group in pages.values()):
        raise ValueError('each merged event needs at least one visual page')
    return pages


def select_merged_page(memory, hit, query_time, output_dir, *, requested_window=None,
                       requested_page_id=None):
    """Read an owned page nearest the matched observation window.

    Older rolling snapshots only stored frame references. They can render a page
    under output_dir without reading any old caption or reconstructing subevents.
    """
    state = validate_merged_memory(memory, query_time)
    if not isinstance(output_dir, (str, Path)):
        raise ValueError('output_dir must be a filesystem path')
    event_id = hit.get('event_id') if isinstance(hit, dict) else None
    if event_id not in state['events']:
        raise ValueError('hit does not belong to this merged memory')
    event = state['events'][event_id]
    bounds, policy = _anchor_window(hit, event)
    if requested_window is not None:
        bounds = _window(requested_window, event=event, field='reader requested window')
        if bounds not in {header_range(line) for line in event['caption'].splitlines()}:
            raise ValueError('requested window must be an observed header of the selected merged event')
        policy = 'reader_requested_window'
    if requested_page_id is not None and 'event_pages' not in memory:
        raise ValueError('requested page requires stored event_pages')
    if requested_page_id is not None and (not isinstance(requested_page_id, str) or not requested_page_id):
        raise ValueError('requested page ID must be a nonempty stored page hash')
    if 'event_pages' in memory:
        pages_by_event = memory['event_pages']
        if not isinstance(pages_by_event, dict) or event_id not in pages_by_event:
            raise ValueError('stored pages are missing for the selected merged event')
        pages = pages_by_event[event_id]
        if not isinstance(pages, list) or not pages:
            raise ValueError('selected merged event has no stored pages')
        owner = set(event['frame_ids'])
        for page in pages:
            if not isinstance(page, dict) or not isinstance(page.get('sha256'), str) or not page['sha256']:
                raise ValueError('stored page has no valid page hash')
            cells = page.get('cells')
            if (not isinstance(cells, list) or not 1 <= len(cells) <= MAX_PAGE_CELLS or
                    [c['frame_id'] for c in cells] != page['frame_ids'] or
                    any(c['frame_id'] not in owner or c['time_s'] != state['frames'][c['frame_id']]['time']
                        for c in cells)):
                raise ValueError('page cells differ from the retained event frames')
            if any(c['time_s'] > query_time for c in cells):
                raise ValueError('stored page contains evidence after the causal query time')
            if not isinstance(page.get('path'), str) or not page['path']:
                raise ValueError('stored page has no image path')
        if requested_page_id is not None:
            selected = [page for page in pages if page['sha256'] == requested_page_id]
            if len(selected) != 1:
                raise ValueError('requested page does not belong to the selected merged event')
            page = selected[0]
            if requested_window is None:
                bounds = (page['cells'][0]['time_s'], page['cells'][-1]['time_s'])
            if not any(bounds[0] <= cell['time_s'] <= bounds[1] for cell in page['cells']):
                raise ValueError('requested page lies outside the selected event time scope')
            policy = 'reader_requested_page'
        else:
            # Prefer coverage of the matched window, then proximity to its midpoint.
            center = sum(bounds) / 2
            def distance(page):
                times = [c['time_s'] for c in page['cells']]
                return (-sum(bounds[0] <= t <= bounds[1] for t in times),
                        min(abs(t - center) for t in times), times[0])
            page = min(pages, key=distance)
        if hashlib.sha256(Path(page['path']).read_bytes()).hexdigest() != page['sha256']:
            raise ValueError('stored merged-event page hash mismatch')
    else:
        candidates = [state['frames'][fid] for fid in event['frame_ids']
                      if bounds[0] <= state['frames'][fid]['time'] <= bounds[1]]
        page = _write_page(_sample_chronological(candidates), output_dir)
    return {**page, 'event_id': event_id, 'anchor_policy': policy,
            'event_time_range': [event['start'], event['end']], 'anchor_time_range': list(bounds)}
