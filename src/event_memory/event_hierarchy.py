"""Current event-only hierarchy; v2 state banks live in legacy_v2 only."""
from copy import deepcopy

from .hierarchy import HierarchyStore


class EventHierarchyStore(HierarchyStore):
    """Keep immutable evidence and stable event order, without state nodes."""

    def __init__(self):
        super().__init__()
        self._creation_order = {}

    def _ordered(self, identifiers):
        return sorted(identifiers, key=lambda key: (
            self._events[key]['start'], self._creation_order[key]))

    def apply(self, decision, now):
        self._require_exact(decision, {'events', 'episodes'}, 'event-only decision')
        if any(not isinstance(decision[key], list) for key in decision):
            raise ValueError('Event-only decision fields must be lists')
        with self._lock:
            allocated, order = [], dict(self._creation_order)
            for raw in decision['events']:
                if not isinstance(raw, dict):
                    raise ValueError('Event update must be an object')
                identifier = self._text(raw.get('id'), 'event id')
                if not identifier.startswith('E') or not identifier[1:].isdigit():
                    raise ValueError('Event IDs must use E<number>')
                allocated.extend(self._strings(raw.get('observation_ids'), 'observation_ids'))
                if identifier not in order:
                    order[identifier] = len(order)
            if len(allocated) != len(set(allocated)) or set(allocated) != self._pending:
                raise ValueError('Allocate every pending observation exactly once')
            for raw in decision['episodes']:
                self._require_exact(raw, {'id', 'summary', 'event_ids', 'status'}, 'episode')
                identifier = self._text(raw['id'], 'episode id')
                if not identifier.startswith('P') or not identifier[1:].isdigit():
                    raise ValueError('Episode IDs must use P<number>')
            # The base validates references, continuation evidence, immutable
            # membership, and parent refresh before committing any mutation.
            super().apply(decision, now)
            self._creation_order = order
            for parent in self._episodes.values():
                parent['event_ids'] = self._ordered(parent['event_ids'])
            audit = self._audit_log[-1]
            audit['applied_episode_order'] = {
                raw['id']: list(self._episodes[raw['id']]['event_ids'])
                for raw in decision['episodes']}
            audit['ordering_rule'] = 'first_support_time_then_stable_creation_order'

    def _public_event(self, identifier):
        node = deepcopy(self._events[identifier])
        node['creation_index'] = self._creation_order[identifier]
        node['evidence_start'], node['evidence_end'] = node['start'], node['end']
        node['time_semantics'] = 'supporting_frame_envelope_not_exact_boundaries'
        node['continuous_presence_verified'] = False
        node['evidence_spans'] = []
        for oid in node['observation_ids']:
            obs = self._observations[oid]
            node['evidence_spans'].append({
                'observation_id': oid, 'start': obs['start'], 'end': obs['end'],
                'evidence_ids': list(obs['evidence_ids']),
                'evidence_times': [self._evidence[fid]['time'] for fid in obs['evidence_ids']],
            })
        if 'uncertainty' in self.observation_fields:
            node['uncertainty'] = list(dict.fromkeys(
                text for oid in node['observation_ids']
                for text in self._observations[oid]['uncertainty']))
        return node

    def _public_episodes(self):
        result = deepcopy(self._episodes)
        for parent in result.values():
            parent['time_semantics'] = 'child_event_evidence_envelope'
            parent['order_semantics'] = 'evidence_chronology_not_strict_action_or_causal_order'
            parent['temporal_links'] = []
            for left, right in zip(parent['event_ids'], parent['event_ids'][1:]):
                a, b = self._events[left], self._events[right]
                if a['start'] == b['start']:
                    relation = 'same_first_evidence_order_unresolved'
                elif a['end'] < b['start']:
                    relation = 'evidence_before'
                elif a['end'] == b['start']:
                    relation = 'shared_evidence_boundary'
                else:
                    relation = 'overlapping_evidence_order_unresolved'
                parent['temporal_links'].append({'from': left, 'to': right, 'relation': relation})
        return result

    def context(self):
        with self._lock:
            events = {key: self._public_event(key) for key in self._ordered(self._events)}
            return deepcopy({
                'pending_observation_ids': sorted(self._pending),
                'pending_observations': {key: self._observations[key] for key in sorted(self._pending)},
                'events': events, 'episodes': self._public_episodes(),
                'roots': [key for key in events if key not in self._event_episode],
                'last_now': self._last_now,
            })

    def snapshot(self):
        with self._lock:
            context = self.context()
            return deepcopy({
                'segments': self._segments, 'observations': self._observations,
                'events': context['events'], 'episodes': context['episodes'],
                'event_episode': self._event_episode,
                'pending_observation_ids': sorted(self._pending), 'audit_log': self._audit_log,
                'context': context,
            })
