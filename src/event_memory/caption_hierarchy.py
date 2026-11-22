"""Caption-only v4: one immutable prose record per causal candidate segment.

The generic storage keys observations/observation_ids denote caption records in
this schema. They are provenance references, not dimension-wise model output.
"""
from .event_hierarchy import EventHierarchyStore
from .protocol import parse_manager_json


class CaptionHierarchyStore(EventHierarchyStore):
    observation_fields = {'id', 'caption', 'evidence_ids'}
    event_fields = {'op', 'id', 'observation_ids', 'summary',
                    'continuation_reason', 'support_evidence_ids'}

    @classmethod
    def _normalize_observation_content(cls, item):
        text = cls._text(item['caption'], 'caption')
        if len(text) > 4000:
            raise ValueError('Caption exceeds 4000 characters')

    @classmethod
    def _validate_segment(cls, segment):
        value, times = super()._validate_segment(segment)
        # Zero records only represents a failed write; the writer parser cannot
        # return an empty successful caption. Keep the frames for diagnosis.
        if len(value['observations']) > 1:
            raise ValueError('One candidate segment has at most one indivisible caption')
        for caption in value['observations']:
            if caption['evidence_ids'] != [frame['id'] for frame in value['evidence']]:
                raise ValueError('Caption provenance must retain all input frames in order')
        return value, times

    def _event_attributes(self, raw, observation_ids):
        # Semantic continuity is a model judgement, not an action-key equality.
        supports = self._strings(raw['support_evidence_ids'], 'event support evidence')
        for oid in observation_ids:
            if not set(supports) & set(self._observations[oid]['evidence_ids']):
                raise ValueError('Event needs support from each newly allocated caption')
        return {}

    def _validate_continuation_content(self, old, attributes):
        # The shared transaction still requires old + new visible evidence,
        # explicit justification, chronological append, and parent refresh.
        pass


def caption_segment(raw, segment):
    parsed, tail = parse_manager_json(raw)
    if not isinstance(parsed, dict) or set(parsed) != {'caption'}:
        raise ValueError('Caption writer must return exactly one caption string')
    record = {**segment, 'observations': [{
        'id': segment['id'] + ':O01', 'caption': parsed['caption'],
        'evidence_ids': [frame['id'] for frame in segment['evidence']],
    }]}
    CaptionHierarchyStore._validate_segment(record)
    return record, tail
