"""Text-only retrieval over one immutable v10 or rolling v11 memory snapshot.

score(E, q) = max(cos(q, caption(E)), max_i cos(q, line_i(E))).
Parents supply context after ranking; no image, parent score or model reranker
is involved. Rebuild the index when captions or hierarchy membership change.
"""
from copy import deepcopy
import math
from numbers import Real
import re
from typing import Protocol, Sequence

from .buffered_phase import BufferedPhaseStore
from .direct_frame_memory import finite
from .timed_caption import header_range
from .rolling_phase import RollingPhaseStore, SCHEMA as ROLLING_SCHEMA


# Chinese sentences can be adjacent. In English, split on a period before
# whitespace, keeping decimals and frame filenames within their sentence.
_ACTION_SENTENCE_BOUNDARY = re.compile(r'(?<=[。！？!?])(?![。！？!?])|(?<=\.)(?=\s)')


class TextEmbedder(Protocol):
    """Both methods must use the same model and vector space."""

    def encode_documents(self, texts: list[str]) -> Sequence[Sequence[float]]: ...

    def encode_query(self, text: str) -> Sequence[float]: ...


def _unit(vector, dimension=None):
    try:
        values = list(vector)
    except TypeError as exc:
        raise ValueError('embedding must be a numeric vector') from exc
    if not values or any(isinstance(x, bool) or not isinstance(x, Real)
                         or not math.isfinite(x) for x in values):
        raise ValueError('embedding must contain finite numbers')
    if dimension is not None and len(values) != dimension:
        raise ValueError('embedding dimensions differ')
    norm = math.hypot(*values)
    if not norm or not math.isfinite(norm):
        raise ValueError('embedding norm must be finite and nonzero')
    return tuple(float(x) / norm for x in values)


def _phrase(node):
    return {key: node[key] for key in ('id', 'title', 'summary')}


class CaptionRetriever:
    """Rank unique E records and return complete captions plus PH/P context."""

    def __init__(self, memory: dict, embedder: TextEmbedder):
        if not isinstance(memory, dict):
            raise ValueError('memory must be a v10/v11 bundle or store snapshot')
        bundle = deepcopy(memory)
        if 'store' in bundle:
            schema = (bundle.get('schema'), bundle.get('schema_version'))
            validators = {
                ('buffered_phase_memory_v10', 10): BufferedPhaseStore.validate,
                (ROLLING_SCHEMA, 11): RollingPhaseStore.validate_reviewed,
            }
            try:
                validate = validators[schema]
            except KeyError as exc:
                raise ValueError('text retrieval requires a v10 or rolling v11 snapshot') from exc
            self._state = bundle['store']
            if finite(bundle['observed_until']) != finite(self._state['observed_until']):
                raise ValueError('bundle/store frontiers differ')
            validate(self._state)
            self._split_action_sentences = schema == (ROLLING_SCHEMA, 11)
        else:
            self._state = bundle
            # Store-only callers predate versioned bundles; preserve their v10
            # contract rather than guessing a rolling policy from state shape.
            BufferedPhaseStore.validate(self._state)
            self._split_action_sentences = False
        self.observed_until = finite(self._state['observed_until'])
        self._embedder = embedder
        self._phase = {eid: pid for pid, p in self._state['phases'].items()
                       for eid in p['event_ids']}
        self._episode = {pid: eid for eid, e in self._state['episodes'].items()
                         for pid in e['phase_ids']}
        self._lines = {}
        for eid, event in self._state['events'].items():
            lines = []
            window = None
            for index, raw in enumerate(event['caption'].splitlines()):
                line = raw.strip()
                if not line:
                    continue
                header = header_range(line)
                if header is not None:
                    window = header
                else:
                    fragments = ([part.strip() for part in _ACTION_SENTENCE_BOUNDARY.split(line)
                                  if part.strip()]
                                 if self._split_action_sentences else [line])
                    lines.extend((index, fragment, window) for fragment in fragments)
            if not lines:
                raise ValueError('an event caption requires at least one observation')
            self._lines[eid] = lines
        texts = list(dict.fromkeys(text for eid, e in self._state['events'].items()
                                  for text in [e['caption'],
                                               *(line for _, line, _ in self._lines[eid])]))
        self._vectors = {}
        self._dimension = None
        if texts:
            vectors = list(embedder.encode_documents(texts))
            if len(vectors) != len(texts):
                raise ValueError('embedder returned the wrong number of vectors')
            for text, vector in zip(texts, vectors):
                unit = _unit(vector, self._dimension)
                self._dimension = len(unit)
                self._vectors[text] = unit

    def search(self, query: str, top_k: int = 1, query_time=None) -> list[dict]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError('query must be nonempty text')
        if type(top_k) is not int or top_k < 1:
            raise ValueError('top_k must be a positive integer')
        if query_time is not None and finite(query_time) < self.observed_until:
            raise ValueError('query precedes this snapshot; load the causal memory version')
        if not self._vectors:
            return []
        q = _unit(self._embedder.encode_query(query), self._dimension)
        scores = {text: max(-1., min(1., math.fsum(a*b for a, b in zip(q, v))))
                  for text, v in self._vectors.items()}
        hits = []
        for eid, event in self._state['events'].items():
            caption = event['caption']
            caption_score = scores[caption]
            index, line, window = max(self._lines[eid], key=lambda item: scores[item[1]])
            line_score = scores[line]
            whole_wins = caption_score >= line_score
            score = max(caption_score, line_score)
            phase_id = self._phase.get(eid)
            episode_id = self._episode.get(phase_id)
            hits.append({
                'event_id': eid, 'caption': caption,
                'start': event['start'], 'end': event['end'],
                'score': score, 'caption_score': caption_score, 'line_score': line_score,
                'matched': {'kind': 'caption' if whole_wins else 'line',
                            'line_index': None if whole_wins else index,
                            'text': caption if whole_wins else line, 'score': score,
                            **({'time_range': list(window)} if window and not whole_wins else {})},
                # A whole-caption match has no honest sentence-level span of
                # its own.  Keep the best action line as a visual anchor so a
                # caller can select evidence without reopening discarded
                # source events.  It is deliberately separate from ``matched``:
                # scoring remains max(full caption, one action line).
                'best_action_line': {'line_index': index, 'text': line,
                                     'score': line_score,
                                     **({'time_range': list(window)} if window else {})},
                'phase': _phrase(self._state['phases'][phase_id]) if phase_id else None,
                'episode': _phrase(self._state['episodes'][episode_id]) if episode_id else None,
            })
        return sorted(hits, key=lambda h: (-h['score'], h['start'], h['event_id']))[:top_k]


def render_context(hits: list[dict]) -> str:
    """Reader-ready text, with complete event captions and short parent phrases."""
    blocks = []
    for hit in hits:
        rows = []
        for label, key in [('大事件', 'episode'), ('阶段', 'phase')]:
            node = hit[key]
            if node is not None:
                rows.append(f"{label} {node['id']}：{node['title']}。{node['summary']}")
        rows.append(f"事件 {hit['event_id']}（{hit['start']:g}–{hit['end']:g} 秒）：\n{hit['caption']}")
        blocks.append('\n'.join(rows))
    return '\n\n'.join(blocks)
