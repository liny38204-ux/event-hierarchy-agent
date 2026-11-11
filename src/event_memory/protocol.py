"""Inference-safe records, strict answer parsing, and fixed-denominator scoring."""
from __future__ import annotations

import json
import math
import re

VARIANTS = ('flat_caption', 'flat_fields', 'hierarchical_fields')


def parse_json(text: str) -> dict:
    value = text.strip()
    if value.startswith('```'):
        value = re.sub(r'^```(?:json)?\s*', '', value)
        value = re.sub(r'\s*```$', '', value)
    result = json.loads(value)
    if not isinstance(result, dict):
        raise ValueError('Expected JSON object')
    return result


def parse_manager_json(text: str) -> tuple[dict, str]:
    """Accept one leading manager object plus logged prose, never two decisions.

    This transport recovery is deliberately separate from answer parsing and
    writer parsing. The returned decision still goes through EventStore checks.
    """
    try:
        return parse_json(text), ''
    except json.JSONDecodeError:
        value = text.strip()
        result, end = json.JSONDecoder().raw_decode(value)
        suffix = value[end:].strip()
        if not isinstance(result, dict) or not suffix:
            raise ValueError('Expected one manager JSON object')
        if any(marker in suffix for marker in ('{', '}', '[', ']', '```')):
            raise ValueError('Ambiguous trailing structured manager output')
        return result, suffix


def writer_record(raw: str, segment: dict, kind: str) -> dict:
    """Identifiers, times and evidence come only from the observer, never the VLM."""
    parsed = parse_json(raw)
    summary = parsed.get('summary')
    if not isinstance(summary, str) or not summary.strip() or len(summary) > 1200:
        raise ValueError('Missing or oversized summary')
    result = {key: segment[key] for key in ('id', 'start', 'end', 'evidence')}
    result['summary'] = summary.strip()
    for key in ('entities', 'actions', 'state_changes'):
        value = parsed.get(key, []) if kind == 'fields' else []
        if not isinstance(value, list) or len(value) > 12 or not all(
                isinstance(x, str) and len(x) <= 240 for x in value):
            raise ValueError(f'Invalid {key}')
        result[key] = value
    return result


def answer_letter(text: str, count: int) -> str | None:
    # Only the final exact protocol tag or a bare letter is accepted.
    values = re.findall(r'<answer>\s*([A-Z])\s*</answer>', text, re.I)
    candidate = values[-1].upper() if values else text.strip().upper()
    return candidate if len(candidate) == 1 and 'A' <= candidate < chr(65+count) else None


def question_for_reader(item: dict) -> dict:
    return {key: item[key] for key in ('id','video_id','video_path','query_time_seconds','question','options')}


def validate_reader(manifest: dict) -> None:
    seen = set()
    allowed = {'id','video_id','video_path','query_time_seconds','question','options'}
    for q in manifest['questions']:
        if set(q) != allowed:
            raise ValueError(f'Unsafe or missing reader fields: {set(q) ^ allowed}')
        if q['id'] in seen:
            raise ValueError('Duplicate question ID')
        seen.add(q['id'])
        if not math.isfinite(q['query_time_seconds']) or q['query_time_seconds'] <= 0:
            raise ValueError('Invalid question frontier')
        if not isinstance(q['options'], list) or not 2 <= len(q['options']) <= 7:
            raise ValueError('Invalid options')


def score(reader: dict, labels: dict, predictions: list[dict]) -> dict:
    ids = {q['id'] for q in reader['questions']}
    if set(labels) != ids:
        raise ValueError('Scorer labels must match the entire frozen reader')
    by_key = {}
    for p in predictions:
        key = (p['variant'], p['id'])
        if key in by_key or p['id'] not in ids or p['variant'] not in VARIANTS:
            raise ValueError('Duplicate or unknown prediction')
        by_key[key] = p
    results = {}
    for variant in VARIANTS:
        rows = []
        for q in reader['questions']:
            p = by_key.get((variant, q['id']))
            pred = p.get('prediction') if p else None
            valid = p is not None and p.get('status') == 'ok' and pred is not None
            target = chr(65+labels[q['id']]['correct_option_index'])
            rows.append({'id':q['id'], 'video_id':q['video_id'], 'prediction':pred,
                         'target':target, 'correct':bool(valid and pred == target),
                         'status':p.get('status') if p else 'missing'})
        correct = sum(r['correct'] for r in rows)
        complete = sum(r['status'] in ('ok','invalid','error') for r in rows)
        results[variant] = {'correct':correct,'denominator':len(rows),
                            'accuracy':correct/len(rows) if rows else None,
                            'completed':complete,'coverage_complete':complete==len(rows),
                            'rows':rows}
    return {'evaluation':'NExT-GQA video-end development diagnostic',
            'official_streaming_benchmark':False,'results':results}
