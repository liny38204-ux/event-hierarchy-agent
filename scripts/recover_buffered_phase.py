"""Recover an interrupted diagnostic using an exact, ordered model-call prefix."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from build_buffered_phase import build
from event_memory.backend_qwen38 import Qwen38Backend


RECEIPT_KEYS = ('repo_id', 'revision', 'verified_model_sha256', 'torch',
                'transformers', 'dtype', 'attention', 'patch_embed',
                'enable_thinking', 'do_sample', 'seed', 'max_pixels',
                'trainable_parameters', 'device_map')


class PrefixBackend:
    def __init__(self, source, count, model, output, factory=Qwen38Backend):
        records = [json.loads(x) for x in (source / 'calls.jsonl').read_text().splitlines()]
        if not 0 < count <= len(records):
            raise ValueError('invalid recovery prefix length')
        self.records = records[:count]
        if any(r.get('error') or 'raw' not in r for r in self.records):
            raise ValueError('cannot reuse failed generation')
        self.receipt = json.loads((source / 'config.json').read_text())['model_receipt']
        self.model, self.output, self.factory = model, output, factory
        self.source = source.name
        self.calls, self.live = [], None

    def generate(self, system, text, evidence, *, stage, max_tokens=2048):
        index = len(self.calls)
        for frame in evidence:
            if hashlib.sha256(Path(frame['path']).read_bytes()).hexdigest() != frame['sha256']:
                raise RuntimeError('recovery image hash mismatch')
        if index < len(self.records):
            old = self.records[index]
            expected = [{k: v for k, v in f.items() if k != 'path'} for f in old['evidence']]
            actual = [{k: v for k, v in f.items() if k != 'path'} for f in evidence]
            expected_tokens = 1536 if stage.startswith('buffered-organize:') else 2048
            if (old['system'], old['text'], old['stage'], expected, expected_tokens) != (system, text, stage, actual, max_tokens):
                raise RuntimeError('recovery prefix mismatch; refusing stale output')
            record = deepcopy(old)
            record.update(evidence=list(evidence), cached=True, cache_source=self.source,
                          cache_index=index, source_seconds=record['seconds'], seconds=0.)
        else:
            if self.live is None:
                self.live = self.factory(self.model, self.output / 'fresh_calls.jsonl')
                if any(self.live.receipt[k] != self.receipt[k] for k in RECEIPT_KEYS):
                    raise RuntimeError('recovery backbone/inference mismatch')
                (self.output / 'reloaded_model_receipt.json').write_text(json.dumps(self.live.receipt, indent=2) + '\n')
            before = len(self.live.calls)
            try:
                self.live.generate(system, text, evidence, stage=stage, max_tokens=max_tokens)
            finally:
                if len(self.live.calls) > before:
                    record = deepcopy(self.live.calls[-1])
                    record['cached'] = False
                    self._save(record)
            if record.get('error'):
                raise RuntimeError(record['error'])
            return record['raw']
        self._save(record)
        return record['raw']

    def _save(self, record):
        self.calls.append(record)
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / 'calls.jsonl').open('a') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    for name in ('input', 'frames-run', 'model', 'output', 'recovery-run'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--cache-prefix', type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('output exists')
    if json.loads(args.input.read_text()) != json.loads((args.recovery_run / 'input_manifest.json').read_text()):
        raise ValueError('recovery input mismatch')
    backend = PrefixBackend(args.recovery_run, args.cache_prefix, args.model, args.output)
    try:
        result = build(args.input, args.frames_run, args.output, backend)
        for entry in result['videos']:
            vid = entry['video_id']
            records = [r for r in backend.calls if any('/' + vid + '/' in f['path'] for f in r['evidence'])]
            stats = entry['statistics']
            stats.update(recorded_model_calls=len(records),
                         cache_hits=sum(r.get('cached', False) for r in records),
                         actual_model_calls=sum(not r.get('cached', False) for r in records))
            (args.output / vid / 'verification.json').write_text(json.dumps(stats, indent=2) + '\n')
        (args.output / 'index.json').write_text(json.dumps(result, indent=2) + '\n')
    finally:
        args.output.mkdir(parents=True, exist_ok=True)
        info = {'source': args.recovery_run.name, 'requested_prefix': args.cache_prefix,
                'cache_hits': sum(r.get('cached', False) for r in backend.calls),
                'fresh_calls': sum(not r.get('cached', False) for r in backend.calls),
                'no_caption_edits': True, 'no_training_or_qa': True}
        (args.output / 'recovery.json').write_text(json.dumps(info, indent=2) + '\n')


if __name__ == '__main__':
    main()
