"""CPU fixtures for the fresh-video EPM bridge; they do not score a model."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from event_memory.text_retrieval import CaptionRetriever
SPEC = importlib.util.spec_from_file_location('epm_prefix', ROOT / 'scripts' / 'epm_build_prefix.py')
PREFIX = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PREFIX)


class Backend:
    def __init__(self, fail_writer=False): self.calls, self.fail_writer = [], fail_writer
    def token_count(self, text): return len(text.split())
    def generate(self, system, text, evidence, *, stage, max_tokens):
        self.calls.append({'stage': stage, 'system': system, 'text': text, 'max_tokens': max_tokens,
                           'evidence': [dict(item) for item in evidence]})
        if stage.startswith('write:'):
            return json.dumps({'caption': '' if self.fail_writer else '货架上摆放着盒装玩具。'})
        if stage.startswith('manage:'):
            return json.dumps({'events': [{'op': 'new', 'id': 'E1',
                'observation_ids': ['S0001:O01'], 'summary': '观察玩具货架。',
                'continuation_reason': '', 'support_evidence_ids': [evidence[0]['id']]}], 'episodes': []})
        if stage.startswith('buffered-organize:'):
            return json.dumps({'groups': [{'event_ids': ['E1'], 'phase_id': 'PH1',
                'phase_title': '玩具货架观察', 'phase_summary': '观察货架上的玩具盒。',
                'episode_id': 'P1', 'episode_title': '商店商品浏览',
                'episode_summary': '在商店内浏览玩具商品。'}],
                'deferred_event_ids': [], 'refreshes': [], 'reason': '同一局部观察已清楚。'})
        if stage.startswith('rolling-organize:'):
            return json.dumps({'groups': [{'event_ids': ['E1'], 'phase_id': 'PH1',
                'phase_title': '玩具货架观察', 'phase_summary': '观察货架上的玩具盒。',
                'episode_id': 'P1', 'episode_title': '商店商品浏览',
                'episode_summary': '在商店内浏览玩具商品。',
                'caption': json.loads(text.split('\n修复错误：')[0])['pending_events'][0]['caption']}], 'reason': '同一局部观察已清楚。'})
        raise AssertionError(stage)


class Encoder:
    def encode(self, pixels): return [1.0, 0.0]


class WriterBudget:
    def __init__(self, backend, *args, **kwargs): self.backend = backend
    def fits(self, *args, **kwargs): return True
    def generate(self, system, text, evidence, *, stage):
        return self.backend.generate(system, text, evidence, stage=stage, max_tokens=1536)


class Filter:
    def __init__(self, keep=True): self.keep, self.audit = keep, []
    def push(self, record, pixels):
        self.audit.append({'frame_id': record['id'], 'time': record['time'],
                           'decision': 'keep_fixture' if self.keep else 'drop_fixture'})
        return [(record, pixels)] if self.keep else []
    def flush(self): return []


class EpmPrefixBridgeTests(unittest.TestCase):
    def setUp(self):
        self.original = (PREFIX.sampled_frames, PREFIX.make_image_filter, PREFIX.QwenCaptionWriter,
                         PREFIX.candidate_batches)
        PREFIX.make_image_filter = lambda config: Filter()
        PREFIX.QwenCaptionWriter = WriterBudget
    def tearDown(self):
        (PREFIX.sampled_frames, PREFIX.make_image_filter, PREFIX.QwenCaptionWriter,
         PREFIX.candidate_batches) = self.original
    @staticmethod
    def sampler(_video, frontier, destination, fps):
        assert fps == 1.0
        destination.mkdir(parents=True, exist_ok=True)
        for number, colour in ((1, 10), (2, 20)):
            path = destination / f'f{number:06d}.jpg'
            Image.new('RGB', (16, 16), (colour, 0, 0)).save(path)
            record = {'id': path.stem, 'time': float(number), 'path': str(path),
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            yield record, np.asarray(Image.open(path).convert('RGB')).copy()
    def test_bridge_creates_complete_v10_memory_with_hardlinked_frames(self):
        PREFIX.sampled_frames = self.sampler
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); video = root / 'source.mp4'; video.write_bytes(b'fixture video')
            result = PREFIX.build_prefix(video, 2.0, root / 'output', 'ovo-1', Backend(), Encoder(), memory_mode='buffered')
            memory = json.loads(Path(result['memory_path']).read_text())
            self.assertEqual((memory['schema'], memory['schema_version']), ('buffered_phase_memory_v10', 10))
            self.assertEqual(memory['observed_until'], 2.0)
            self.assertEqual(result['metrics']['v10_pending_events'], 0)
            status = json.loads(Path(result['status_path']).read_text())
            self.assertEqual(status['status'], 'complete')
            self.assertIn('no questions/options/labels/answers', status['input_contract'])
            source = root / 'output' / 'source_frames' / 'ovo-1' / 'frames' / 'f000001.jpg'
            target = root / 'output' / 'v10' / 'ovo-1' / 'frames' / 'f000001.jpg'
            self.assertEqual(source.stat().st_ino, target.stat().st_ino)
    def test_window_and_history_build_reuse_the_same_sampled_prefix(self):
        from event_memory.short_window import recent_frames

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / 'video.mp4'
            video.write_bytes(b'fixture video')
            PREFIX.sampled_frames = self.sampler
            samples = PREFIX.prepare_prefix_samples(video, 2.0, root / 'window_input')
            frames = recent_frames(samples.root / 'sampled_frames', 2.0)
            self.assertEqual([f['time'] for f in frames], [1.0, 2.0])

            def unexpected_resample(*args):
                self.fail('fallback decoded the prefix twice')

            PREFIX.sampled_frames = unexpected_resample
            built = PREFIX.build_prefix(video, 2.0, root / 'build', 'ovo-1', Backend(), Encoder(),
                                        prepared_samples=samples)
            self.assertTrue(Path(built['memory_path']).is_file())
            self.assertEqual(recent_frames(root / 'build' / 'sampled_frames', 2.0), frames)

    def test_direct_caption_path_never_calls_local_manage_and_makes_immutable_new_events(self):
        PREFIX.sampled_frames = self.sampler
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); video = root / 'source.mp4'; video.write_bytes(b'fixture video'); backend = Backend()
            result = PREFIX.build_prefix(video, 2.0, root / 'output', 'ovo-1', backend, Encoder(), memory_mode='buffered')
            stages = [call['stage'] for call in backend.calls]
            self.assertTrue(any(stage.startswith('write:') for stage in stages))
            self.assertFalse(any(stage.startswith('manage:') for stage in stages))
            write = next(call for call in backend.calls if call['stage'].startswith('write:'))
            self.assertEqual(write['system'], PREFIX.caption_prompts.WRITER + PREFIX.LANGUAGE)
            self.assertEqual(write['max_tokens'], PREFIX.WRITER_OUTPUT_TOKENS)
            source = json.loads((root / 'output' / 'buffered_input.json').read_text())
            self.assertEqual(source['kind'], 'epm_direct_caption_immutable_e_v1')
            event = source['videos'][0]['steps'][0]['events'][0]
            self.assertEqual(event['op'], 'new')
            self.assertEqual(event['source_event_id'], 'S0001')
            self.assertEqual(event['frame_ids'], source['videos'][0]['steps'][0]['frame_ids'])
            self.assertEqual(event['caption'], '货架上摆放着盒装玩具。')
            self.assertEqual(result['metrics']['caption_policy'], 'one_candidate_caption_to_one_immutable_new_E; no_local_manage')
    def test_each_candidate_becomes_one_new_e_without_continuations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); frame_root = root / 'frames' / 'ovo-1' / 'frames'; frame_root.mkdir(parents=True)
            batches = []
            for number in (1, 2):
                path = frame_root / f'f{number}.jpg'; path.write_bytes(str(number).encode())
                batches.append({'start': float(number), 'end': float(number), 'closed_at': float(number),
                                'evidence': [{'id': f'f{number}', 'time': float(number), 'path': str(path)}]})
            class Writer:
                def __init__(self): self.stages = []
                def generate(self, _system, _text, _evidence, *, stage):
                    self.stages.append(stage); return json.dumps({'caption': '可见物体。'})
            writer = Writer()
            source, stats = PREFIX._captions_to_immutable_events(writer, batches, root / 'frames', 'ovo-1', 2.0)
            events = [step['events'][0] for step in source['videos'][0]['steps']]
            self.assertEqual([event['source_event_id'] for event in events], ['S0001', 'S0002'])
            self.assertEqual([event['op'] for event in events], ['new', 'new'])
            self.assertEqual(writer.stages, ['write:S0001:0', 'write:S0002:0'])
            self.assertEqual(stats['writer_segments'], 2)

    def test_writer_caption_uses_only_its_displayed_frame_time_envelope(self):
        evidence = [
            {'id': 'f_late', 'time': 63.125},
            {'id': 'f_early', 'time': 54.0},
            {'id': 'f_mid', 'time': 60.5},
        ]
        self.assertEqual(
            PREFIX._timed_writer_caption('先观察展示架。\n随后镜头靠近标签。', evidence),
            '<54s - 63.125s>\n先观察展示架。\n随后镜头靠近标签。')

    def test_writer_caption_rejects_model_supplied_time_header_even_inline(self):
        with self.assertRaisesRegex(ValueError, 'must not emit a time header'):
            PREFIX._timed_writer_caption('手放下一张牌。 <54s - 63s>',
                                         [{'id': 'f1', 'time': 54.0}])

    def test_two_invalid_time_headers_fail_at_writer_before_creating_an_event(self):
        class Writer:
            def generate(self, *_args, **_kwargs):
                return json.dumps({'caption':'捏住杯子。 <7s - 8s>'})
        batch = {'start':1., 'end':1., 'evidence':[{'id':'f1', 'time':1.}]}
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, 'writer failed.*after one repair'):
                PREFIX._captions_to_immutable_events(Writer(), [batch], folder, 'ovo-1', 1., timed=True)

    def test_timed_writer_repairs_inline_time_header_and_uses_timed_prompt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); frames = root / 'frames' / 'ovo-1' / 'frames'; frames.mkdir(parents=True)
            path = frames / 'f1.jpg'; path.write_bytes(b'1')
            batch = [{'start': 54., 'end': 63., 'closed_at': 63.,
                      'evidence': [{'id': 'f1', 'time': 54., 'path': str(path)},
                                   {'id': 'f2', 'time': 63., 'path': str(path)}]}]
            # Both evidence IDs need materialized source frames for the direct-E transaction.
            (frames / 'f2.jpg').write_bytes(b'2')
            class Writer:
                def __init__(self): self.calls = []
                def generate(self, system, text, _evidence, *, stage):
                    self.calls.append((system, text, stage))
                    return json.dumps({'caption': '拿起物品 <54s - 63s>' if len(self.calls) == 1 else '拿起物品。'})
            writer = Writer()
            source, stats = PREFIX._captions_to_immutable_events(
                writer, batch, root / 'frames', 'ovo-1', 63., timed=True)
            self.assertEqual([call[2] for call in writer.calls], ['write:S0001:0', 'write:S0001:1'])
            self.assertEqual(writer.calls[0][0], PREFIX.caption_prompts.TIMED_WRITER + PREFIX.LANGUAGE)
            self.assertIn('Validation error:', writer.calls[1][1])
            self.assertEqual(source['videos'][0]['steps'][0]['events'][0]['caption'], '<54s - 63s>\n拿起物品。')
            self.assertEqual(stats['writer_repaired_segments'], 1)
    def test_truncation_uses_a_fresh_bounded_repair_with_same_images(self):
        from unittest.mock import Mock
        images = [{'id': 'f1', 'time': 1., 'path': 'frame.jpg'}]
        segment = {'id': 'S0028', 'start': 1., 'end': 1., 'evidence': images}
        for fits in ([True], [False, True]):
            writer = Mock()
            writer.fits.side_effect = fits
            writer.generate.side_effect = [PREFIX.CaptionTruncated('length'), '{"caption":"整理扑克牌。"}']
            record, raw, attempts = PREFIX._write_caption(writer, 'system', segment, timed=True)
            self.assertEqual(attempts, 2)
            self.assertEqual(record['caption'], '<1s - 1s>\n整理扑克牌。')
            calls = writer.generate.call_args_list
            self.assertEqual([c.kwargs['stage'] for c in calls], ['write:S0028:0', 'write:S0028:1'])
            self.assertTrue(all(c.args[2] == images for c in calls))
            self.assertIn('concise JSON', calls[1].args[1])
            self.assertNotIn('Previous response', calls[1].args[1])
            self.assertEqual(writer.fits.call_args.args[1], calls[1].args[1])

    def test_writer_runtime_error_keeps_its_type_without_a_json_retry(self):
        from unittest.mock import Mock
        writer = Mock()
        writer.generate.side_effect = MemoryError('out of memory')
        with self.assertRaisesRegex(MemoryError, 'out of memory'):
            PREFIX._write_caption(writer, 'system', {'id':'S0001', 'evidence':[]}, timed=True)
        self.assertEqual(writer.generate.call_count, 1)

    def test_two_truncations_persist_the_attempt_causes(self):
        class Limited(Backend):
            def generate(self, *args, **kwargs):
                raise PREFIX.CaptionTruncated('private response')
        PREFIX.sampled_frames = self.sampler
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / 'source.mp4'
            video.write_bytes(b'fixture video')
            with self.assertRaises(PREFIX.CaptionWriteFailure):
                PREFIX.build_prefix(video, 2., root/'out', 'ovo-1', Limited(), Encoder())
            status = json.loads((root/'out/status.json').read_text())
            self.assertEqual(status['failure'], {'stage':'write', 'segment':'S0001',
                             'attempt_errors':['output_truncated', 'output_truncated']})
            self.assertNotIn('private response', str(status))

    def test_retained_and_prepared_empty_history_have_matching_audits(self):
        PREFIX.sampled_frames = self.sampler
        PREFIX.make_image_filter = lambda config: Filter(False)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root/'source.mp4'
            video.write_bytes(b'fixture video')
            prepared = PREFIX.prepare_prefix_samples(video, 2., root/'samples')
            for name, samples in [('fresh', None), ('prepared', prepared)]:
                with self.assertRaises(PREFIX.NoRetainedFrames):
                    PREFIX.build_prefix(video, 2., root/name, 'ovo-1', Backend(), Encoder(),
                                        prepared_samples=samples)
                audit = json.loads((root/name/'filter_audit.json').read_text())['decisions']
                self.assertEqual(audit, prepared.decisions)
                self.assertEqual(len(audit), 2)

    def test_future_evidence_is_rejected_before_writer_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); frames = root / 'frames' / 'ovo-1' / 'frames'; frames.mkdir(parents=True)
            path = frames / 'f1.jpg'; path.write_bytes(b'1')
            class Writer:
                def __init__(self): self.called = False
                def generate(self, *_args, **_kwargs): self.called = True; raise AssertionError('must not run')
            writer = Writer()
            batch = [{'start': 1., 'end': 1., 'closed_at': 1.,
                      'evidence': [{'id': 'f1', 'time': 2., 'path': str(path)}]}]
            with self.assertRaisesRegex(ValueError, 'fully causal before writer'):
                PREFIX._captions_to_immutable_events(writer, batch, root / 'frames', 'ovo-1', 2.)
            self.assertFalse(writer.called)
    def test_missing_retained_frame_is_rejected_before_v10(self):
        PREFIX.sampled_frames = self.sampler
        def omit_first(records, _encoder, frontier, **_kwargs):
            yield {'start': 2., 'end': frontier, 'closed_at': frontier, 'evidence': [records[1]]}
        PREFIX.candidate_batches = omit_first
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); video = root / 'source.mp4'; video.write_bytes(b'fixture video'); backend = Backend()
            with self.assertRaisesRegex(RuntimeError, 'frame coverage differs'):
                PREFIX.build_prefix(video, 2.0, root / 'output', 'ovo-1', backend, Encoder(), memory_mode='buffered')
            self.assertFalse(any(call['stage'].startswith('buffered-organize:') for call in backend.calls))
    def test_writer_failure_is_recorded_not_hidden(self):
        PREFIX.sampled_frames = self.sampler
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); video = root / 'source.mp4'; video.write_bytes(b'fixture video')
            with self.assertRaisesRegex(RuntimeError, 'caption writer failed'):
                PREFIX.build_prefix(video, 2.0, root / 'output', 'ovo-1', Backend(fail_writer=True), Encoder(), memory_mode='buffered')
            self.assertEqual(json.loads((root / 'output' / 'status.json').read_text())['status'], 'failed')
    def test_no_retained_frame_fails_before_model_calls(self):
        PREFIX.sampled_frames = self.sampler
        PREFIX.make_image_filter = lambda config: Filter(False)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); video = root / 'source.mp4'; video.write_bytes(b'fixture video'); backend = Backend()
            with self.assertRaisesRegex(ValueError, 'retained no frame'):
                PREFIX.build_prefix(video, 2.0, root / 'output', 'ovo-1', backend, Encoder(), memory_mode='buffered')
            self.assertEqual(backend.calls, [])
            self.assertEqual(json.loads((root / 'output' / 'status.json').read_text())['status'], 'failed')

    def test_default_mode_builds_rolling_v11_memory(self):
        PREFIX.sampled_frames = self.sampler
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); video = root / 'source.mp4'; video.write_bytes(b'fixture video'); backend = Backend()
            result = PREFIX.build_prefix(video, 2.0, root / 'output', 'ovo-1', backend, Encoder())
            memory = json.loads(Path(result['memory_path']).read_text())
            self.assertEqual((memory['schema'], memory['schema_version']), ('rolling_phase_memory_v11', 11))
            self.assertTrue(any(c['stage'].startswith('rolling-organize:') for c in backend.calls))
            self.assertFalse(any(c['stage'].startswith('buffered-organize:') for c in backend.calls))
            write = next(c for c in backend.calls if c['stage'].startswith('write:'))
            self.assertEqual(write['system'], PREFIX.caption_prompts.TIMED_WRITER + PREFIX.LANGUAGE)
            self.assertFalse((root / 'output' / 'buffered_input.json').exists())
            self.assertFalse((Path(result['memory_path']).parent / 'trace.jsonl').exists())
            self.assertEqual(memory['store']['events']['E1']['caption'],
                             '<1s - 2s>\n货架上摆放着盒装玩具。')
            self.assertEqual(result['metrics']['memory_mode'], 'rolling')
            self.assertEqual(result['metrics']['memory_pending_events'], 0)
            status = json.loads(Path(result['status_path']).read_text())
            self.assertEqual(status['memory_mode'], 'rolling')
            self.assertEqual(status['metrics']['memory_schema'], 'rolling_phase_memory_v11')
            class Embedder:
                def encode_documents(self, texts): return [[1.0] for _ in texts]
                def encode_query(self, text): return [1.0]
            hits = CaptionRetriever(memory, Embedder()).search('玩具', query_time=2.0)
            self.assertEqual(hits[0]['event_id'], 'E1')
            pages = memory['event_pages']['E1']
            self.assertEqual([p['frame_ids'] for p in pages], [['f000001', 'f000002']])
            self.assertTrue(Path(pages[0]['path']).is_file())


if __name__ == '__main__': unittest.main()
