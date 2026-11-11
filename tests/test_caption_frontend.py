"""Causal boundary and all-retained-input coverage tests; no model quality claims."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from event_memory.caption_frontend import CaptionMemory, FilterConfig, FrameFilter, OnlineChangeDetector, parse_caption_response
from event_memory.caption_runtime import QwenCaptionWriter, sampled_frames

spec = importlib.util.spec_from_file_location('caption_builder', ROOT/'scripts/build_caption_memory.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def frame(i, t=None):
    return {'id': f'f{i}', 'time': float(i if t is None else t), 'path': 'fixture.jpg'}


class Writer:
    def __init__(self, capacity=5, fail=False):
        self.capacity, self.fail, self.calls = capacity, fail, []

    def fits(self, system, text, evidence):
        return len(evidence) <= self.capacity

    def generate(self, system, text, evidence, *, stage):
        request = json.loads(text)
        self.calls.append({'request': request, 'evidence': evidence, 'stage': stage})
        return 'invalid JSON' if self.fail else json.dumps({'caption': 'Observed '+','.join(request['new_frame_ids'])})


class Encoder:
    def __init__(self, fail=False):
        self.fail = fail

    def encode(self, pixels):
        if self.fail:
            raise RuntimeError('fixture encoder failure')
        return [1, 0]


class CaptionFrontendTests(unittest.TestCase):
    def test_real_writer_json_fence_is_accepted_without_content_repair(self):
        body = '{"caption":"A white double-decker bus travels along a city street."}'
        for raw in [body, '```json\n'+body+'\n```', '```\n'+body+'\n```']:
            self.assertEqual(parse_caption_response(raw), json.loads(body)['caption'])
        for raw in ['Here is the caption:\n'+body, '```json\n'+body, body+' trailing',
                    '```json\n'+body+'\n```\n'+body, '{"caption":""}',
                    '{"caption":"observed", "state":"extra"}', '["caption"]']:
            with self.subTest(raw=raw), self.assertRaises((ValueError, json.JSONDecodeError)):
                parse_caption_response(raw)

    def test_long_segment_has_no_duration_cap_or_four_frame_quota(self):
        writer = Writer(capacity=100)
        memory = CaptionMemory(writer)
        for i in range(30):
            memory.push(frame(i, i*100), [1, 0])
        data = memory.finish(3000)
        self.assertEqual(len(data['segments']), 1)
        self.assertEqual(len(writer.calls[0]['evidence']), 30)
        self.assertEqual(data['coverage']['captioned_frames'], 30)
        self.assertTrue(data['coverage']['all_retained_frames_read'])

    def test_budget_batches_preserve_every_frame_and_candidate_identity(self):
        writer = Writer(capacity=4)
        memory = CaptionMemory(writer)
        for i in range(17):
            memory.push(frame(i), [1, 0])
        data = memory.finish(17)
        self.assertEqual(len(data['segments']), 1)
        self.assertGreater(len(writer.calls), 1)
        new_ids = [fid for call in writer.calls for fid in call['request']['new_frame_ids']]
        self.assertEqual(new_ids, [f'f{i}' for i in range(17)])
        for call in writer.calls:
            self.assertLessEqual(len(call['evidence']), 4)
            self.assertEqual(call['request']['candidate_id'], 'C0001')
        for part in data['segments'][0]['caption_parts']:
            self.assertLessEqual(max(data['frames'][fid]['time'] for fid in part['input_frame_ids']), part['written_at'])

    def test_content_boundary_and_caption_inputs_are_causal(self):
        writer = Writer(100)
        memory = CaptionMemory(writer)
        for i in range(6):
            memory.push(frame(i), [1, 0] if i < 3 else [0, 1])
        data = memory.finish(6)
        self.assertEqual([s['frame_ids'] for s in data['segments']], [['f0','f1','f2'],['f3','f4','f5']])
        self.assertEqual([f['id'] for f in writer.calls[0]['evidence']], ['f0','f1','f2'])
        self.assertEqual(data['segments'][0]['closed_at'], 3)
        self.assertEqual(data['segments'][-1]['status'], 'input_ended')

    def test_future_suffix_cannot_change_past_detector_outputs(self):
        vectors = [[1,0]]*7 + [[0,1]]*4
        def decisions(seq):
            detector = OnlineChangeDetector()
            return [detector.observe(x) for x in seq]
        self.assertEqual(decisions(vectors)[:8], decisions(vectors[:8]))
        with self.assertRaises(ValueError): OnlineChangeDetector().observe([float('nan'),0])
        with self.assertRaises(ValueError): OnlineChangeDetector().observe([0,0])

    def test_static_run_endpoints_and_reappearing_scene_survive(self):
        filtering = FrameFilter()
        a, b = np.zeros((16,16,3), dtype=np.uint8), np.full((16,16,3), 80, dtype=np.uint8)
        kept = []
        for i, pixels in enumerate([a,a,a,b,a]):
            kept.extend(filtering.push(frame(i), pixels))
        kept.extend(filtering.flush())
        self.assertEqual([r['id'] for r,_ in kept], ['f0','f2','f3','f4'])
        self.assertEqual(filtering.audit[1]['decision'], 'drop_duplicate_interior')
        self.assertEqual(filtering.audit[2]['decision'], 'keep_duplicate_endpoint')

    def test_small_local_change_is_not_lost_to_whole_image_similarity(self):
        filtering = FrameFilter()
        a = np.zeros((128,128,3), dtype=np.uint8)
        b = a.copy(); b[60,60] = 255
        filtering.push(frame(0), a)
        result = filtering.push(frame(1), b)
        self.assertEqual([r['id'] for r,_ in result], ['f1'])
        self.assertEqual(filtering.audit[-1]['decision'], 'keep_change')

    def test_query_checkpoint_does_not_end_candidate_or_recount_old_frames(self):
        writer = Writer(10)
        memory = CaptionMemory(writer)
        memory.push(frame(0), [1,0]); memory.checkpoint(1)
        self.assertEqual(memory.snapshot()['segments'][0]['status'], 'open')
        memory.push(frame(2), [1,0]); data = memory.finish(3)
        self.assertEqual(len(data['segments']), 1)
        self.assertEqual(data['coverage']['captioned_frames'], 2)
        self.assertEqual(writer.calls[1]['request']['context_frame_ids'], ['f0'])
        self.assertEqual(writer.calls[1]['request']['new_frame_ids'], ['f2'])
        with self.assertRaises(ValueError): memory.push(frame(4), [1,0])

    def test_one_frame_budget_omits_optional_bridge_without_losing_new_input(self):
        writer = Writer(1)
        memory = CaptionMemory(writer)
        for i in range(5): memory.push(frame(i), [1,0])
        data = memory.finish(6)
        self.assertTrue(data['coverage']['all_retained_frames_read'])
        self.assertEqual(len(writer.calls), 5)

    def test_full_pipeline_filtering_captioning_and_failure_coverage(self):
        samples = [(frame(i), np.full((16,16,3), i*20, dtype=np.uint8)) for i in range(10)]
        with tempfile.TemporaryDirectory() as tmp:
            data = builder.build_stream(iter(samples), Encoder(), Writer(3), Path(tmp)/'ok', frontier=10,
                                        filter_config=FilterConfig())
            self.assertEqual(data['status'], 'complete_caption_build')
            self.assertEqual(data['coverage']['captioned_frames'], 10)
            self.assertEqual(len(data['segments']), 1)
            for name, encoder, writer in [('writer', Encoder(), Writer(3,True)), ('encoder', Encoder(True), Writer())]:
                with self.assertRaises(Exception):
                    builder.build_stream(iter(samples), encoder, writer, Path(tmp)/name, frontier=10,
                                         filter_config=FilterConfig())
                failed = json.loads((Path(tmp)/name/'memory.json').read_text())
                self.assertEqual(failed['status'], 'failed')
                self.assertFalse(failed['coverage']['all_retained_frames_read'])
                self.assertTrue(failed['coverage']['unread_frame_ids'])

    def test_duplicate_tail_is_consumed_at_eof(self):
        samples = [(frame(i), np.zeros((16,16,3), dtype=np.uint8)) for i in range(10)]
        with tempfile.TemporaryDirectory() as tmp:
            data = builder.build_stream(iter(samples), Encoder(), Writer(), tmp, frontier=10,
                                        filter_config=FilterConfig())
        self.assertEqual(data['segments'][0]['frame_ids'], ['f0','f9'])
        self.assertEqual(data['coverage']['retained_frames'], 2)
        self.assertTrue(data['coverage']['all_retained_frames_read'])

    def test_pts_sampling_never_converts_future_pixels(self):
        converted = []
        class Frame:
            def __init__(self, pts): self.pts = pts
            def to_image(self):
                converted.append(self.pts)
                return Image.new('RGB', (8,8), 'red')
        class Stream:
            time_base = 0.1
            class codec_context: thread_count = 0
        class Container:
            class streams: video = [Stream()]
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def decode(self, stream): return iter([Frame(0),Frame(4),Frame(12),Frame(22)])
        class AV:
            @staticmethod
            def open(path): return Container()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'av': AV}):
            result = list(sampled_frames('unused.mp4', 1.5, tmp, 1.0))
        self.assertEqual([r['time'] for r,_ in result], [0.0,0.4])
        self.assertNotIn(22, converted)

    def test_token_budget_and_truncated_captions_fail_explicitly(self):
        class Processor:
            def apply_chat_template(self, *args, **kwargs): return 'fixture prompt'
            def __call__(self, **kwargs): return {'input_ids': np.zeros((1,100), dtype=int)}
        class Backend:
            processor, max_pixels = Processor(), 100352
            calls = []
            def generate(self, *args, **kwargs):
                self.calls.append({'finish_reason':'length'})
                return '{"caption":"cut off"}'
        writer = QwenCaptionWriter(Backend(), context_limit=1200, output_tokens=1024)
        self.assertTrue(writer.fits('system','text',[]))
        writer.context_limit = 1100
        self.assertFalse(writer.fits('system','text',[]))
        with self.assertRaisesRegex(ValueError, 'truncated'):
            writer.generate('system','text',[],stage='fixture')


if __name__ == '__main__':
    unittest.main()
