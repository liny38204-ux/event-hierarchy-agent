import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from event_memory.short_window import (CONTACT_SHEET_POLICY, FRAME_REVIEW_POLICY,
                                       ROW_CONTACT_SHEET_POLICY, answer_from_window,
                                       recent_frames, recent_video_frames)


class ShortWindowTests(unittest.TestCase):
    def test_row_sheet_keeps_four_chronological_frames_and_uses_one_visual_input(self):
        from PIL import Image
        from hashlib import sha256

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames = []
            for i, color in enumerate(('red', 'green', 'blue', 'yellow'), 1):
                path = root / f'{i}.png'
                Image.new('RGB', (80, 40), color).save(path)
                frames.append({'id': f'f{i:06d}', 'time': i / 2, 'sample_time': i / 2,
                               'path': str(path), 'sha256': sha256(path.read_bytes()).hexdigest()})
            backend = Mock()
            backend.generate.return_value = '{"answer":"B","frame_ids":["f000004"]}'
            path = root / 'row.png'
            result = answer_from_window('Q', ['a', 'b'], frames, backend, stage='row',
                                        policy=ROW_CONTACT_SHEET_POLICY, sheet_path=path)
            sheet = result['window_attempt']['contact_sheet']
            self.assertEqual((sheet['rows'], sheet['columns']), (1, 4))
            self.assertEqual(sheet['frame_ids'], [frame['id'] for frame in frames])
            self.assertEqual(result['route'], 'short_window')
            self.assertEqual(len(backend.generate.call_args.kwargs['evidence']), 1)
            self.assertEqual(backend.generate.call_args.kwargs['evidence'][0]['kind'], 'contact_sheet')
            with Image.open(path) as image:
                self.assertGreater(image.width, image.height)

    def test_sheet_positions_and_original_frame_citations(self):
        from PIL import Image
        from hashlib import sha256

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames = []
            colors = ['red', 'green', 'blue', 'yellow', 'white']
            for i, color in enumerate(colors):
                path = root / f'{i}.png'
                Image.new('RGB', (220, 64), color).save(path)
                frames.append({'id': f'f{i+1:06d}', 'time': i / 2, 'sample_time': i / 2,
                               'path': str(path), 'sha256': sha256(path.read_bytes()).hexdigest()})
            backend = Mock()
            # A citation to the composite image itself must trigger review on original pixels.
            backend.generate.side_effect = ['{"answer":"B","frame_ids":["contact_sheet"]}',
                                            '{"answer":"A","frame_ids":["f000001"]}']
            path = root / 'sheets' / 'case.png'
            gate = answer_from_window('Q', ['a','b'], frames, backend, stage='q1',
                                      policy=CONTACT_SHEET_POLICY, sheet_path=path)
            self.assertEqual(gate['route'], 'short_window_frame')
            sheet = gate['window_attempt']['contact_sheet']
            self.assertEqual(sheet['frame_ids'], [f['id'] for f in frames])
            self.assertEqual(sheet['sha256'], sha256(path.read_bytes()).hexdigest())
            self.assertEqual(len(backend.generate.call_args_list[0].kwargs['evidence']), 1)
            self.assertEqual(backend.generate.call_args_list[1].kwargs['evidence'], frames[:1])
            with Image.open(path) as image:
                from PIL import ImageColor
                for i, color in enumerate(colors):
                    self.assertEqual(image.getpixel((i % 3 * 223 + 110, i // 3 * 97 + 60)), ImageColor.getrgb(color))
                self.assertEqual(image.getpixel((556, 157)), ImageColor.getrgb('#24282d'))
            backend.reset_mock()
            backend.generate.side_effect = ['{"answer":"B","frame_ids":["f000004"]}']
            direct = answer_from_window('Q', ['a','b'], frames, backend, stage='q2',
                                        policy=CONTACT_SHEET_POLICY, sheet_path=path)
            self.assertEqual((direct['route'], direct['cited_frame_ids']), ('short_window', ['f000004']))
            self.assertEqual(backend.generate.call_count, 1)
            empty = answer_from_window('Q', ['a','b'], [], backend, stage='q3', policy=CONTACT_SHEET_POLICY)
            self.assertEqual((empty['route'], empty['frame_attempts']), ('history', []))
            self.assertEqual(backend.generate.call_count, 1)

    def test_two_fps_reads_source_pts_and_only_the_latest_six_distinct_frames(self):
        try:
            import av
        except ImportError:
            self.skipTest('requires video decoding with PyAV')
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root/'source.mp4'
            with av.open(str(video), 'w') as output:
                # Fractional PTS between the half-second targets test floor selection.
                stream = output.add_stream('mpeg4', rate=3)
                stream.width = stream.height = 32
                stream.pix_fmt = 'yuv420p'
                for i in range(25):
                    image = Image.new('RGB', (32, 32), (i * 8, 0, 0))
                    for packet in stream.encode(av.VideoFrame.from_image(image)):
                        output.mux(packet)
                for packet in stream.encode():
                    output.mux(packet)

            early = recent_video_frames(video, 2.0, root/'early')
            self.assertEqual([f['sample_time'] for f in early], [0, 0.5, 1, 1.5, 2])
            self.assertEqual(len({f['sha256'] for f in early}), 5)
            self.assertAlmostEqual(early[1]['time'], 1 / 3)
            self.assertAlmostEqual(early[3]['time'], 4 / 3)
            tail = recent_video_frames(video, 6.3, root/'tail')
            self.assertEqual([f['sample_time'] for f in tail], [3.5, 4, 4.5, 5, 5.5, 6])
            self.assertEqual(len(list((root/'tail').glob('*.jpg'))), 6)
            self.assertTrue(all(3.3 < f['time'] <= f['sample_time'] <= 6.3 for f in tail))
            self.assertEqual(len(recent_video_frames(video, 0, root/'start')), 1)
            tail_four = recent_video_frames(video, 6.3, root/'tail-four', maximum_frames=4)
            self.assertEqual([f['sample_time'] for f in tail_four], [4.5, 5, 5.5, 6])
            held = recent_video_frames(video, 15, root/'past-end')
            self.assertEqual([f['sample_time'] for f in held], [5.5, 6, 6.5, 7, 7.5, 8])
            # The six most recent samples persist even when the oldest source PTS
            # falls just outside a three-second wall-clock range at fractional t.
            fractional = recent_video_frames(video, 6.49, root/'fractional')
            self.assertEqual(len(fractional), 6)
            self.assertLess(fractional[0]['time'], 6.49 - 3)


    def test_uses_actual_pts_and_includes_filtered_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'sampled_frames'
            root.mkdir()
            times = [0, 1.25, 1.99, 3.0, 4.0, 5.2, 5.26]
            decisions = []
            for i, t in enumerate(times, 1):
                fid = f'f{i:06d}'
                (root / (fid + '.jpg')).write_bytes(bytes([i]))
                decisions.append({'frame_id': fid, 'time': t, 'decision': 'drop_blur'})
            (root.parent / 'filter_audit.json').write_text(json.dumps({'decisions': decisions}))
            self.assertEqual([f['time'] for f in recent_frames(root, 5.25)], [1.99, 3.0, 4.0, 5.2])
            self.assertEqual([f['time'] for f in recent_frames(root, 2)], [0, 1.25, 1.99])
            self.assertEqual(recent_frames(root, 12), [])

    def test_frame_review_stops_on_first_single_frame_with_its_own_evidence(self):
        frames = [{'id': f'f{i:06d}', 'time': i/2, 'sample_time': i/2, 'sha256': '1'*64}
                  for i in range(1, 4)]
        decline = '{"answer":null,"frame_ids":[]}'
        answer = '{"answer":"B","frame_ids":["f000002"]}'
        for responses, route, ids in [
            ([answer], 'short_window', []),
            ([decline, answer, answer], 'short_window_frame', [1, 2]),
            ([decline, decline, 'bad output', decline], 'history', [1, 2, 3]),
        ]:
            backend = Mock()
            backend.generate.side_effect = responses
            check = answer_from_window('Q', ['a','b'], frames, backend, stage='epm-window:1',
                                       policy=FRAME_REVIEW_POLICY)
            self.assertEqual(check['route'], route)
            self.assertEqual(check['prediction'], 'B' if route != 'history' else None)
            evidence = [call.kwargs['evidence'] for call in backend.generate.call_args_list]
            self.assertEqual(evidence, [frames] + [[frames[i-1]] for i in ids])
            if route == 'short_window_frame':
                self.assertEqual(check['frame_attempts'][0]['status'], 'invalid_gate')
                self.assertEqual(check['cited_frame_ids'], ['f000002'])
        backend = Mock()
        check = answer_from_window('Q', ['a','b'], [], backend, stage='empty', policy=FRAME_REVIEW_POLICY)
        self.assertEqual((check['route'], check['frame_attempts']), ('history', []))
        backend.generate.assert_not_called()

    def test_gate_accepts_only_a_legal_option_and_an_actual_visual_reference(self):
        frame = {'id':'f000003', 'time':2.0, 'path':'x', 'sha256':'0'*64}
        model = Mock()
        for raw, expected_status, expected_answer in [
            ('{"answer":"E","frame_ids":["f000003"]}', 'answered', 'E'),
            ('{"answer":null,"frame_ids":[]}', 'needs_history', None),
            ('{"answer":"A","frame_ids":["f000001"]}', 'invalid_gate', None),
            ('{"answer":"F","frame_ids":["f000003"]}', 'invalid_gate', None),
            ('A', 'invalid_gate', None),
        ]:
            model.generate.return_value = raw
            result = answer_from_window('Which one?', list('abcde'), [frame], model, stage='epm-window:9')
            self.assertEqual((result['status'], result['prediction']), (expected_status, expected_answer))
            self.assertEqual(model.generate.call_args.kwargs['evidence'], [frame])
        model.reset_mock()
        self.assertEqual(answer_from_window('Which?', ['a','b'], [], model, stage='none')['status'], 'no_frames')
        model.generate.assert_not_called()
