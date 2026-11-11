"""Image-filter behavior and downstream coverage, without model inference."""
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests.test_caption_frontend import Encoder, Writer, builder, frame
from event_memory.image_filter import VAMImageFilter, VAMImageFilterConfig


class VAMImageFilterTests(unittest.TestCase):
    def setUp(self):
        a = np.random.default_rng(27).integers(0, 180, (64, 64), dtype=np.uint8)
        self.a = np.repeat(a[..., None], 3, axis=2)

    def test_dropped_frames_do_not_move_reference_and_tail_is_not_rescued(self):
        filt = VAMImageFilter(VAMImageFilterConfig(0, 20, 0))
        kept = []
        for i, pixels in enumerate([self.a, self.a+10, self.a+20, self.a+20]):
            kept.extend(filt.push(frame(i), pixels))
        kept.extend(filt.flush())
        self.assertEqual([r['id'] for r, _ in kept], ['f0', 'f2'])
        self.assertEqual([d['reference_frame_id'] for d in filt.audit], [None,'f0','f0','f2'])
        self.assertEqual(filt.audit[2]['gray_mad'], 20)  # Strict '<', not '<='.

    def test_pixel_gate_short_circuits_ssim_even_for_a_local_change(self):
        filt = VAMImageFilter()
        filt.push(frame(0), self.a)
        b = self.a.copy(); b[0, 0] = 255
        with patch.object(filt, 'ssim', side_effect=AssertionError('SSIM must not run')):
            self.assertEqual(filt.push(frame(1), b), [])
        self.assertEqual(filt.audit[-1]['decision'], 'drop_pixel_diff')
        self.assertIsNone(filt.audit[-1]['ssim'])

    def test_ssim_strict_threshold_and_blur_rejection(self):
        filt = VAMImageFilter()
        self.assertEqual(filt.push(frame(0), np.zeros_like(self.a)), [])
        self.assertEqual(filt.audit[-1]['decision'], 'drop_blur')
        filt.push(frame(1), self.a)
        with patch.object(filt, 'ssim', return_value=0.92):
            self.assertEqual(len(filt.push(frame(2), self.a+25)), 1)
        with patch.object(filt, 'ssim', return_value=0.92001):
            self.assertEqual(filt.push(frame(3), self.a), [])
        self.assertEqual(filt.audit[-1]['decision'], 'drop_ssim')

    def test_prefix_is_independent_of_future_and_repeated_scene_can_return(self):
        seq = [self.a, self.a, 255-self.a, self.a]
        def run(images):
            filt = VAMImageFilter()
            kept = [r['id'] for i, p in enumerate(images) for r, _ in filt.push(frame(i), p)]
            return kept, filt.audit
        kept, audit = run(seq)
        self.assertEqual(kept, ['f0','f2','f3'])
        self.assertEqual(run(seq[:2])[1], audit[:2])

    def test_default_builder_reads_only_kept_frames_without_embedding_dedup(self):
        writer = Writer()
        samples = [(frame(i), p) for i, p in enumerate([self.a, self.a, 255-self.a])]
        with tempfile.TemporaryDirectory() as tmp:
            data = builder.build_stream(samples, Encoder(), writer, tmp, frontier=3)
        read = [fid for call in writer.calls for fid in call['request']['new_frame_ids']]
        # Encoder returns the SAME embedding: both image-distinct frames must survive.
        self.assertEqual(read, ['f0','f2'])
        self.assertTrue(data['coverage']['all_retained_frames_read'])
        self.assertEqual(data['image_filter_mode'], 'vam_image')
        self.assertFalse(data['embedding_deduplication'])

    def test_empty_visual_evidence_is_not_successful_memory(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, 'No retained frames'):
            builder.build_stream([(frame(0), np.zeros_like(self.a))], Encoder(), Writer(), tmp, frontier=1)


if __name__ == '__main__':
    unittest.main()
