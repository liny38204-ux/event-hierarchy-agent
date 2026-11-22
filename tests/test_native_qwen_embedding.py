"""Resource backoff must preserve every full document, order and hard limits."""
import sys
import types
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from native_qwen_embedding import NativeQwenTextEmbedder

class SimulatedOOM(RuntimeError):
    pass

class EmbeddingBackoffTests(unittest.TestCase):
    def make_encoder(self, capacity):
        e = NativeQwenTextEmbedder.__new__(NativeQwenTextEmbedder)
        e.batch_size = 8
        e.device = 'cuda:1'
        e.limit = 100
        e.oom_retries = []
        e.tokenizer = lambda text, **kwargs: {'input_ids': list(text)}
        e.torch = types.SimpleNamespace(OutOfMemoryError=SimulatedOOM,
                                       cuda=types.SimpleNamespace(empty_cache=lambda: None))
        calls = []
        def encode(values):
            calls.append(list(values))
            if len(values) > capacity:
                raise SimulatedOOM('injected allocation failure')
            return [[len(x), ord(x[-1])] for x in values]
        e._encode_batch = encode
        return e, calls

    def test_oom_backoff_retries_same_full_documents_without_drop_or_duplicate(self):
        e, calls = self.make_encoder(2)
        texts = ['document ' + str(i) for i in range(13)]
        actual = e._encode(texts, 'prefix: ')
        expected = [[len('prefix: '+x), ord(x[-1])] for x in texts]
        self.assertEqual(actual, expected)
        self.assertEqual([r['retry_batch_size'] for r in e.oom_retries], [4, 2])
        self.assertEqual(calls[0], ['prefix: '+x for x in texts[:8]])

    def test_one_document_oom_is_reported_without_truncation_or_infinite_retry(self):
        e, calls = self.make_encoder(0)
        with self.assertRaises(SimulatedOOM):
            e._encode(['the complete caption'], '')
        self.assertEqual(calls, [['the complete caption']])

    def test_overflow_is_rejected_before_any_forward(self):
        e, calls = self.make_encoder(2)
        with self.assertRaises(ValueError):
            e._encode(['x' * 101], '')
        self.assertEqual(calls, [])

    def test_non_memory_errors_are_not_retried(self):
        e, _ = self.make_encoder(8)
        def invalid(values):
            raise ValueError('invalid input')
        e._encode_batch = invalid
        with self.assertRaises(ValueError):
            e._encode(['hello'], '')
        self.assertEqual(e.oom_retries, [])
