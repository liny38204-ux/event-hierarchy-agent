"""Adapter contract tests; deliberately no real model/download/GPU."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from event_memory.text_embedding import LocalSentenceEmbedder


class FakeModel:
    max_seq_length = 5

    def __init__(self, path, **kwargs):
        self.options = kwargs
        self.calls = []

    def tokenizer(self, text, **kwargs):
        assert kwargs == {'truncation': False, 'add_special_tokens': True}
        return {'input_ids': list(range(len(text) + 1))}

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return SimpleNamespace(tolist=lambda: [[1., 0.] for _ in texts])


class TextEmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.patch = patch.dict(sys.modules, {
            'sentence_transformers': SimpleNamespace(SentenceTransformer=FakeModel)})
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_local_only_explicit_prompts_and_same_model(self):
        embedder = LocalSentenceEmbedder(self.tmp.name, query_prompt='Q', document_prompt='D')
        self.assertEqual(embedder.encode_documents(['one']), [[1., 0.]])
        self.assertEqual(embedder.encode_query('two'), [1., 0.])
        self.assertTrue(embedder.model.options['local_files_only'])
        self.assertFalse(embedder.model.options['trust_remote_code'])
        self.assertEqual(embedder.model.options['device'], 'cpu')
        self.assertEqual([c[1]['prompt'] for c in embedder.model.calls], ['D', 'Q'])

    def test_full_caption_and_prompt_overflow_fail_before_encode(self):
        embedder = LocalSentenceEmbedder(self.tmp.name)
        with self.assertRaisesRegex(ValueError, 'rather than truncate'):
            embedder.encode_documents(['very long caption'])
        self.assertEqual(embedder.model.calls, [])
        embedder.query_prompt = 'query: '
        with self.assertRaises(ValueError):
            embedder.encode_query('a')
        self.assertEqual(embedder.model.calls, [])

    def test_missing_model_does_not_become_a_download(self):
        with self.assertRaisesRegex(ValueError, 'existing local directory'):
            LocalSentenceEmbedder(Path(self.tmp.name) / 'missing')


if __name__ == '__main__':
    unittest.main()
