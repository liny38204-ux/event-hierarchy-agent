"""Optional local Sentence Transformers adapter for text retrieval."""
from pathlib import Path


class LocalSentenceEmbedder:
    """Use a local dense text model, without silently truncating full captions.

    Set query/document prompts explicitly according to the chosen model card.
    Defaults are empty; no instruction or language translation is invented.
    """

    def __init__(self, model_path, *, device='cpu', batch_size=16,
                 query_prompt='', document_prompt=''):
        path = Path(model_path).expanduser().resolve()
        if not path.is_dir():
            raise ValueError('embedding model must be an existing local directory')
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        if not isinstance(query_prompt, str) or not isinstance(document_prompt, str):
            raise ValueError('embedding prompts must be strings')
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError('Install the optional retrieval dependencies: pip install -r requirements-retrieval.txt') from exc
        self.model = SentenceTransformer(str(path), device=device,
                                         local_files_only=True, trust_remote_code=False)
        self.batch_size = batch_size
        self.query_prompt = query_prompt
        self.document_prompt = document_prompt
        self.receipt = {'type': 'local_sentence_transformer', 'model_path': str(path),
                        'device': device, 'query_prompt': query_prompt,
                        'document_prompt': document_prompt,
                        'max_seq_length': self.model.max_seq_length,
                        'overflow_policy': 'reject_without_truncation'}

    def _encode(self, texts, prompt):
        # encode() normally truncates at max_seq_length. That would silently
        # turn the requested whole-caption embedding into a prefix embedding.
        limit = self.model.max_seq_length
        if not isinstance(limit, int) or limit < 1:
            raise ValueError('text encoder must expose a positive max_seq_length')
        for i, text in enumerate(texts):
            ids = self.model.tokenizer(prompt + text, truncation=False,
                                       add_special_tokens=True)['input_ids']
            if len(ids) > limit:
                raise ValueError(f'embedding text {i} has {len(ids)} tokens; model limit {limit}; '
                                 'choose a longer-context text model rather than truncate')
        return self.model.encode(texts, prompt=prompt, batch_size=self.batch_size,
                                 normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False).tolist()

    def encode_documents(self, texts):
        return self._encode(texts, self.document_prompt)

    def encode_query(self, text):
        return self._encode([text], self.query_prompt)[0]
