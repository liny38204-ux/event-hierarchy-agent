"""Official Qwen3 text-embedding last-token pooling without Sentence Transformers.

Reference: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B#transformers-usage
The retrieval formula and frozen prompts are unchanged.
"""
import gc
import json
from pathlib import Path


class NativeQwenTextEmbedder:
    def __init__(self, model_path, *, device='cuda:0', batch_size=8,
                 query_prompt='', document_prompt=''):
        import torch
        from transformers import AutoTokenizer, AutoModel
        path = Path(model_path)
        receipt = json.loads((path / 'verified_model.json').read_text())
        if receipt['repo_id'] != 'Qwen/Qwen3-Embedding-0.6B':
            raise ValueError('unexpected text embedding checkpoint')
        pooling = json.loads((path / '1_Pooling/config.json').read_text())
        if not pooling.get('pooling_mode_lasttoken'):
            raise ValueError('checkpoint does not use last-token pooling')
        self.tokenizer = AutoTokenizer.from_pretrained(path, padding_side='left', local_files_only=True)
        self.model = AutoModel.from_pretrained(path, dtype=torch.bfloat16,
                                               attn_implementation='sdpa', local_files_only=True).to(device)
        self.model.eval().requires_grad_(False)
        self.limit = int(self.model.config.max_position_embeddings)
        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.query_prompt = query_prompt
        self.document_prompt = document_prompt
        self.oom_retries = []
        self.receipt = {'type': 'native_transformers_qwen3_last_token_pooling',
                        'repo_id': receipt['repo_id'], 'revision': receipt['revision'],
                        'device': device, 'weight_dtype': 'bfloat16', 'output_dtype': 'float32',
                        'attention': 'sdpa', 'padding_side': 'left', 'max_seq_length': self.limit,
                        'query_prompt': query_prompt, 'document_prompt': document_prompt,
                        'overflow_policy': 'reject_without_truncation',
                        'use_cache': False,
                        'oom_policy': 'halve_batch_to_one_without_truncation',
                        'model_class': type(self.model).__name__}

    def _encode(self, texts, prompt):
        values = [prompt + text for text in texts]
        for i, text in enumerate(values):
            count = len(self.tokenizer(text, truncation=False)['input_ids'])
            if count > self.limit:
                raise ValueError(f'text {i} exceeds embedding context: {count} > {self.limit}')
        result = []
        start = 0
        batch_size = self.batch_size
        while start < len(values):
            end = min(start + batch_size, len(values))
            try:
                vectors = self._encode_batch(values[start:end])
            except self.torch.OutOfMemoryError:
                if end - start == 1:
                    raise
                reduced = max(1, (end - start) // 2)
                self.oom_retries.append({'start': start, 'failed_batch_size': end-start,
                                         'retry_batch_size': reduced})
                batch_size = reduced
                # The failed helper frame is released when this except ends.
            else:
                result.extend(vectors)
                start = end
                continue
            gc.collect()
            if str(self.device).startswith('cuda'):
                self.torch.cuda.empty_cache()
        return result

    def _encode_batch(self, values):
        with self.torch.inference_mode():
            batch = self.tokenizer(values, padding=True, truncation=False,
                                   return_tensors='pt').to(self.device)
            if not self.torch.all(batch['attention_mask'][:, -1] == 1):
                raise ValueError('last-token pooling requires nonpadding final tokens')
            # Embedding pools a complete sequence once. A generation KV cache
            # is never consumed and needlessly competes with the resident VLM.
            hidden = self.model(**batch, use_cache=False).last_hidden_state[:, -1].float()
            vectors = self.torch.nn.functional.normalize(hidden, p=2, dim=1)
            return vectors.cpu().tolist()

    def encode_documents(self, texts):
        return self._encode(texts, self.document_prompt)

    def encode_query(self, text):
        return self._encode([text], self.query_prompt)[0]
