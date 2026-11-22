#!/usr/bin/env python3
"""Search a v10 memory with max(whole-caption cosine, per-line cosine)."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from event_memory.text_embedding import LocalSentenceEmbedder
from event_memory.text_retrieval import CaptionRetriever, render_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--memory', required=True, type=Path)
    parser.add_argument('--embedding-model', required=True, type=Path,
                        help='Existing local dense Sentence Transformers text model')
    parser.add_argument('--query', required=True)
    parser.add_argument('--query-time', type=float,
                        help='Must not precede the supplied memory snapshot frontier')
    parser.add_argument('--top-k', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--query-prompt', default='')
    parser.add_argument('--document-prompt', default='')
    args = parser.parse_args()
    memory = json.loads(args.memory.read_text())
    if args.top_k < 1 or not args.query.strip():
        parser.error('top-k must be positive and query must be nonempty')
    embedder = LocalSentenceEmbedder(args.embedding_model, device=args.device,
                                    batch_size=args.batch_size,
                                    query_prompt=args.query_prompt,
                                    document_prompt=args.document_prompt)
    retriever = CaptionRetriever(memory, embedder)
    hits = retriever.search(args.query, top_k=args.top_k, query_time=args.query_time)
    print(json.dumps({'query': args.query, 'observed_until': retriever.observed_until,
                      'scoring': 'max(full_caption_cosine, max_line_cosine)',
                      'embedding': embedder.receipt, 'hits': hits,
                      'context': render_context(hits)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
