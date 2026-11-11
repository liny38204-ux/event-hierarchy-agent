#!/usr/bin/env python3
"""Independent caption front end. Does not invoke v1/v2 hierarchy management."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from event_memory.caption_frontend import CaptionMemory, FilterConfig, OnlineChangeDetector
from event_memory.image_filter import VAMImageFilterConfig, image_filter_mode, make_image_filter
from event_memory.caption_runtime import LocalCLIPEncoder, QwenCaptionWriter, sampled_frames, sha256_file


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def build_stream(samples, encoder, writer, output, *, frontier, filter_config=VAMImageFilterConfig(),
                 fallback_distance=0.2, minimum_distance=0.05):
    """Injection-friendly real pipeline; failures keep partial coverage visible."""
    output = Path(output)
    filtering = make_image_filter(filter_config)
    memory = CaptionMemory(writer, OnlineChangeDetector(fallback_distance, minimum_distance))
    status, error = 'building', None

    def persist(_=None):
        data = memory.snapshot()
        expected = [d['frame_id'] for d in filtering.audit if d['decision'].startswith('keep_')]
        read = [fid for s in data['segments'] for part in s['caption_parts'] for fid in part['new_frame_ids']]
        data['coverage'] = {'retained_frames': len(expected), 'captioned_frames': len(read),
                            'unread_frame_ids': [fid for fid in expected if fid not in set(read)],
                            'all_retained_frames_read': bool(expected) and set(expected) == set(read)
                                and len(read) == len(set(read))}
        data.update(status=status, error=error, filter_config=asdict(filter_config),
                    image_filter_mode=image_filter_mode(filter_config), embedding_deduplication=False,
                    filter_decisions=filtering.audit,
                    sampling_note='Coverage refers to retained sampled frames, not all source-video frames')
        save(output / 'memory.json', data)
        return data

    memory.on_update = persist
    def consume(items, now):
        for record, pixels in items:
            embedding = encoder.encode(pixels)
            memory.push(record, embedding, observed_at=now)

    try:
        if not math.isfinite(frontier) or frontier <= 0:
            raise ValueError('Invalid frontier')
        for record, pixels in samples:
            if record['time'] > frontier:
                raise ValueError('Sample lies beyond permitted prefix')
            consume(filtering.push(record, pixels), float(record['time']))
        consume(filtering.flush(), frontier)
        result = memory.finish(frontier)
        if not result['frames']:
            raise ValueError('No retained frames: cannot report a complete caption memory')
        if not result['coverage']['all_retained_frames_read']:
            raise ValueError('Unread retained frames')
        status = 'complete_caption_build'
        return persist()
    except Exception as exc:
        status, error = 'failed', repr(exc)
        persist()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--frontier', type=float, required=True)
    parser.add_argument('--fps', type=float, required=True)
    parser.add_argument('--embedding-model', type=Path, required=True,
                        help='Local pretrained CLIP weights; embedding runs on CPU')
    parser.add_argument('--writer-model', type=Path, required=True,
                        help='Locally verified full Qwen3.8-27B directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image-filter', choices=('vam_image', 'conservative'), default='vam_image')
    parser.add_argument('--blur-min', type=float, default=None)
    parser.add_argument('--duplicate-mad-max', type=float, default=None)
    parser.add_argument('--duplicate-ssim-min', type=float, default=None)
    parser.add_argument('--duplicate-pixel-max', type=float, default=None,
                        help='Only valid for the legacy conservative filter')
    parser.add_argument('--boundary-fallback-distance', type=float, default=0.2)
    parser.add_argument('--boundary-minimum-distance', type=float, default=0.05)
    parser.add_argument('--writer-context-tokens', type=int, default=16384)
    args = parser.parse_args()
    if args.image_filter == 'vam_image' and args.duplicate_pixel_max is not None:
        parser.error('--duplicate-pixel-max only applies to --image-filter conservative')
    config_type = VAMImageFilterConfig if args.image_filter == 'vam_image' else FilterConfig
    overrides = {key: getattr(args, key) for key in asdict(config_type()) if getattr(args, key) is not None}
    cfg = config_type(**overrides)
    make_image_filter(cfg)  # Validate thresholds/dependencies before loading models.
    if args.output.exists():
        raise ValueError('Use a new output directory')
    if not args.video.is_file() or not all(math.isfinite(x) and x > 0 for x in (args.fps, args.frontier)):
        raise ValueError('Valid video, FPS and frontier are required')
    # The existing two-GPU backend checks identity; launch only on explicitly
    # allocated idle GPUs under the project shared-storage/compute policy.
    from event_memory.backend_qwen38 import Qwen38Backend
    args.output.mkdir(parents=True)
    files = [Path(__file__), ROOT/'src/event_memory/caption_frontend.py',
             ROOT/'src/event_memory/image_filter.py',
             ROOT/'src/event_memory/caption_runtime.py', ROOT/'src/event_memory/backend_qwen38.py']
    source_hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in files}
    for p in files:
        target = args.output / 'source_snapshot' / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    config = {'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'video_sha256': sha256_file(args.video), 'source_sha256': source_hashes,
              'reference': 'VAM 4f759407945674e0b296ecee22bd53cf910ae39b, adapted causal algorithm',
              'image_filter_mode': image_filter_mode(cfg), 'filter_config': asdict(cfg),
              'embedding_deduplication': False,
              'training': False, 'purpose': 'caption memory construction only',
              'semantic_accuracy_verified': False, 'status': 'loading_models'}
    save(args.output/'config.json', config)
    try:
        encoder = LocalCLIPEncoder(args.embedding_model)
        backend = Qwen38Backend(str(args.writer_model), args.output/'calls.jsonl')
        writer = QwenCaptionWriter(backend, context_limit=args.writer_context_tokens)
        config.update(embedding=encoder.receipt, writer=backend.receipt, status='building')
        save(args.output/'config.json', config)
        build_stream(sampled_frames(args.video, args.frontier, args.output/'frames', args.fps),
                     encoder, writer, args.output, frontier=args.frontier, filter_config=cfg,
                     fallback_distance=args.boundary_fallback_distance,
                     minimum_distance=args.boundary_minimum_distance)
        config['status'] = 'complete_caption_build'
    except Exception as exc:
        config.update(status='failed', error=repr(exc))
        raise
    finally:
        save(args.output/'config.json', config)


if __name__ == '__main__':
    main()
