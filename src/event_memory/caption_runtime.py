"""Optional video/model adapters; heavy dependencies load only at runtime."""
import hashlib
import math
from pathlib import Path

import numpy as np
from PIL import Image


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def sampled_frames(path, frontier, destination, fps):
    """PTS-floor sampling; preserve original-resolution JPEGs for audit.

    FPS is explicit, independent of content boundaries and writer batch size.
    A later decoded PTS can trigger emission but its pixels never enter input.
    """
    import av
    if not math.isfinite(frontier) or not math.isfinite(fps) or min(frontier, fps) <= 0:
        raise ValueError('Positive finite frontier and FPS required')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    target, previous, last, number = 0.0, None, -1.0, 0

    def packet(item):
        nonlocal last, number
        pts, image = item
        if pts <= last:
            return None
        if pts > frontier:
            raise ValueError('Future frame')
        last, number = pts, number + 1
        file = destination / f'f{number:06d}.jpg'
        image.save(file, quality=95)
        # Filter, encoder and writer all operate on the same persisted pixels.
        with Image.open(file) as saved:
            pixels = np.asarray(saved.convert('RGB')).copy()
        return {'id': f'f{number:06d}', 'time': pts, 'path': str(file.resolve()),
                'sha256': sha256_file(file)}, pixels

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 2
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError('Missing source timestamp')
            pts = float(frame.pts * stream.time_base)
            if pts < 0:
                continue
            if previous is not None and pts < previous[0]:
                raise ValueError('Nonmonotonic source timestamp')
            while previous is not None and target < pts and target <= frontier:
                if previous[0] <= target:
                    out = packet(previous)
                    if out is not None:
                        yield out
                target += 1 / fps
            if pts > frontier:
                break
            previous = (pts, frame.to_image().convert('RGB'))
        while previous is not None and target <= frontier:
            if previous[0] <= target:
                out = packet(previous)
                if out is not None:
                    yield out
            target += 1 / fps


class LocalCLIPEncoder:
    """Explicit local pretrained visual features; CPU, no hidden model download.

    CLIP is a replaceable initial encoder, not VAM's reported embedding model.
    No questions/text/whole-video temporal features are supplied to this encoder.
    """
    def __init__(self, path):
        import torch
        from transformers import AutoProcessor, CLIPVisionModelWithProjection
        path = Path(path).resolve()
        if not path.is_dir():
            raise ValueError('Provide a local pretrained CLIP model directory')
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.model = CLIPVisionModelWithProjection.from_pretrained(path, local_files_only=True).cpu().eval()
        self.model.requires_grad_(False)
        self.receipt = {'type': 'local_CLIP_visual_embedding', 'path': str(path), 'device': 'cpu',
                        'files_sha256': {p.name: sha256_file(p) for p in path.iterdir()
                                         if p.is_file() and p.suffix in ('.json', '.safetensors', '.bin')}}

    def encode(self, pixels):
        inputs = self.processor(images=Image.fromarray(pixels.astype(np.uint8)), return_tensors='pt')
        with self.torch.inference_mode():
            result = self.model(pixel_values=inputs['pixel_values']).image_embeds[0]
        return result.float().cpu().numpy()


class QwenCaptionWriter:
    """Use existing frozen Qwen backend; measure actual token input before calls."""
    def __init__(self, backend, context_limit=16384, output_tokens=1024):
        if not 0 < output_tokens < context_limit <= 16384:
            raise ValueError('Invalid writer token budget')
        self.backend = backend
        self.context_limit, self.output_tokens = context_limit, output_tokens

    def fits(self, system, text, evidence):
        content, images = [{'type': 'text', 'text': text}], []
        for frame in evidence:
            with Image.open(frame['path']) as source:
                img = source.convert('RGB')
            images.append(img)
            content += [{'type': 'text', 'text': f"Frame {frame['id']}, observed t={frame['time']:.3f}s"},
                        {'type': 'image', 'image': img}]
        messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}]
        prompt = self.backend.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.backend.processor(text=[prompt], images=images or None, return_tensors='pt',
                                        max_pixels=self.backend.max_pixels, min_pixels=4096)
        return int(inputs['input_ids'].shape[-1]) + self.output_tokens <= self.context_limit

    def generate(self, system, text, evidence, *, stage):
        raw = self.backend.generate(system, text, evidence, stage=stage, max_tokens=self.output_tokens)
        if self.backend.calls[-1].get('finish_reason') != 'stop':
            raise ValueError('Caption truncated; input coverage cannot be marked successful')
        return raw
