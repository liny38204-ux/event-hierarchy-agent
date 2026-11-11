"""VAM's image-only gates; sampling and embedding deduplication are separate.

Source: yiliu-li/Visual-Agentic-Memory, commit 4f759407945674e0b296ecee22bd53cf910ae39b,
vam/video.py:264-307. Histogram filtering is disabled, as in its defaults.
"""
from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class VAMImageFilterConfig:
    blur_min: float = 20.0
    duplicate_mad_max: float = 20.0
    duplicate_ssim_min: float = 0.92


class VAMImageFilter:
    """Short-circuit gates against the last kept frame, without tail rescue."""

    def __init__(self, config=VAMImageFilterConfig()):
        values = (config.blur_min, config.duplicate_mad_max, config.duplicate_ssim_min)
        if not all(math.isfinite(v) for v in values) or min(values) < 0:
            raise ValueError('Image filter thresholds must be finite and nonnegative')
        if config.duplicate_ssim_min > 1:
            raise ValueError('SSIM threshold must be in [0, 1]; zero disables its gate')
        try:
            import cv2
            from skimage.metrics import structural_similarity
        except ImportError as exc:
            raise RuntimeError('VAM image filtering requires opencv-python-headless and scikit-image') from exc
        self.cv2, self.ssim = cv2, structural_similarity
        self.config = config
        self.previous_gray, self.reference_id = None, None
        self.audit, self.ids, self.last_time = [], set(), -1.0

    def push(self, record, rgb):
        t = float(record['time'])
        if not math.isfinite(t) or t <= self.last_time or record['id'] in self.ids:
            raise ValueError('Sample IDs must be unique and times strictly increasing')
        pixels = np.asarray(rgb)
        if (pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[-1] != 3
                or min(pixels.shape[:2]) < 3):
            raise ValueError('Expected a uint8 RGB image at least 3x3')
        # Inputs are PIL RGB. Convert explicitly to the official OpenCV BGR path.
        bgr = self.cv2.cvtColor(pixels, self.cv2.COLOR_RGB2BGR)
        gray = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2GRAY)
        if self.previous_gray is not None and gray.shape != self.previous_gray.shape:
            raise ValueError('Frame dimensions changed; do not silently resize filter inputs')
        entry = {'frame_id': record['id'], 'time': t, 'reference_frame_id': self.reference_id,
                 'sharpness': float(self.cv2.Laplacian(gray, self.cv2.CV_64F).var()),
                 'gray_mad': None, 'ssim': None}
        self.last_time = t
        self.ids.add(record['id'])
        self.audit.append(entry)
        if entry['sharpness'] < self.config.blur_min:
            entry['decision'] = 'drop_blur'
            return []
        if self.previous_gray is not None:
            entry['gray_mad'] = float(np.mean(self.cv2.absdiff(gray, self.previous_gray)))
            if entry['gray_mad'] < self.config.duplicate_mad_max:
                entry['decision'] = 'drop_pixel_diff'
                return []
            if self.config.duplicate_ssim_min > 0:
                entry['ssim'] = float(self.ssim(self.cv2.resize(gray, (64, 64)),
                                              self.cv2.resize(self.previous_gray, (64, 64)),
                                              data_range=255))
                if entry['ssim'] > self.config.duplicate_ssim_min:
                    entry['decision'] = 'drop_ssim'
                    return []
        entry['decision'] = 'keep_change' if self.previous_gray is not None else 'keep_first'
        self.previous_gray, self.reference_id = gray.copy(), record['id']
        return [(deepcopy(record), pixels.copy())]

    def flush(self):
        # VAM's image stage discards duplicates immediately, including the tail.
        return []


def make_image_filter(config):
    if isinstance(config, VAMImageFilterConfig):
        return VAMImageFilter(config)
    from .caption_frontend import FilterConfig, FrameFilter
    if isinstance(config, FilterConfig):
        return FrameFilter(config)
    raise TypeError('Unsupported image filter config')


def image_filter_mode(config):
    return 'vam_image' if isinstance(config, VAMImageFilterConfig) else 'conservative'
