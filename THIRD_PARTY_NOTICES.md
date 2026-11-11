# Third-party references

## Visual Agentic Memory (VAM)

`src/event_memory/image_filter.py` adapts the image-only filtering rules from
[Visual-Agentic-Memory/vam/video.py](https://github.com/yiliu-li/Visual-Agentic-Memory/blob/4f759407945674e0b296ecee22bd53cf910ae39b/vam/video.py#L264-L307),
commit `4f759407945674e0b296ecee22bd53cf910ae39b`.

The adaptation preserves blur / mean absolute pixel difference / SSIM gates,
uses explicit RGB-to-BGR conversion, and adds causal reference tracking,
validation and audit records. Embedding deduplication is not enabled.

The upstream MIT notice is reproduced in [third_party/VAM-LICENSE](third_party/VAM-LICENSE).
This notice does not assign a project-wide license to the rest of this repository.

## Model and library dependencies

Qwen and CLIP weights are external dependencies, not distributed here. The
recorded Qwen revision is metadata only; the downloader verifies local files.
PyTorch, Transformers, NumPy, Pillow, OpenCV, scikit-image and PyAV are installed
separately under their respective licenses.
