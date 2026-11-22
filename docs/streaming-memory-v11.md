# Streaming event-hierarchy memory v11

The v11 pipeline builds a memory from one causal video prefix. It samples at
one frame per second, applies the VAM-style image filter, makes causal content
boundaries, writes a caption for each candidate segment, and reviews captions
in a rolling buffer.

The buffer contains at most five events. A review happens when it reaches five
events, after 30 observed video seconds since the previous review, or at the
specified frontier. A review assigns the pending events to phases and compresses
each phase into one final event while retaining its ordered evidence frames.

The writer uses factual object relationships. It is instructed to avoid
screen-relative descriptions and camera-motion prose where a stable relation
can be stated instead. It does not receive questions, options, labels, or
future frames.

Run it with explicitly supplied local paths:

```bash
python scripts/build_streaming_memory.py \
  --video /path/to/video.mp4 \
  --frontier 59 \
  --video-id demo-001 \
  --writer-model /path/to/Qwen3.8-27B \
  --vision-encoder /path/to/clip-model \
  --output /path/to/new-run
```

The output includes `rolling/<video-id>/memory.json`, `status.json`, source
frame hashes, filtering and boundary audits, and `build_receipt.json`. The
output directory must be new.
