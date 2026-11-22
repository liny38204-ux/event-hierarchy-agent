"""Deterministic Qwen3-VL-8B reader for the frozen EPM evidence protocol.

This adapter is intentionally limited to reading already-created evidence.
It has the same small backend interface used by ``short_window`` and
``HistoryReader``: ``generate``, ``token_count``, ``calls``, ``receipt`` and
``torch``.  It never builds, rewrites, or updates video memory.
"""
from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path

class FatalCudaError(BaseException):
    """Terminate a reader worker after a poisoned CUDA context."""


def json_ready(value):
    """Make model loading diagnostics JSON serializable."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return [json_ready(item) for item in sorted(value, key=repr)]
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("unexpected receipt value: " + type(value).__name__)


MODEL_REPO = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"


def validate_model_manifest(model_path: Path) -> dict:
    """Validate the pinned local checkpoint without loading it onto a GPU."""
    model_path = Path(model_path)
    manifest_path = model_path / "verified_model.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("repo_id") != MODEL_REPO:
        raise ValueError("unexpected 8B reader repository")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError("unexpected 8B reader revision")
    for item in manifest.get("files", []):
        candidate = model_path / item["name"]
        if not candidate.is_file() or candidate.stat().st_size != item["bytes"]:
            raise ValueError("8B reader checkpoint file changed: " + item["name"])
        # The shard checksums are retained in the signed local manifest.  Size
        # validation avoids reading 17 GB of shards at every worker launch.
        if candidate.suffix != ".safetensors":
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            if digest != item["sha256"]:
                raise ValueError("8B reader metadata changed: " + item["name"])
    config = json.loads((model_path / "config.json").read_text())
    if config.get("architectures") != ["Qwen3VLForConditionalGeneration"]:
        raise ValueError("checkpoint is not Qwen3-VL-8B-Instruct")
    return manifest


def install_linear_patch_projection(model):
    """Replace a non-overlapping visual Conv3d with the same BF16 linear map.

    Qwen3-VL packs one complete ``C×T×H×W`` patch per row before the visual
    projection.  When the Conv3d has no overlap, padding, dilation, or groups,
    every output voxel is exactly a dot product with that row.  The replacement
    retains the checkpoint's weight and bias.  BF16 numerical identity is not
    claimed because Conv3d and ``linear`` can use different CUDA kernels.
    """
    import torch

    layer = model.model.visual.patch_embed
    convolution = layer.proj
    if not isinstance(convolution, torch.nn.Conv3d):
        raise RuntimeError("unexpected Qwen3-VL visual projection type")
    if not (
        convolution.kernel_size == convolution.stride
        and convolution.padding == (0, 0, 0)
        and convolution.dilation == (1, 1, 1)
        and convolution.groups == 1
    ):
        raise RuntimeError("visual Conv3d is not a non-overlapping patch projection")

    patch_width = convolution.weight[0].numel()

    def project(hidden_states):
        values = hidden_states.reshape(-1, patch_width).to(convolution.weight.dtype)
        return torch.nn.functional.linear(values, convolution.weight.flatten(1), convolution.bias)

    layer.forward = project


class Qwen3VL8BBackend:
    """One-GPU deterministic reader for timestamped images and text evidence."""

    def __init__(self, model_path, log_path, *, device_index=0, max_pixels=100352, linear_patches=False):
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if not isinstance(device_index, int) or device_index < 0:
            raise ValueError("device_index must be a non-negative integer")
        if torch.cuda.device_count() <= device_index:
            raise RuntimeError("the selected 8B reader GPU is unavailable")
        if not isinstance(max_pixels, int) or max_pixels < 4096:
            raise ValueError("invalid visual pixel budget")

        path = Path(model_path)
        manifest = validate_model_manifest(path)
        torch.set_num_threads(8)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # Other worker components may have already started OpenMP work.
            pass
        torch.manual_seed(20261105)

        self.torch = torch
        self.calls = []
        self.log_path = Path(log_path)
        self.max_pixels = max_pixels
        self.device_index = device_index
        self.model, loading_info = Qwen3VLForConditionalGeneration.from_pretrained(
            path,
            dtype=torch.bfloat16,
            device_map={"": device_index},
            attn_implementation="sdpa",
            local_files_only=True,
            output_loading_info=True,
        )
        if loading_info.get("missing_keys") or loading_info.get("mismatched_keys"):
            raise RuntimeError("incomplete 8B reader load: " + repr(loading_info))
        # Transformers omits ``hf_device_map`` for some single-device loads.
        # Inspect real parameter placements in every case, because a receipt
        # field alone cannot prove that no CPU or disk offload occurred.
        declared_map = getattr(self.model, "hf_device_map", None)
        if declared_map is not None and any(str(device) in ("cpu", "disk") for device in declared_map.values()):
            raise RuntimeError("8B reader CPU/disk offload is not permitted")
        parameter_devices = {str(parameter.device) for parameter in self.model.parameters()}
        expected_device = f"cuda:{device_index}"
        if parameter_devices != {expected_device}:
            raise RuntimeError("8B reader parameters are not entirely on " + expected_device + ": " + repr(parameter_devices))
        self.model.eval().requires_grad_(False)
        if not isinstance(linear_patches, bool):
            raise ValueError("linear_patches must be boolean")
        if linear_patches:
            install_linear_patch_projection(self.model)
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.input_device = self.model.get_input_embeddings().weight.device
        self.receipt = {
            "repo_id": manifest["repo_id"],
            "revision": manifest["revision"],
            "verified_model_sha256": hashlib.sha256((path / "verified_model.json").read_bytes()).hexdigest(),
            "model_path": str(path),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "dtype": "bfloat16",
            "attention": "sdpa",
            "patch_embed": "linear_same_weights" if linear_patches else "native",
            "patch_embed_note": (
                "non-overlapping Conv3d replaced by same-weight linear projection; BF16 numerical identity not claimed"
                if linear_patches else "official native Conv3d"
            ),
            "enable_thinking": False,
            "do_sample": False,
            "seed": 20261105,
            "max_pixels": max_pixels,
            "device_index": device_index,
            "device_map": declared_map if declared_map is not None else {"": device_index},
            "parameter_devices": sorted(parameter_devices),
            "gpu": torch.cuda.get_device_name(device_index),
            "trainable_parameters": sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad),
            "loading_info": json_ready(loading_info),
        }

    def token_count(self, text):
        return len(self.processor.tokenizer.encode(text, add_special_tokens=False))

    def choose_option(self, system, text, *, evidence, options, stage):
        from event_memory.answer_options import option_letters
        letters=option_letters(options)
        return self.generate(system, text, evidence=evidence, stage=stage, max_tokens=1,
                             allowed_letters=letters)

    def generate(self, system, text, evidence=(), *, stage, max_tokens=2048, allowed_letters=None):
        """Generate under the exact 27B evidence-binding convention."""
        from PIL import Image

        images = []
        pixel_caps = []
        for item in evidence:
            path = Path(item["path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError("evidence bytes disagree with the source manifest")
            with Image.open(path) as opened:
                images.append(opened.convert("RGB"))
            cells = len(item["frame_ids"]) if item.get("kind") == "contact_sheet" else 1
            if not 1 <= cells <= 6:
                raise ValueError("unexpected historical evidence page size")
            pixel_caps.append(self.max_pixels * cells)
        # Four raw frames get the same per-image budget. History pages may have varying cell counts.
        pixel_cap = max(pixel_caps, default=self.max_pixels)
        for image, cap in zip(images, pixel_caps):
            if cap < pixel_cap and image.width * image.height > cap:
                ratio = (cap / (image.width * image.height)) ** 0.5
                image.thumbnail((max(1, int(image.width * ratio)), max(1, int(image.height * ratio))), Image.Resampling.LANCZOS)
        content = [{"type":"image", "image":image} for image in images]
        content.append({"type":"text", "text":text})
        messages = [{"role":"system", "content":system}, {"role":"user", "content":content}]
        started = time.monotonic()
        record = {"stage": stage, "system": system, "text": text, "evidence": list(evidence), "user_content_order": [i["type"] for i in content], "per_image_pixel_caps": pixel_caps}
        inputs = output = generated_ids = None
        try:
            self.torch.cuda.reset_peak_memory_stats(self.device_index)
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            inputs = self.processor(
                text=[prompt], images=images or None, return_tensors="pt", images_kwargs={"size":{"longest_edge":pixel_cap,"shortest_edge":4096}}
            ).to(self.input_device)
            grids=inputs.get("image_grid_thw")
            if images and (grids is None or len(grids)!=len(images)):
                raise ValueError("visual encoder expected one separate grid per input image")
            input_length = int(inputs["input_ids"].shape[-1])
            record.update(input_tokens=input_length, max_pixels_per_image=pixel_cap)
            if input_length + max_tokens > 16384:
                raise ValueError("context limit exceeded; no silent truncation")
            self.torch.cuda.synchronize(self.device_index)
            with self.torch.inference_mode():
                if allowed_letters is None:
                    output = self.model.generate(
                        **inputs, max_new_tokens=max_tokens, do_sample=False,
                        pad_token_id=self.processor.tokenizer.eos_token_id)
                    generated_ids = output[:,input_length:]
                    raw = self.processor.batch_decode(generated_ids,skip_special_tokens=True,
                        clean_up_tokenization_spaces=False)[0]
                else:
                    tokenizer=self.processor.tokenizer
                    ids=[]
                    for letter in allowed_letters:
                        tokens=tokenizer.encode(letter,add_special_tokens=False)
                        if len(tokens)!=1 or tokenizer.decode(tokens)!=letter:
                            raise ValueError(f"option letter is not a standalone model token: {letter}")
                        ids.append(tokens[0])
                    # One forward pass, score precisely the next assistant token after the chat template.
                    output=self.model(**inputs,use_cache=False,logits_to_keep=1)
                    scores=output.logits[0,-1,ids].float()
                    if not self.torch.isfinite(scores).all().item():
                        raise ValueError("non-finite model logits for the available choices")
                    pos=int(scores.argmax().item())
                    raw=allowed_letters[pos]
                    generated_ids=self.torch.tensor([[ids[pos]]])
                    record.update(option_token_ids=ids,option_logits=dict(zip(allowed_letters,scores.tolist())),
                                  decoding="argmax_over_actual_option_letters")
            self.torch.cuda.synchronize(self.device_index)
            record.update(
                raw=raw,
                output_tokens=int(generated_ids.shape[-1]),
                finish_reason="option_selected" if allowed_letters is not None else ("length" if generated_ids.shape[-1] >= max_tokens else "stop"),
                image_grid_thw=inputs["image_grid_thw"].tolist() if "image_grid_thw" in inputs else [],
            )
            return raw
        except Exception as error:
            record["error"] = repr(error)
            poisoned = ("illegal memory access", "device-side assert", "cublas_status_execution_failed", "cublas_status_internal_error")
            if any(marker in str(error).lower() for marker in poisoned):
                raise FatalCudaError(str(error)) from error
            raise
        finally:
            inputs = output = generated_ids = None
            gc.collect()
            try:
                self.torch.cuda.empty_cache()
                record["peak_allocated_bytes"] = self.torch.cuda.max_memory_allocated(self.device_index)
            except Exception as cleanup_error:
                record["cleanup_error"] = repr(cleanup_error)
            record["seconds"] = time.monotonic() - started
            self.calls.append(record)
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps({key: record.get(key) for key in ("stage", "seconds", "input_tokens", "output_tokens", "error")}), flush=True)
