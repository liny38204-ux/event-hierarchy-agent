"""Frozen official Qwen3.8-27B, shared between independent writer/manager calls."""
from contextlib import contextmanager
from contextvars import ContextVar
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import time


# Writer outputs and manager inputs can contain superseded observations.
# The live prefix builder only retains per-call metrics while consuming them.
TRANSIENT_MEMORY_IO = ContextVar('transient_memory_io', default=False)


@contextmanager
def transient_memory_calls():
    token = TRANSIENT_MEMORY_IO.set(True)
    try:
        yield
    finally:
        TRANSIENT_MEMORY_IO.reset(token)


def memory_call_metrics(record):
    return {k: v for k, v in record.items() if k in (
        'stage', 'input_tokens', 'output_tokens', 'finish_reason', 'seconds',
        'peak_allocated_bytes', 'image_grid_thw', 'max_pixels_per_image')}


def json_ready(value):
    """Preserve loading diagnostics when Transformers returns sets."""
    if isinstance(value,dict):return {str(k):json_ready(v) for k,v in value.items()}
    if isinstance(value,(set,frozenset)):return [json_ready(v) for v in sorted(value,key=repr)]
    if isinstance(value,(tuple,list)):return [json_ready(v) for v in value]
    if value is None or isinstance(value,(str,int,float,bool)):return value
    raise TypeError('Unexpected receipt type: '+type(value).__name__)


class FatalCudaError(BaseException):
    """Prevent a poisoned CUDA context from entering JSON-repair loops."""


class Qwen38Backend:
    def __init__(self, model_path, log_path, max_pixels=100352):
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        from transformers.models.qwen3_5 import modeling_qwen3_5
        from .bounded_delta_prefill import bounded_delta_rule
        if not getattr(modeling_qwen3_5.torch_chunk_gated_delta_rule, "_epm_bounded", False):
            bounded = bounded_delta_rule(modeling_qwen3_5.torch_chunk_gated_delta_rule, tile_size=512)
            bounded._epm_bounded = True
            modeling_qwen3_5.torch_chunk_gated_delta_rule = bounded
        if importlib.util.find_spec('fla') is not None:
            raise RuntimeError('This frozen bounded-torch runtime requires no FLA override')
        if torch.cuda.device_count()!=2:
            raise RuntimeError('Explicitly allocate exactly two idle GPUs for the 27B BF16 model')
        path=Path(model_path)
        verified=json.loads((path/'verified_model.json').read_text())
        if verified['repo_id']!='Qwen/Qwen3.8-27B':raise ValueError('Wrong model identity')
        for item in verified['files']:
            file=path/item['name']
            if file.stat().st_size!=item['bytes']:raise ValueError('Model file changed: '+item['name'])
            if file.suffix!='.safetensors' and hashlib.sha256(file.read_bytes()).hexdigest()!=item['sha256']:
                raise ValueError('Model metadata changed: '+item['name'])
        torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.manual_seed(20261105)
        self.torch=torch;self.calls=[];self.log_path=Path(log_path);self.max_pixels=max_pixels
        self.model,loading_info=Qwen3_5ForConditionalGeneration.from_pretrained(
            path,dtype=torch.bfloat16,device_map='balanced',
            max_memory={0:'27GiB',1:'27GiB'},attn_implementation='sdpa',
            local_files_only=True,output_loading_info=True)
        if loading_info.get('missing_keys') or loading_info.get('mismatched_keys'):
            raise RuntimeError('Incomplete full model load: '+repr(loading_info))
        if any(str(d) in ('cpu','disk') for d in self.model.hf_device_map.values()):
            raise RuntimeError('CPU/disk model offload is not permitted in this run')
        self.model.eval().requires_grad_(False)
        self.processor=AutoProcessor.from_pretrained(path,local_files_only=True)
        # Same per-patch linear algebra as the official non-overlapping Conv3d;
        # preserve all weights. BF16 numerical identity is not claimed.
        layer=self.model.model.visual.patch_embed;conv=layer.proj
        if not (conv.kernel_size==conv.stride and conv.padding==(0,0,0)
                and conv.dilation==(1,1,1) and conv.groups==1):
            raise RuntimeError('Unexpected patch projection')
        def project(hidden_states):
            values=hidden_states.reshape(-1,conv.weight[0].numel()).to(conv.weight.dtype)
            return torch.nn.functional.linear(values,conv.weight.flatten(1),conv.bias)
        layer.forward=project
        self.input_device=self.model.get_input_embeddings().weight.device
        self.receipt={'repo_id':verified['repo_id'],'revision':verified['revision'],
            'verified_model_sha256':hashlib.sha256((path/'verified_model.json').read_bytes()).hexdigest(),
            'model_path':str(path),'torch':torch.__version__,'transformers':transformers.__version__,
            'dtype':'bfloat16','attention':'sdpa','patch_embed':'linear_same_weights',
            'runtime_kernel':'torch_gdn_outer_tile512_inner64_v1',
            'enable_thinking':False,'do_sample':False,'seed':20261105,'max_pixels':max_pixels,
            'device_map':self.model.hf_device_map,'gpus':[torch.cuda.get_device_name(i) for i in range(2)],
            'trainable_parameters':sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'loading_info':json_ready(loading_info)}

    def token_count(self,text):
        return len(self.processor.tokenizer.encode(text,add_special_tokens=False))

    def generate(self,system,text,evidence=(),*,stage,max_tokens=2048):
        from PIL import Image
        content=[{'type':'text','text':text}]
        images=[]
        pixel_cap = self.max_pixels
        for e in evidence:
            with Image.open(e['path']) as im:img=im.convert('RGB')
            images.append(img)
            if e.get('kind') == 'contact_sheet':
                if len(evidence) != 1 or not 1 <= len(e['frame_ids']) <= 6:
                    raise ValueError('expected a single contact sheet with one to six frames')
                pixel_cap *= len(e['frame_ids'])
                label = 'Contact sheet, chronological frame IDs: ' + ', '.join(e['frame_ids'])
            else:
                label = f"Frame {e['id']}, observed t={e['time']:.3f}s"
            content += [{'type':'text','text':label},
                        {'type':'image','image':img}]
        messages=[{'role':'system','content':system},{'role':'user','content':content}]
        started=time.monotonic();record={'stage':stage,'system':system,'text':text,'evidence':list(evidence)}
        inputs=output=ids=None
        try:
            for i in range(2):self.torch.cuda.reset_peak_memory_stats(i)
            prompt=self.processor.apply_chat_template(messages,tokenize=False,
                add_generation_prompt=True,enable_thinking=False)
            inputs=self.processor(text=[prompt],images=images or None,return_tensors='pt',
                max_pixels=pixel_cap,min_pixels=4096).to(self.input_device)
            length=int(inputs['input_ids'].shape[-1])
            record['max_pixels_per_image']=pixel_cap
            record['input_tokens']=length
            if length+max_tokens>16384:raise ValueError('Context limit exceeded; no silent truncation')
            for i in range(2):self.torch.cuda.synchronize(i)
            with self.torch.inference_mode():
                output=self.model.generate(**inputs,max_new_tokens=max_tokens,do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id)
            for i in range(2):self.torch.cuda.synchronize(i)
            ids=output[:,length:]
            raw=self.processor.batch_decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
            record.update(raw=raw,input_tokens=length,output_tokens=int(ids.shape[-1]),
                finish_reason='length' if ids.shape[-1]>=max_tokens else 'stop',
                image_grid_thw=inputs['image_grid_thw'].tolist() if 'image_grid_thw' in inputs else [])
            return raw
        except Exception as exc:
            record['error']=repr(exc)
            if any(marker in str(exc).lower() for marker in ('illegal memory access','device-side assert','cublas_status_execution_failed','cublas_status_internal_error')):
                raise FatalCudaError(str(exc)) from exc
            raise
        finally:
            # Drop generation tensors even after OOM before a caller retries.
            inputs=output=ids=None
            gc.collect()
            try:
                self.torch.cuda.empty_cache()
                record['peak_allocated_bytes']=[self.torch.cuda.max_memory_allocated(i) for i in range(2)]
            except Exception as cleanup_error:
                record['cleanup_error']=repr(cleanup_error)
            record['seconds']=time.monotonic()-started
            if TRANSIENT_MEMORY_IO.get():
                record=memory_call_metrics(record)
            self.calls.append(record)
            self.log_path.parent.mkdir(parents=True,exist_ok=True)
            with self.log_path.open('a') as out:out.write(json.dumps(record,ensure_ascii=False)+'\n')
            print(json.dumps({k:record.get(k) for k in ('stage','seconds','input_tokens','output_tokens','error')}),flush=True)
