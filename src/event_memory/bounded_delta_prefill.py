"""Bound FP32 temporaries in the official torch Gated DeltaNet fallback.

Every token is evaluated in order. Outer tiles align with the reference kernel's
64-token chunks and carry its FP32 recurrent state; no memory/attention tokens
are dropped and multimodal positions are untouched. Inference only.
"""
from functools import wraps


def bounded_delta_rule(reference, tile_size=512):
    @wraps(reference)
    def forward(query, key, value, g, beta, chunk_size=64, initial_state=None,
                output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs):
        import torch
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4 or g.ndim != 3 or beta.ndim != 3:
            raise ValueError('Unexpected GDN tensor ranks')
        if len({tuple(x.shape[:2]) for x in (query,key,value,g,beta)}) != 1:
            raise ValueError('GDN batch/sequence lengths differ')
        if len({x.device for x in (query,key,value,g,beta)}) != 1:
            raise ValueError('GDN input devices differ')
        if tile_size < chunk_size or tile_size % chunk_size:
            raise ValueError('Outer tile must be a positive multiple of the reference chunk size')
        if kwargs.get('cu_seqlens') is not None:
            raise ValueError('Packed variable-length sequences are unsupported by bounded fallback')
        if torch.is_grad_enabled() and any(x.requires_grad for x in (query,key,value,g,beta,initial_state) if x is not None):
            raise RuntimeError('Bounded fallback is inference-only')
        n = query.shape[1]
        if n <= tile_size:
            return reference(query,key,value,g,beta,chunk_size=chunk_size,
                             initial_state=initial_state,output_final_state=output_final_state,
                             use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,**kwargs)
        output = torch.empty((*value.shape[:-1],value.shape[-1]),device=value.device,dtype=query.dtype)
        state = initial_state
        for start in range(0,n,tile_size):
            end = min(n,start+tile_size)
            piece,state = reference(query[:,start:end],key[:,start:end],value[:,start:end],
                g[:,start:end],beta[:,start:end],chunk_size=chunk_size,initial_state=state,
                output_final_state=True,use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,**kwargs)
            output[:,start:end].copy_(piece)
        return output,state if output_final_state else None
    return forward
