## Note: while I could use the normal library and
## just apply RoPE like normal. Since RoPE is relative
## this should not be a problem. However, since I use
## fp16 or bf16, the relative positions accumulate error.
## This means applying RoPE on a sequence with 2 batches
## in the order [1, 2] and [2, 1] are not the same. The fix
## is not just making the input/output of RoPE fp32 but
## also the kernel, which is not happening. So I make a
## varlen version of RoPE to make fwd passes batch order invariant.
# Adapted from
# https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py

from src.exp_configs import ModelConfigs
from src.utils import generate_random_cu_seqlens

import torch
from torch.amp import autocast
from torch.nn import Module
from torch import nn, Tensor
from einops import rearrange, repeat
from typing import Optional

# rotary embedding helper functions

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

@autocast('cuda', enabled = False)
def apply_rotary_emb(
    freqs,
    t,
):
    dtype = t.dtype
    rot_dim = freqs.shape[-1]
    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'

    # Apply rotary embeddings without modifying t in place    
    t_transformed = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())

    return t_transformed.type(dtype)




def get_seq_pos(
        seq_len,
        cu_seqlens, 
        device, 
    ):
    # Inference path
    if seq_len == cu_seqlens.shape[0]-1:
        return cu_seqlens.diff()-1

    # Lengths of all sequences
    seq_lens = cu_seqlens.diff()

    # Generate indices starting at 0 for each sequence
    indices = torch.arange(seq_len, device=device, dtype=cu_seqlens.dtype)
    offsets = torch.repeat_interleave(cu_seqlens[:-1], seq_lens, output_size=seq_len)
    arange_indices = indices - offsets

    return arange_indices




class VarlenRotaryEmbedding(Module):
    def __init__(
        self,
        dim,
        theta = 10_000,
        dtype = torch.float32
    ):
        super().__init__()

        freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))

        self.freqs = nn.Parameter(freqs, requires_grad = False)

        # dummy for device
        self.register_buffer('dummy', torch.tensor(0), persistent = False)

        # default sequence dimension
        self.seq_before_head_dim = False
        self.default_seq_dim = 1

        # add apply_rotary_emb as static method
        self.apply_rotary_emb = staticmethod(apply_rotary_emb)
        
        self.dtype = dtype

    @property
    def device(self):
        return self.dummy.device

    @autocast('cuda', enabled = False)
    def get_frequencies(
        self,
        t: Tensor,
    ):
        freqs = self.freqs

        # Scale the frequencies by the timestep
        # (bss, d/2)
        freqs = t[:, None] * self.freqs[None, :]
        
        # Repeat frequencies for sin/cos
        # This is just a repeat interleaved
        # (bss, d)
        # torch.repeat_interleave(freqs, 2, 1)
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)

        return freqs
    
    @autocast('cuda', enabled = False)
    def forward(
            self, 
            X: Tensor,
            cu_seqlens: torch.Tensor,
            seq_pos: Optional[torch.Tensor] = None,
        ):
        if (
            seq_pos is None
            and X.is_cuda
            and X.ndim == 3
            and X.shape[-1] == self.freqs.numel() * 2
        ):
            from src.kernels.rope_varlen_fast import apply_rope_varlen
            return apply_rope_varlen(X, cu_seqlens, self.freqs)
        # Cast to dtype
        dtype_orig = X.dtype
        X = X.to(self.dtype)

        seq_dim = self.default_seq_dim
        device, dtype, seq_len = X.device, X.dtype, X.shape[seq_dim]

        # Sequence positions for each token
        if seq_pos is None:
            seq_pos = get_seq_pos(
                seq_len,
                cu_seqlens,
                device=device, 
            )

        # Get frequencies for each position
        freqs = self.get_frequencies(
            seq_pos
        )

        # Apply the RoPE frequencies to the sequence
        # and cast dtype back
        return apply_rotary_emb(
            freqs, 
            X
        ).to(dtype_orig)
    
    
    
