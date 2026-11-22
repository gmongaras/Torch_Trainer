"""Varlen rotary embedding.

Same positions and pair rotation as ``VarlenRotaryEmbedding``. Cos and sin are
built once per sequence and reused across heads. The Python path recomputed
them per head and materialized several fp32 copies.
"""
import torch
import triton
import triton.language as tl

from src.model.layers.VarlenRotaryEmbedding import get_seq_pos


@triton.jit
def _rope_kernel(
        X, Y, cos, sin,
        SEQ, NROWS, DIM,
        BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
        SIGN: tl.constexpr, DTYPE: tl.constexpr,
    ):
    row0 = tl.program_id(0) * BLOCK_T
    pair0 = tl.program_id(1) * BLOCK_D
    offs_row = row0 + tl.arange(0, BLOCK_T)
    offs_d = pair0 + tl.arange(0, BLOCK_D)
    mask_row = offs_row < NROWS
    mask_d = offs_d < (DIM // 2)
    token = offs_row - (offs_row // SEQ) * SEQ
    base_f = (token * (DIM // 2))[:, None] + offs_d[None, :]
    valid_f = mask_row[:, None] & mask_d[None, :]
    c = tl.load(cos + base_f, mask=valid_f, other=0).to(tl.float32)
    s = tl.load(sin + base_f, mask=valid_f, other=0).to(tl.float32) * SIGN
    offs_col = pair0 * 2 + tl.arange(0, BLOCK_D * 2)
    mask_col = offs_col < DIM
    base = (offs_row * DIM)[:, None] + offs_col[None, :]
    valid = mask_row[:, None] & mask_col[None, :]
    x = tl.load(X + base, mask=valid, other=0).to(tl.float32)
    x0, x1 = tl.reshape(x, (BLOCK_T, BLOCK_D, 2)).split()
    y0 = x0 * c - x1 * s
    y1 = x0 * s + x1 * c
    y = tl.reshape(tl.join(y0, y1), (BLOCK_T, BLOCK_D * 2)).to(DTYPE)
    tl.store(Y + base, y, mask=valid)


_DTYPE = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


def _tables(cu, freqs, seq):
    pos = get_seq_pos(seq, cu, freqs.device)
    angle = pos.to(torch.float32)[:, None] * freqs.to(torch.float32)[None, :]
    return angle.cos().contiguous(), angle.sin().contiguous()


def _launch(x, cos, sin, sign):
    heads, seq, dim = x.shape
    block_t, block_d = 32, 32
    y = torch.empty_like(x)
    nrows = heads * seq
    _rope_kernel[(triton.cdiv(nrows, block_t), triton.cdiv(dim // 2, block_d))](
        x, y, cos, sin,
        seq, nrows, dim,
        BLOCK_T=block_t, BLOCK_D=block_d,
        SIGN=sign, DTYPE=_DTYPE[x.dtype],
        num_warps=4,
    )
    return y


class _rope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cu, freqs):
        cos, sin = _tables(cu, freqs, x.shape[1])
        y = _launch(x, cos, sin, 1)
        ctx.save_for_backward(cu, freqs)
        ctx.shape = x.shape
        return y

    @staticmethod
    def backward(ctx, dy):
        cu, freqs = ctx.saved_tensors
        cos, sin = _tables(cu, freqs, ctx.shape[1])
        dx = _launch(dy.contiguous().view(ctx.shape), cos, sin, -1)
        return dx, None, None


def apply_rope_varlen(x, cu_seqlens, freqs):
    """Rotate ``x`` of shape ``(heads, seq, dim)`` with varlen positions."""
    assert x.ndim == 3 and x.shape[-1] % 2 == 0
    if not x.is_contiguous():
        x = x.contiguous()
    cu = cu_seqlens.to(device=x.device, dtype=torch.int32).contiguous()
    return _rope.apply(x, cu, freqs)
