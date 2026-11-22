import torch
from torch import nn
from src.model.layers.Layer import Layer
from enum import Enum
from typing import Optional
from torch.nn.attention.varlen import varlen_attn

from src.model.layers.VarlenRotaryEmbedding import VarlenRotaryEmbedding
from src.types import (
    AttentionType,
    InfMode
)
from src.configs import (
    InfConfig,
    ModelConfig,
    AttentionConfig
)
from src.utils import (
    generate_random_cu_seqlens,
    get_num_params
)
from src.exp_configs import ModelConfigs
from src.model.Metrics import Metrics, dummy_metrics
from src.model.Timing import Timing, dummy_timing
from src.model.Losses import Losses, dummy_losses
from src.model.layers.FusedRMSNorm import FusedRMSNorm



# Used to get QKV projection
def get_qkvo_proj(
        d_model,
        qk_head_dim,
        v_head_dim,
        num_heads_q,
        num_heads_kv,
    ):
    return (
        nn.Linear(
            d_model,
            qk_head_dim*num_heads_q 
                + qk_head_dim*num_heads_kv 
                + v_head_dim*num_heads_kv,
            bias=False
        ),
        nn.Linear(
            v_head_dim*num_heads_q, 
            d_model,
            bias=False
        )
    )



class Attention(Layer):
    def __init__(
            self,
            model_config: ModelConfig,
            attn_config: AttentionConfig,
            layer_num: int,
        ):
        super().__init__()
        
        self.model_config = model_config
        self.attn_config = attn_config
        self.d_model = model_config.d_model
        self.attn_type = attn_config.attn_type
        self.qk_head_dim = attn_config.qk_head_dim
        self.v_head_dim = attn_config.v_head_dim
        self.num_heads_q = attn_config.num_heads_q
        self.num_heads_kv = attn_config.num_heads_kv
        self.do_qk_norm = attn_config.do_qk_norm
        self.use_rope = attn_config.use_rope
        self.is_causal = attn_config.is_causal
        self.checkpoint = attn_config.checkpoint
        self.max_seqlen = model_config.max_seqlen
        self.layernorm_eps = model_config.layernorm_eps
        self.scale = 1/(self.qk_head_dim**0.5)
        self.layer_num = layer_num
        
        assert self.num_heads_q % self.num_heads_kv == 0
        self.gqa_factor = self.num_heads_q//self.num_heads_kv
        
        assert type(self.attn_type) is AttentionType
        
        # QKVO projections
        self.qkv_proj, self.o_proj = get_qkvo_proj(
            self.d_model, 
            self.qk_head_dim, 
            self.v_head_dim, 
            self.num_heads_q, 
            self.num_heads_kv
        )
        
        if self.attn_type == AttentionType.TORCH_SDPA:
            self.attention = self.torch_sdpa_fwd
        else:
            assert False
            
        # RoPE
        if self.use_rope:
            self.rotary_emb = VarlenRotaryEmbedding(
                self.qk_head_dim
            )
            
        # QK Norm
        if self.do_qk_norm:
            self.q_norm = FusedRMSNorm(
                self.qk_head_dim, 
                eps=self.layernorm_eps
            )
            self.k_norm = FusedRMSNorm(
                self.qk_head_dim, 
                eps=self.layernorm_eps
            )

        # kv cache
        self.k_cache = None
        self.v_cache = None
            
    
    def proj_to_qkv(self, X, transpose=False):
        bss, d_model = X.shape
        
        # Get qkv tensors
        q, k, v = self.qkv_proj(X).split(
            (
                self.num_heads_q*self.qk_head_dim,
                self.num_heads_kv*self.qk_head_dim,
                self.num_heads_kv*self.v_head_dim
            ),
            dim=-1
        )
        
        # Create heads
        q = q.reshape(bss, self.num_heads_q, self.qk_head_dim).transpose(0, 1)
        k = k.reshape(bss, self.num_heads_kv, self.qk_head_dim).transpose(0, 1)
        v = v.reshape(bss, self.num_heads_kv, self.v_head_dim).transpose(0, 1)
        
        # Output of shape
        # (H, bss, d_H)
        return q,k,v
    
            
    def torch_sdpa_fwd(
            self, 
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            cu_seqlens: Optional[torch.Tensor],
            inf_cfg: Optional[InfConfig] = None,
        ):
        _, bss, _ = q.shape
        prefill = inf_cfg is None or inf_cfg.inf_mode == InfMode.PREFILL
        out = varlen_attn( # Expects (BSS, H, d)
            query=q.half().contiguous().transpose(0, 1),
            key=k.half().contiguous().transpose(0, 1),
            value=v.half().contiguous().transpose(0, 1),
            cu_seq_q=(
                cu_seqlens.to(torch.int32).contiguous()
                if prefill
                else torch.arange(
                    0,
                    bss+1,
                    device=cu_seqlens.device,
                    dtype=torch.int32
                )
            ),
            cu_seq_k=cu_seqlens.to(torch.int32).contiguous(),
            max_q=(
                bss
                if prefill
                else 1
            ),
            max_k=(
                bss
                if prefill
                else cu_seqlens.diff().max()
            ),
            scale=self.scale,
            window_size=(-1, 0) if self.is_causal else (-1, -1)
        ).to(q.dtype)
        return out.reshape(bss, self.v_head_dim*self.num_heads_q)



    def handle_kv_cache(
            self,
            k: torch.Tensor,
            v: torch.Tensor,
            cu_seqlens: torch.Tensor,
            inf_cfg: Optional[InfConfig] = None
        ):
        # If the inf config is None, don't do anything
        # with the kv cache
        if inf_cfg is None:
            return k, v
        
        # Reset cache
        if inf_cfg.reset_cache:
            # Just going to allocate a block to the max
            # size if this is the first allocation. Otherwise,
            # set the positions to 0
            self.kv_cache_batch_size = cu_seqlens.shape[0]-1
            if self.k_cache is None:
                self.k_cache = torch.empty(
                    self.kv_cache_batch_size,
                    self.num_heads_kv,
                    self.model_config.max_seqlen,
                    self.qk_head_dim,
                    dtype=k.dtype,
                    device=k.device,
                )
            if self.v_cache is None:
                self.v_cache = torch.empty(
                    self.kv_cache_batch_size,
                    self.num_heads_kv,
                    self.model_config.max_seqlen,
                    self.v_head_dim,
                    dtype=v.dtype,
                    device=v.device,
                )
            self.k_cache_pos = [0]*self.kv_cache_batch_size
            self.v_cache_pos = [0]*self.kv_cache_batch_size

        # Ensure the kv cache isn't None
        assert (
            self.k_cache is not None 
            and self.v_cache is not None
        ), "kv cache cannot be None. Did you prefill with reset_cache=True?"

        # Ensure we pass a batch with the same
        # saved batch size
        assert self.kv_cache_batch_size == cu_seqlens.shape[0]-1

        cur_lo_k = 0
        cur_lo_v = 0
        for b in range(0, self.kv_cache_batch_size):
            # Don't update the KV cache
            if not inf_cfg.update_cache[b]:
                if inf_cfg.inf_mode == InfMode.DECODE:
                    cur_lo_k += 1
                    cur_lo_v += 1
                continue

            lo, hi = cu_seqlens[b], cu_seqlens[b+1]

            # Number of new tokens for k and v
            num_new_toks_k = hi-lo-self.k_cache_pos[b]
            num_new_toks_v = hi-lo-self.v_cache_pos[b]

            # Get the ks and vs to update the cache with
            k_ = k[:, cur_lo_k:cur_lo_k+num_new_toks_k]
            v_ = v[:, cur_lo_v:cur_lo_v+num_new_toks_v]

            # Update the k and v cache
            self.k_cache[b, :, self.k_cache_pos[b]:self.k_cache_pos[b]+num_new_toks_k] = k_
            self.v_cache[b, :, self.v_cache_pos[b]:self.v_cache_pos[b]+num_new_toks_v] = v_

            # Update cur_lo positions in the cu_seqlens for the next iter
            cur_lo_k += num_new_toks_k
            cur_lo_v += num_new_toks_v

            # Update cache positions
            self.k_cache_pos[b] += num_new_toks_k
            self.v_cache_pos[b] += num_new_toks_v

        # Get entire kv cache to attend over
        return (
            torch.cat(
                [
                    self.k_cache[b, :, :self.k_cache_pos[b]]
                    for b in range(0, self.kv_cache_batch_size)
                ], dim=1
            ).clone(),
            torch.cat(
                [
                    self.v_cache[b, :, :self.v_cache_pos[b]]
                    for b in range(0, self.kv_cache_batch_size)
                ], dim=1
            ).clone()
        )
    
    
    def forward_(
            self,
            X: torch.Tensor,
            cu_seqlens: torch.Tensor,
            metrics: Metrics = dummy_metrics,
            timing: Timing = dummy_timing,
            losses: Losses = dummy_losses,
            cache: dict = dict(),
            inf_cfg: Optional[InfConfig] = None,
        ):

        # QKV projection
        # (H, bss, d_H)
        timing.start_span("tfm/qkv")
        q, k, v = self.proj_to_qkv(
            X,
            transpose=False
        )
        timing.end_span("tfm/qkv")

        # Apply norm
        if self.do_qk_norm:
            timing.start_span("tfm/qk_norm")
            q = self.q_norm(q)
            k = self.k_norm(k)
            timing.end_span("tfm/qk_norm")

        # Apply RoPE
        # Expects (H, bss, d_H)
        if self.use_rope:
            timing.start_span("tfm/rope")
            q = self.rotary_emb(
                q,
                cu_seqlens=cu_seqlens
            )
            k = self.rotary_emb(
                k,
                cu_seqlens=cu_seqlens
            )
            timing.end_span("tfm/rope")

        # KV cache
        k, v = self.handle_kv_cache(
            k=k, v=v, cu_seqlens=cu_seqlens, inf_cfg=inf_cfg
        )
            
        # gqa
        k = k.repeat_interleave(self.gqa_factor, dim=0)
        v = v.repeat_interleave(self.gqa_factor, dim=0)
        
        timing.start_span("tfm/sdpa")
        out = self.attention(
            q,
            k,
            v,
            cu_seqlens,
            inf_cfg,
        )
        timing.end_span("tfm/sdpa")

        timing.start_span("tfm/o_proj")
        out = self.o_proj(out)
        timing.end_span("tfm/o_proj")
        return out
    

    def num_params(self,):
        return get_num_params(self)
    
    
    def num_active(self,):
        return self.num_params()
        