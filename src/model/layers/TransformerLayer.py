import torch
from torch import nn
from typing import Optional

from src.types import NormType
from src.configs import (
    ModelConfig,
    TransformerLayerConfig,
    FFConfig,
    AttentionConfig
)
from src.utils import get_num_params
from src.model.layers.FeedForward import FeedForward
from src.model.layers.Attention import Attention
from src.model.layers.FusedRMSNorm import FusedRMSNorm
from src.model.Metrics import Metrics, dummy_metrics
from src.model.Timing import Timing, dummy_timing
from src.model.Losses import Losses, dummy_losses
from src.configs import InfConfig



class TransformerLayer(nn.Module):
    def __init__(
            self,
            model_config: ModelConfig,
            layer_config: TransformerLayerConfig,
            layer_num: int,
        ):
        super().__init__()
        
        self.model_config = model_config
        self.layer_config = layer_config
        self.d_model = model_config.d_model
        self.ff_config = layer_config.ff_config
        self.attn_config = layer_config.attn_config
        self.norm_type = layer_config.norm_type
        self.layernorm_eps = model_config.layernorm_eps
        self.parallel_blocks = layer_config.parallel_blocks
        self.layer_num = layer_num
        
        # Attention and ff layers
        self.attention = Attention(
            self.model_config,
            self.attn_config,
            self.layer_num,
        )
        self.feedforward = FeedForward(
            self.model_config,
            self.ff_config,
            self.layer_num,
        )
        
        # Norms
        self.prenorm = self.norm_type in (NormType.PRENORM, NormType.BOTH)
        self.postnorm = self.norm_type in (NormType.POSTNORM, NormType.BOTH)
        self.norm_attn = FusedRMSNorm(self.d_model, eps=self.layernorm_eps)
        self.norm_ff = FusedRMSNorm(self.d_model, eps=self.layernorm_eps)
        
        # Cached streams
        if self.parallel_blocks:
            self._attn_stream = None
            self._ff_stream = None
            self._stream_device = None
        
    def attn(
            self, 
            X: torch.Tensor,
            cu_seqlens: torch.Tensor,
            metrics: Metrics = dummy_metrics,
            timing: Timing = dummy_timing,
            losses: Losses = dummy_losses,
            cache: dict = dict(),
            inf_cfg: Optional[InfConfig] = None,
        ):
        timing.start_span("attn")
        X = self.attention(
            self.norm_attn(X) if self.prenorm else X,
            cu_seqlens,
            metrics,
            timing,
            losses,
            cache,
            inf_cfg,
        ) + (X if not self.parallel_blocks else 0)
        if self.postnorm:
            X = self.norm_attn(X)
        timing.end_span("attn")
        return X
    
    def ff(
            self, 
            X: torch.Tensor,
            cu_seqlens: torch.Tensor,
            metrics: Metrics = dummy_metrics,
            timing: Timing = dummy_timing,
            losses: Losses = dummy_losses,
            cache: dict = dict(),
            inf_cfg: Optional[InfConfig] = None,
        ):
        timing.start_span("ff")
        X = self.feedforward(
            self.norm_ff(X) if self.prenorm else X,
            cu_seqlens,
            metrics,
            timing,
            losses,
            cache,
            inf_cfg,
        ) + (X if not self.parallel_blocks else 0)
        if self.postnorm:
            X = self.norm_ff(X)
        timing.end_span("ff")
        return X
    
    
    def parallel_blocks_(
            self, 
            X, 
            cu_seqlens,
            metrics: Metrics = dummy_metrics,
            timing: Timing = dummy_timing,
            losses: Losses = dummy_losses,
            cache: dict = dict(),
            inf_cfg: Optional[InfConfig] = None,
        ):
        if not X.is_cuda:
            return (
                self.attn(
                    X, 
                    cu_seqlens,
                    metrics,
                    timing,
                    losses,
                    cache,
                    inf_cfg,
                )
                + self.ff(
                    X, 
                    cu_seqlens,
                    metrics,
                    timing,
                    losses,
                    cache,
                    inf_cfg,
                )
                + X
            )
            
        # Get streams to run attn and ff on
        if self._stream_device != X.device:
            self._attn_stream = torch.cuda.Stream(device=X.device)
            self._ff_stream = torch.cuda.Stream(device=X.device)
            self._stream_device = X.device
            
        # Current stream X and cu_seqlens operate on
        current_stream = torch.cuda.current_stream(X.device)
            
        # Wait for X and cu_seqlens to be ready
        self._attn_stream.wait_stream(current_stream)
        self._ff_stream.wait_stream(current_stream)
        
        # Do attn and ff in parallel on different streams
        with torch.cuda.stream(self._attn_stream):
            attn = self.attn(X, cu_seqlens, inf_cfg)
        with torch.cuda.stream(self._ff_stream):
            ff = self.ff(X, cu_seqlens, inf_cfg)
        
        # Wait for attn and ff streams to complete
        current_stream.wait_stream(self._attn_stream)
        current_stream.wait_stream(self._ff_stream)
        
        return attn + ff + X
        
        
    def forward(
            self, 
            X: torch.Tensor,
            cu_seqlens: torch.Tensor,
            metrics: Metrics = dummy_metrics,
            timing: Timing = dummy_timing,
            losses: Losses = dummy_losses,
            cache: dict = dict(),
            inf_cfg: Optional[InfConfig] = None,
        ):
        # Parallel blocks
        if self.parallel_blocks:
            return self.parallel_blocks_(
                X, 
                cu_seqlens,
                metrics,
                timing,
                losses,
                cache,
                inf_cfg,
            )
        # Sequential blocks
        return self.ff(
            self.attn(
                X, 
                cu_seqlens,
                metrics,
                timing,
                losses,
                cache,
                inf_cfg,
            ), 
            cu_seqlens,
            metrics,
            timing,
            losses,
            cache,
            inf_cfg,
        )
    
    
    def num_params(self):
        return (
            self.attention.num_params()
            + self.feedforward.num_params()
            + get_num_params(self.norm_attn)
            + get_num_params(self.norm_ff)
        )
        
    def num_active(self):
        return (
            self.attention.num_active()
            + self.feedforward.num_active()
            + get_num_params(self.norm_attn)
            + get_num_params(self.norm_ff)
        )