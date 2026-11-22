import torch
from torch import nn
from src.model.layers.Layer import Layer
from src.types import FFType
from src.configs import (
    InfConfig,
    ModelConfig,
    FFConfig,
)
from src.utils import get_num_params
from typing import Optional
from src.model.Metrics import Metrics, dummy_metrics
from src.model.Timing import Timing, dummy_timing
from src.model.Losses import Losses, dummy_losses



class FeedForward(Layer):
    def __init__(
            self,
            model_config: ModelConfig,
            ff_config: FFConfig,
            layer_num: int,
        ):
        super().__init__()
        
        self.model_config = model_config
        self.d_model = model_config.d_model
        self.d_ff = ff_config.d_ff
        self.ff_type = ff_config.ff_type
        self.checkpoint = ff_config.checkpoint
        self.layer_num = layer_num
        
        assert type(self.ff_type) is FFType
        
        # Kernel
        if self.ff_type == FFType.SWIGLU_KERNEL:
            from xformers.ops.swiglu_op import SwiGLU
            self.swiglu = SwiGLU(
                in_features=self.d_model,
                hidden_features=self.d_ff,
                out_features=self.d_model,
                bias=False,
                _pack_weights=True
            )
        else:
            # Init weights
            self.up_proj = nn.Linear(
                self.d_model,
                self.d_ff*2 if self.ff_type == FFType.SWIGLU else self.d_ff,
                bias=False
            )
            self.down_proj = nn.Linear(
                self.d_ff,
                self.d_model,
                bias=False
            )
        
    
    def forward_swiglu(self, X: torch.Tensor):
        gate, up = self.up_proj(X).chunk(2, -1)
        return self.down_proj(up*torch.nn.functional.silu(gate))
    
    def forward_gelu(self, X: torch.Tensor):
        return self.down_proj(torch.nn.functional.silu(self.up_proj(X)))
    
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
        if timing is None:
            timing = dummy_timing
        timing.start_span("ff body")
        if self.ff_type == FFType.SWIGLU:
            out = self.forward_swiglu(X)
        elif self.ff_type == FFType.SWIGLU_KERNEL:
            out = self.swiglu(X)
        elif self.ff_type == FFType.GELU:
            out = self.forward_gelu(X)
        else:
            assert False
        timing.end_span("ff body")
        return out
    
    
    def num_params(self):
        return get_num_params(self)
    
    def num_active(self):
        return self.num_params()
            
            
            
            
