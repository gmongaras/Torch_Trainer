import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from src.configs import InfConfig
from src.utils import checkpoint_context_fn
from typing import Optional
from src.model.Metrics import Metrics, dummy_metrics
from src.model.Timing import Timing, dummy_timing
from src.model.Losses import Losses, dummy_losses
from abc import ABC, abstractmethod



class Layer(nn.Module, ABC):
    def __init__(
            self,
        ):
        super().__init__()
        
    @abstractmethod
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
        pass
        
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
        if hasattr(self, "checkpoint") and self.checkpoint:
            return checkpoint(
                self.forward_,
                X,
                cu_seqlens,
                metrics,
                timing,
                losses,
                inf_cfg=inf_cfg,
                use_reentrant=False,
                context_fn=checkpoint_context_fn
            )
        return self.forward_(
            X,
            cu_seqlens,
            metrics,
            timing,
            losses,
            cache,
            inf_cfg=inf_cfg,
        )
    
    @abstractmethod
    def num_params(self):
        pass
    
    @abstractmethod
    def num_active(self):
        pass
    
