import torch
from dataclasses import dataclass
from src.model.Metrics import Metrics
from torch.distributed import ReduceOp
from src.utils import disable_methods


@dataclass
class Loss:
    value: torch.Tensor
    weight: float
    
    def __post_init__(self):
        assert type(self.value) == torch.Tensor
        assert self.value.numel() == 1
        if type(self.weight) in [int, bool]:
            self.weight = float(self.weight)
        assert type(self.weight) == float



class Losses:
    def __init__(self):
        self.losses: dict[str, Loss] = dict()
        
    def add_loss(
            self,
            loss_name: str,
            loss_value: torch.Tensor,
            loss_weight: float = 1.0,
        ):
        self.losses[loss_name] = Loss(loss_value, loss_weight)
        
    def __len__(self,):
        return len(self.losses)
        
    def get(self, loss_name):
        # Get a single loss
        return self.losses[loss_name].value
    
    def get_weighted(self, loss_name):
        # Weighted single loss
        return self.losses[loss_name].weight*self.losses[loss_name].value
        
    def get_final_loss(self,):
        # Get weighted sum of losses
        return sum([
            self.get_weighted(loss_name)
            for loss_name in self.losses.keys()
        ])
        
    @torch.no_grad()
    def log(
            self, 
            metrics: Metrics, 
            section_name: str
        ):
        # Used to log the individual losses
        # to a given set of metrics
        for loss_name, loss in self.losses.items():
            metrics.add_metric(
                section_name,
                loss_name,
                loss.value,
                ReduceOp.AVG
            )
            


# Dummy losses with no functionality
@disable_methods(Losses)
class DummyLosses(Losses):
    def __init__(self):
        return
dummy_losses = DummyLosses()
            
            