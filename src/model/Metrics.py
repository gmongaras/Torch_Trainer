import zlib

import torch
import wandb
from typing import Optional
from dataclasses import dataclass
from torch.distributed import ReduceOp
import numpy as np
from src.utils import (
    get_distributed_info,
    disable_methods
)


@dataclass
class Metric:
    metric: torch.Tensor
    # Reduce op is used when log_dist is called.
    # If None, no reduction is done across ranks
    # and only the rank 0 value is used.
    reduce_op: Optional[ReduceOp] = None
    is_histogram: bool = False
    
    def __post_init__(self,):
        # Convert integers, floats, and booleans into tensors
        if type(self.metric) == int:
            self.metric = torch.tensor(self.metric, dtype=torch.int32)
        elif type(self.metric) == float:
            self.metric = torch.tensor(self.metric, dtype=torch.float32)
        elif type(self.metric) == bool:
            self.metric = torch.tensor(self.metric, dtype=torch.bool)
        assert type(self.metric) == torch.Tensor
        assert (
            self.reduce_op is None 
            or isinstance(self.reduce_op, ReduceOp)
        )
        
        if self.is_histogram:
            assert len(self.metric.shape) == 1



class Metrics:
    # Class just stores metrics for a forward pass
    # which are then plotted in wandb.
    def __init__(self, *section_names: list[str]):
        self.metrics: dict[str, Metric] = dict()
        self.section_names: list[str] = section_names
        for section_name in section_names:
            assert type(section_name) == str
        self.logged: bool = False
        
    @torch.no_grad()
    def add_metric(
            self, 
            section_name: str, 
            metric_name: str, 
            metric_value: torch.Tensor,
            reduce_op: Optional[ReduceOp] = None,
            is_histogram = False
        ):
        assert section_name in self.section_names, \
            f"{section_name} is not a known section."
        self.metrics[f"{section_name}/{metric_name}"] = Metric(
            metric=metric_value.detach().cpu(),
            reduce_op=reduce_op,
            is_histogram=is_histogram,
        )
        
    @torch.no_grad()
    def add_metrics(
            self,
            section_name: str,
            **section_values: dict[str, Metric]
        ):
        for metric_name, value in section_values.items():
            # Handle the non distributed case
            if type(value) == torch.Tensor:
                reduce_op = None
            elif isinstance(value, Metric):
                reduce_op = value.reduce_op
                value = value.metric
            else:
                assert type(value) in [list, tuple]
                assert len(value) == 2
                value, reduce_op = value
            self.add_metric(
                section_name, 
                metric_name, 
                value,
                reduce_op,
            )
            
    @torch.no_grad()
    def _log(self, step):
        assert not self.logged, \
            "Metrics were already logged to wandb"
        self.logged = True
        if len(self.metrics) > 0:
            wandb.log(
                data=self.metrics,
                step=step,
            )
        
    @torch.no_grad()
    def log_dist(self, step):
        # Same as log but mean aggregates over all
        # ranks and then logs on rank 0
        rank, world_size, local_rank = get_distributed_info()
        device = f"cuda:{local_rank}"
        
        # Sort by key so we get our ops corret
        self.metrics = dict(sorted(self.metrics.items()))
        
        ##### Sloptimized version of the above and reduction in the metric update loop
        # Ranks compare a hash of their metric names and
        # only exchange the full list when those hashes disagree.
        if world_size > 1:
            # Get all reducible metrics
            reducible = [k for k, v in self.metrics.items() if v.reduce_op is not None]
            # Hash the metric names into one integer.
            sig = torch.tensor(
                [zlib.adler32("\n".join(reducible).encode())],
                dtype=torch.int64, device=device,
            )
            # Collect that hash from every rank.
            gathered = [torch.empty_like(sig) for _ in range(world_size)]
            torch.distributed.all_gather(gathered, sig)
            # A different hash means the name lists disagree.
            if any(int(g) != int(sig) for g in gathered):
                # Send each rank's names and assert they match.
                for rank in range(world_size):
                    rank_keys = list(reducible)
                    torch.distributed.broadcast_object_list(
                        rank_keys, src=rank, device=torch.device(device),
                    )
                    assert rank_keys == reducible
        # One reduce per op and dtype.
        reduced = {}
        groups = {}
        # Group metric names that share a reduce op and dtype.
        for key, metric in self.metrics.items():
            if metric.reduce_op is not None:
                groups.setdefault((metric.reduce_op, metric.metric.dtype), []).append(key)
        # Concatenate each group, reduce it once, then split it back apart.
        for (op, _), keys in groups.items():
            # Cat the group together across ranks
            packed = torch.cat([
                self.metrics[k].metric.reshape(-1) for k in keys
            ]).to(device)
            # Do the reduction
            torch.distributed.reduce(packed, dst=0, op=op)
            # split matrics
            packed = packed.cpu()
            offset = 0
            for key in keys:
                n = self.metrics[key].metric.numel()
                reduced[key] = packed[offset:offset + n].reshape(self.metrics[key].metric.shape)
                offset += n
        #####

        # Iter over all metrics
        for metric_name, metric_value in self.metrics.items():
            metric_val, dist_op = metric_value.metric, metric_value.reduce_op
            if dist_op is not None:
                metric_val = reduced[metric_name]
            # Replace vectors with histograms
            if metric_value.is_histogram:
                assert len(metric_val.shape) == 1
                bins = np.arange(metric_val.shape[0]+1, dtype=np.float32) - 0.5
                self.metrics[metric_name] = (
                    wandb.Histogram(
                        np_histogram=(
                            metric_val.numpy(),
                            bins,
                        )
                    )
                )
            else:
                self.metrics[metric_name] = (
                    wandb.Histogram(metric_val.cpu().double().numpy())
                    if metric_val.numel() > 1 and (metric_val.min() != metric_val.max())
                    else metric_val.cpu()
                )
        if rank == 0:
            self._log(step)





# Dummy metrics with no functionality
@disable_methods(Metrics)
class DummyMetrics(Metrics):
    def __init__(self):
        self.section_names = []
dummy_metrics = DummyMetrics()