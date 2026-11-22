from typing import Optional

import torch
from src.model.Metrics import Metric
from torch.distributed import ReduceOp
import torch.distributed as dist
from src.utils import (
    is_checkpoint_recompute,
    get_distributed_info,
    disable_methods,
)


class Span:
    
    def __init__(
            self, 
            start, 
            end, 
            elapsed
        ):
        self.start_times: list[float] = list()
        self.end_times: list[float] = list()
        self.elapsed_times: list[float] = list()
        self.total_time: int = 0 # in ms
        self.add_span(start, end, elapsed)
    
    def add_span(self, start, end, elapsed):
        self.start_times.append(start)
        self.end_times.append(end)
        self.elapsed_times.append(elapsed)
        self.total_time += elapsed
        
    def get_total_time(self, in_seconds=True):
        scale = 1000 if in_seconds else 1
        return self.total_time / scale
        
    def get_avg_time(self, in_seconds=True):
        scale = 1000 if in_seconds else 1
        return (self.total_time / scale) / len(self.elapsed_times)


class Timing:
    def __init__(self, enabled):
        self.enabled = enabled
        
        # Dictionary of span start times
        self.cur_spans = dict()
        
        # Dictionary of span results
        self.span_results = dict()
    
    # Start a new span. Name should not currenlty be active
    def start_span(self, span_name):
        if not self.enabled or is_checkpoint_recompute():
            return
        assert span_name not in self.cur_spans
        self.cur_spans[span_name] = torch.cuda.Event(
            enable_timing=True
        )
        self.cur_spans[span_name].record()
        
    def end_span(self, span_name):
        if not self.enabled or is_checkpoint_recompute():
            return
        assert span_name in self.cur_spans
        start = self.cur_spans.pop(span_name)
        end = torch.cuda.Event(
            enable_timing=True
        )
        end.record()
        end.synchronize()
        elapsed = start.elapsed_time(end)
        try:
            self.span_results[span_name].add_span(
                start,
                end,
                elapsed,
            )
        except KeyError:
            self.span_results[span_name] = Span(
                start,
                end,
                elapsed
            )
            
    def get_span_total(self, span_name):
        if not self.enabled:
            return 0
        assert span_name not in self.cur_spans
        return self.span_results[span_name].get_total_time()
    
    def get_span_avg(self, span_name):
        if not self.enabled:
            return 0
        assert span_name not in self.cur_spans
        return self.span_results[span_name].get_avg_time()
          
    @staticmethod
    def _maybe_get_metric(val, return_as_metric):
        return (
            Metric(val, ReduceOp.AVG)
            if return_as_metric
            else val
        )
        
    # Different ranks can have different keys. Ensure
    # all ranks have all keys
    def _add_dist_keys(self):
        if not dist.is_initialized():
            return list(self.span_results.keys())
        _, world_size, local_rank = get_distributed_info()
        
        # Get all keys form all ranks
        all_keys = set()
        for rank in range(0, world_size):
            obj = list(self.span_results.keys())
            torch.distributed.broadcast_object_list(
                object_list=obj,
                src=rank,
                device=torch.device(f"cuda:{local_rank}")
            )
            all_keys.update(obj)
            
        # Add null spans for missing keys
        missing_keys = all_keys - set(list(self.span_results.keys()))
        for key in missing_keys:
            self.span_results[key] = Span(
                0,
                0,
                0
            )
            
    def get_all_span_totals(
            self,
            skip_spans: list[str] = [],
            return_as_metrics: bool = False,
        ):
        if not self.enabled:
            return dict()
        
        # Need to ensure we have all keys
        self._add_dist_keys()
        
        assert len(self.cur_spans) == 0
        return {
            span_name: self._maybe_get_metric(
                span.get_total_time(),
                return_as_metrics
            )
            for span_name, span in self.span_results.items()
            if span_name not in skip_spans
        }
    
    def get_all_span_avgs(
            self,
            skip_spans: list[str] = [],
            return_as_metrics: bool = False,
        ):
        if not self.enabled:
            return dict()
        
        # Need to ensure we have all keys
        self._add_dist_keys()
        
        assert len(self.cur_spans) == 0
        return {
            span_name: self._maybe_get_metric(
                span.get_avg_time(),
                return_as_metrics
            )
            for span_name, span in self.span_results.items()
            if span_name not in skip_spans
        }
        
     
# Dummy timer with no functionality
@disable_methods(Timing)
class DummyTiming(Timing):
    def __init__(self):
        return
dummy_timing = DummyTiming()
        
        