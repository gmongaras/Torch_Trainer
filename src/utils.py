import json
import random
import torch
import os
from datetime import timedelta
import torch.distributed as dist
import contextvars
from contextlib import contextmanager
from torch import nn
from typing import Optional
import inspect




def init_hf_path():
    cache_path = open("HF_CACHE_PATH", "r").read().strip()
    assert os.path.exists(cache_path)
    os.environ["HF_HOME"] = cache_path
    os.environ["HF_HUB_CACHE"] = cache_path




def init_distributed():
    # Initializes the distributed backend which will take care of synchronization
    dist_url = "env://" # default

    # only works with torch.distributed.launch // torchrun
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            device_id=torch.device(f"cuda:{local_rank}"),
            rank=rank, 
            timeout=timedelta(seconds=1800))

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)

    # synchronizes all the threads to reach this point before moving on
    dist.barrier()
    
    print(f"Rank {rank}: Distributed initialized - world size {world_size}, local rank {local_rank}")
    
    return rank, world_size, local_rank




def get_distributed_info(is_distributed=True):
    if not is_distributed:
        return 0, 1, 0
    assert dist.is_initialized()
    
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    return rank, world_size, local_rank





def safe_int_div(x, y):
    assert x % y == 0
    return x // y




def get_vocab_size(tokenizer):
    # vocab_size can lie. Instead we get the length directly
    return max(list(tokenizer.vocab.values()))+1





def generate_random_cu_seqlens(
        bss, 
        max_per_seq_bss,
        min_per_seq_bss,
        seed=None,
        device=None,
    ):
    # Given a bss, create random cu seqlens across this sequence
    seqlens = [0]
    random.seed(seed)
    while seqlens[-1] < bss:
        cur_ = seqlens[-1]
        max_bss_ = min(max_per_seq_bss, bss-cur_)
        if max_bss_ < min_per_seq_bss:
            seqlens.append(cur_ + max_bss_)
        else:
            seqlens.append(cur_ + random.randint(min_per_seq_bss, max_bss_))
    return torch.tensor(seqlens, device=device, dtype=torch.int32)



def get_num_local_params(model):
    return int(sum(p.numel() for p in model if p.requires_grad))


def get_model_num_params(model):
    # Return the number of params in billions
    num_params = model.num_params()
    num_params_no_emb = model.num_params_no_emb()
    num_active = model.num_active()
    num_active_no_emb = model.num_active_no_emb()

    return (
        num_params / 1_000_000_000, # total
        num_params_no_emb / 1_000_000_000, # total no emb
        num_active / 1_000_000_000, # active
        num_active_no_emb / 1_000_000_000, # active no emb
    )






# Used to check if we are doing a forward pass
# normally or if we are doing a forward pass due to checkpointing
_is_checkpoint_recompute = contextvars.ContextVar(
    "is_checkpoint_recompute",
    default=False,
)
def is_checkpoint_recompute():
    return _is_checkpoint_recompute.get()

# Used to change the context when checkpointing
@contextmanager
def checkpoint_phase(is_recompute: bool):
    token = _is_checkpoint_recompute.set(is_recompute)
    try:
        yield
    finally:
        _is_checkpoint_recompute.reset(token)
def checkpoint_context_fn():
    return (
        checkpoint_phase(False),  # original forward
        checkpoint_phase(True),   # backward recomputation
    )
    
    
def get_num_params(module):
    def get_module_params(module):
        if type(module) == nn.Parameter:
            return module.numel()
        return sum(
            p.numel() 
            for p in module.parameters() 
            if p.requires_grad
        )
    return int(get_module_params(module))




def find_highest_step(
        step: Optional[str],
        model_save_path: str,
    ):
    # Find the highest step if the step is not specified
    if step is None:
        step_dirs = os.listdir(model_save_path)
        step_dirs_num = [int(step) for step in step_dirs if step.isnumeric()]
        if not step_dirs_num:
            raise RuntimeError("No checkpoint found with a valid step number")
        highest_step = max(step_dirs_num)
        step = highest_step
    step_str = str(step)
    return step_str





def disable_methods(base):
    def noop(*args, **kwargs):
        return None

    def decorator(cls):
        for name, method in inspect.getmembers(base, predicate=inspect.isfunction):
            if not name.startswith("__"):
                setattr(cls, name, noop)
        return cls

    return decorator


