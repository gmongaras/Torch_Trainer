from enum import Enum


# Different types of norm placements
class NormType(Enum):
    PRENORM = 1
    POSTNORM = 2
    BOTH = 3

# Different types of FF layers
class FFType(Enum):
    SWIGLU = 1
    SWIGLU_KERNEL = 2
    GELU = 3

# Different types of ways to apply attention
class AttentionType(Enum):
    TORCH_SDPA = 1
    

# Different datasets that are supported
# Note that the % will split into dataset%name
class DatasetName(str, Enum):
    ThePile = "gmongaras/EleutherAI_the_pile_deduplicated"
    SlimPJ = "gmongaras/SlimPajama-627B_Reupload"
    FineWeb = "HuggingFaceFW/fineweb%sample-350BT"
    Dummy = "gmongaras/dummy_text_dataset"
    

# Different tokenizers that are supported
class TokenizerName(str, Enum):
    Qwen3 = "Qwen/Qwen3-0.6B"
    Neox = "EleutherAI/gpt-neox-20b"
    Llama = "meta-llama/Llama-3.1-8B"


# Different optimizer lr schedulers
class LRSchedulerType(Enum):
    # Constant learning rate through all of trianing
    CONSTANT = 1
    # Warmup then linear decay
    LINEAR = 2
    # Warmup then constant
    WARMUP = 3


# Run the model for prefill or inference
class InfMode(Enum):
    PREFILL = 1
    DECODE = 2