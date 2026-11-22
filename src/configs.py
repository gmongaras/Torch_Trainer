from dataclasses import dataclass
from typing import Annotated, Literal, Optional, get_type_hints, get_origin
from mashumaro.mixins.json import DataClassJSONMixin
from mashumaro.types import Discriminator
    

from src.types import (
    NormType, 
    FFType, 
    AttentionType, 
    DatasetName,
    TokenizerName,
    LRSchedulerType,
    InfMode,
)


# Class that forces strict typing. This is so you
# don't make any stupid mistakes. If this isn't
# checked, the config can save/load incorrectly
# but silently, leading to undefined run behavior
class StrictDataClassJSONMixin(DataClassJSONMixin):
    def __post_init__(self):
        attrs = [a for a in dir(self) if not a.startswith('__') and not a.startswith('_')]
        types = get_type_hints(self)
        for a in attrs:
            v = getattr(self, a)
            # Skip callables
            if callable(v):
                continue
            t = types[a]
            # skip literals
            if get_origin(t) is Literal:
                continue
            assert isinstance(v, t), \
                f"type of {a} ({v}) is not {t.__name__}"


# Params specifically for attention
@dataclass
class AttentionConfig(StrictDataClassJSONMixin):
    attn_type: AttentionType
    qk_head_dim: int
    v_head_dim: int
    num_heads_q: int
    num_heads_kv: int
    do_qk_norm: bool = False
    use_rope: bool = True
    is_causal: bool = True
    checkpoint: bool = True
    layer_type: Literal["attn"] = "attn"
    
# Params for feedforward layer
@dataclass
class FFConfig(StrictDataClassJSONMixin):
    ff_type: FFType
    d_ff: int
    checkpoint: bool = True
    layer_type: Literal["ff"] = "ff"
    
# Params for a transformer layer
@dataclass
class TransformerLayerConfig(StrictDataClassJSONMixin):
    attn_config: Annotated[
        AttentionConfig,
        Discriminator(
            field="layer_type",
            include_supertypes=True,
        ),
    ]
    ff_config: Annotated[
        FFConfig,
        Discriminator(
            field="layer_type",
            include_supertypes=True,
        ),
    ]
    norm_type: NormType
    parallel_blocks: bool = False
    layer_type: Literal["tfm"] = "tfm"



# Params related to the architecture of a model
@dataclass
class ModelConfig(StrictDataClassJSONMixin):
    tokenizer_type: TokenizerName
    d_model: int
    layer_config: Annotated[
        TransformerLayerConfig,
        Discriminator(
            field="layer_type",
            include_supertypes=True,
        ),
    ]
    num_layers: int
    # Max per sequence length
    max_seqlen: int
    # Eps for layernorms
    layernorm_eps: Optional[float] = None
    # Used by the trainer when saving models
    checkpoint_step: Optional[int] = None
    # Used by the trainer to know how many
    # sequences and tokens this cp has been
    # trained for
    num_tokens_trained: Optional[int] = None
    num_sequences_trained: Optional[int] = None
    
    
# Params related to optimizing a model
@dataclass
class OptimConfig(StrictDataClassJSONMixin):
    # This is the batch size split across all
    # ranks. Note that this represents the flattened
    # sequence dimension. So a bss of 16K and
    # a max seqlen of 4K could fit a max of 4 sequences
    # but could also fit more if the sequences
    # are shorter.
    global_bss: int
    learning_rate: float
    grad_clipping_value: Optional[float]
    weight_decay: float
    beta1: float
    beta2: float
    eps: float
    warmup_steps: int
    num_steps: int
    num_steps_early_stop: Optional[int]
    lr_scheduler_type: LRSchedulerType
    grad_acc_steps: int = 1
    # Used by the trainer when saving models
    checkpoint_step: Optional[int] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.num_steps_early_stop is None:
            self.num_steps_early_stop = self.num_steps
    
    
# Params related to the trainer that will optimize a model 
@dataclass
class TrainerConfig(StrictDataClassJSONMixin):
    dataset_type: DatasetName
    run_name: str
    # None wandb_name default to
    # run_name
    wandb_name: Optional[str]
    log_steps: int
    use_amp: bool
    num_devices: int
    # Min per sequence length during training.
    # Prevents training on super short sequences
    # that are likely truncated at the end of the bss
    min_train_seqlen: int
    # None model_save_path defaults to
    # models/{run_name}/{step}/...
    model_save_path: Optional[str] = None
    starting_step: int = 0
    num_save_steps: int = 10_000
    include_step_in_checkpoint: bool = True
    # None load_checkpoint_path defaults to
    # "models/{run_name}/{step}/..."
    load_checkpoint_path: Optional[str] = None
    dataset_test_per: float = 0.1
    num_steps_test: int = 10_000
    test_loss: bool = True
    # True to enable timing spans
    enable_timing: bool = False
    # wandb id will be added for checkpointing
    wandb_id: Optional[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.wandb_name is None:
            self.wandb_name = self.run_name
        if self.model_save_path is None:
            self.model_save_path = f"model_ckpts/{self.run_name}"



# Config passed for inference
@dataclass
class InfConfig:
    # Are we running prefill or decode?
    inf_mode: InfMode
    # Update the cache (for each sequence)?
    # This is on for decode/generate tasks
    # and off for pure prefill tasks
    update_cache: list[bool]
    # Reset the KV cache?
    reset_cache: bool