from dataclasses import dataclass

from src.configs import (
    TrainerConfig,
    ModelConfig, 
    OptimConfig, 
    AttentionConfig,
    FFConfig,
    TransformerLayerConfig,
) 
from src.types import (
    FFType, 
    AttentionType,
    NormType,
    TokenizerName,
    LRSchedulerType,
    DatasetName
)


@dataclass
class TrainerConfigs:
    Tfm = TrainerConfig(
        dataset_type=DatasetName.FineWeb,
        run_name="Tfm",
        wandb_name=None,
        log_steps=10,
        use_amp=True,
        num_devices=2,
        min_train_seqlen=64,
        model_save_path=None,
        starting_step=0,
        num_save_steps=2_500,
        include_step_in_checkpoint=True,
        dataset_test_per=0.0005,
        num_steps_test=10_000,
        test_loss=True,
        enable_timing=True,
    )
    # Add more trainer configs here



@dataclass
class ModelConfigs:
    Tfm = ModelConfig(
        tokenizer_type=TokenizerName.Neox,
        d_model=1280,
        layer_config=TransformerLayerConfig(
            attn_config=AttentionConfig(
                attn_type=AttentionType.TORCH_SDPA,
                qk_head_dim=128,
                v_head_dim=128,
                num_heads_q=10,
                num_heads_kv=10,
                do_qk_norm=True,
                use_rope=True,
                is_causal=True,
                checkpoint=True
            ),
            ff_config=FFConfig(
                ff_type=FFType.SWIGLU_KERNEL,
                d_ff=3420,
                checkpoint=True
            ),
            norm_type=NormType.PRENORM
        ),
        num_layers=9,
        max_seqlen=2048,
        layernorm_eps=None
    )
    # Add more model configs here




@dataclass
class OptimConfigs:
    Tfm = OptimConfig(
        global_bss=128*2048,
        learning_rate=2.5e-4,
        grad_clipping_value=1.0,
        weight_decay=0.1,
        beta1=0.90,
        beta2=0.95,
        eps=1e-10,
        num_steps=25_000,
        num_steps_early_stop=None,
        warmup_steps=2_000,
        lr_scheduler_type=LRSchedulerType.WARMUP,
        grad_acc_steps=1,
    )
    # Add more optim configs here



def get_config(config_class, config_name):
    if not hasattr(config_class, config_name):
        raise NotImplementedError(
            f"config class {config_class} has no cfg {config_name}"
        )
    return getattr(config_class, config_name)
