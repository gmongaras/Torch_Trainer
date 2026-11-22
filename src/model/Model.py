import os
import torch
from torch import nn
from typing import Optional
import transformers
import json
from safetensors.torch import load_file

from src.configs import ModelConfig
from src.model.layers.TransformerLayer import TransformerLayer
from src.exp_configs import ModelConfigs
from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction
from src.model.fast_ce import fast_linear_cross_entropy
from src.utils import (
    get_vocab_size,
    get_num_params
)
from src.configs import (
    TransformerLayerConfig,
    InfConfig
)
from src.model.Metrics import Metrics, dummy_metrics
from src.model.Losses import Losses, dummy_losses
from src.model.Timing import Timing, dummy_timing
from src.utils import find_highest_step
from src.types import InfMode




if "DEBUG" in os.environ and not bool(int(os.environ["DEBUG"])):
    from src.model.LigerCE import LigerFusedLinearCrossEntropyFunction_fwd


layer_config_map = {
    TransformerLayerConfig: TransformerLayer,
}



def load_model_for_inf(
        model_save_path: str,
        step: Optional[int],
        device: Optional[str],
    ):
    step_str = find_highest_step(
        step,
        model_save_path,
    )
    
    # Load in the config
    base_pth = os.path.join(model_save_path, step_str)
    with open(os.path.join(base_pth, "model_cfg.json"), "r") as f:
        model_config = ModelConfig.from_json(json.load(f))
    
    # Load in the model
    print(f"Loading in checkpoint at `{base_pth}`")
    
    # Actual model to train
    model = Model(
        model_config=model_config
    )
    # Load in model checkpoint before DDP
    base_pth = os.path.join(model_save_path, step_str)
    model.load_state_dict(
        load_file(
            os.path.join(base_pth, f"model.safetensors"), 
            device=device
        ),
        strict=True,
        assign=True,
    )
    
    torch.cuda.empty_cache()
    model.eval()
    
    return model


# Interface just so I can use kwargs
# and for torch compile
def liger_forward(
        _input: torch.Tensor,
        weight: torch.Tensor,
        target: torch.Tensor,
        bias: Optional[bool] = None,
        ce_weight: Optional[float] = None,
        ignore_index: int = -100,
        lse_square_scale: float = 0.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        softcap: Optional[float] = None,
        return_z_loss: bool = False,
        accum_dtype: Optional[torch.dtype] = None,
        use_token_scaling: bool = False,
        return_token_accuracy: bool = False,
        return_predicted_tokens: bool = False,
    ) -> torch.Tensor:
    # fast path
    if (
        bias is None
        and ce_weight is None
        and lse_square_scale == 0.0
        and label_smoothing == 0.0
        and reduction == "mean"
        and softcap is None
        and not return_z_loss
        and not use_token_scaling
        and not return_token_accuracy
        and not return_predicted_tokens
    ):
        return fast_linear_cross_entropy(
            _input, weight, target, ignore_index=ignore_index,
        )
    loss, z_loss, token_accuracy, predicted_tokens = LigerFusedLinearCrossEntropyFunction.apply(
        _input,
        weight,
        target,
        bias,
        ce_weight,
        ignore_index,
        lse_square_scale,
        label_smoothing,
        reduction,
        softcap,
        return_z_loss,
        accum_dtype,
        use_token_scaling,
        return_token_accuracy,
        return_predicted_tokens,
    )
    return loss

# Read token from .env file
if os.path.exists(".env"):
    with open(".env", "r") as f:
        TOKEN = f.read().strip()
    # Faster downloads from hf
    os.environ["HF_TOKEN"] = TOKEN
else:
    TOKEN = None




class Model(nn.Module):
    def __init__(
            self,
            model_config: ModelConfig,
        ):
        super().__init__()
        
        self.tokenizer_type = model_config.tokenizer_type
        self.d_model = model_config.d_model
        self.layer_config = model_config.layer_config
        self.num_layers = model_config.num_layers
        self.max_seqlen = model_config.max_seqlen
        
        # Tokenizer
        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.tokenizer_type.value,
                token=TOKEN,
            )
        except OSError:
            raise FileNotFoundError("Token not found in .env file or user does not have access to tokenizer. Please add your Hugging Face token to the .env file.")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.pad_token = torch.tensor([self.tokenizer.pad_token_id])
        self.vocab_size = get_vocab_size(self.tokenizer)
        # Set max sequence length
        self.tokenizer.model_max_length = self.max_seqlen
        
        # Embedding
        self.emb = nn.Embedding(
            self.vocab_size,
            self.d_model,
        )
        
        # Stack of layers
        layer_type = layer_config_map[type(self.layer_config)]
        self.layers = nn.ModuleList([
            layer_type(
                model_config,
                self.layer_config,
                layer_num=layer_num,
            ) for layer_num in range(0, self.num_layers)
        ])
        
        # Unembedding
        self.unemb = nn.Linear(
            self.d_model,
            self.vocab_size,
            bias=False
        )
        
    
    def forward(
            self, 
            inputs: torch.Tensor, 
            cu_seqlens: torch.Tensor,
            return_loss: bool,
            target: Optional[torch.Tensor] = None,
            metrics: Metrics = dummy_metrics,
            losses: Losses = dummy_losses,
            timing: Timing = dummy_timing,
            # Arbitrary cache to write to/read from
            cache: dict = dict(),
            # Inference info. None for training
            inf_cfg: Optional[InfConfig] = None,
        ):
        
        # If doing infernece, we should not be in train mode
        if inf_cfg is not None:
            assert self.training == False
        decoding = (
            inf_cfg is not None
            and inf_cfg.inf_mode == InfMode.DECODE
        )
        
        if return_loss:
            assert target is not None, \
                "target must be passed if return_loss"
        
        # Make sure the input is flat and the the cumulative sequence lengths
        # exhaust the entire input sequence.
        assert len(inputs.shape) == 1
        assert len(cu_seqlens.shape) == 1
        assert cu_seqlens.dtype == torch.long
        assert cu_seqlens[0] == 0
        if not decoding:
            assert inputs.shape == (cu_seqlens[-1],)
        else:
            assert inputs.shape[0] == cu_seqlens.shape[0]-1
        
        # Embed the tokens
        timing.start_span("emb")
        inputs = self.emb(inputs)
        timing.end_span("emb")
        
        # Forward pass all layers
        for layer in self.layers:
            timing.start_span("layer fwd")
            inputs = layer(
                X=inputs, 
                cu_seqlens=cu_seqlens,
                metrics=metrics,
                timing=timing,
                losses=losses,
                cache=cache,
                inf_cfg=inf_cfg,
            )
            timing.end_span("layer fwd")
        
        # If return_loss, do not construct the massive
        # unembed tensor.
        if return_loss:
            timing.start_span("unemb")
            timing.start_span("ce")
            losses.add_loss("ntp", liger_forward(
                _input=inputs,
                weight=self.unemb.weight,
                target=target,
                ignore_index=-100,
                reduction="mean",
                accum_dtype=torch.float32
            ))
            timing.end_span("ce")
            timing.end_span("unemb")
            return
        
        # If decoding, just decode the last token
        timing.start_span("unemb")
        if decoding and cu_seqlens.shape[0]-1 != inputs.shape[0]:
            unemb = self.unemb(inputs[-1])
        else:
            unemb = self.unemb(inputs)
        timing.end_span("unemb")
        return unemb
    
    
    
    def num_params_no_emb(self):
        return sum([
            self.layers[i].num_params()
            for i in range(0, self.num_layers)
        ])
        
    def num_active_no_emb(self):
        return sum([
            self.layers[i].num_active()
            for i in range(0, self.num_layers)
        ])
        
    def num_params_emb(self):
        return (
            get_num_params(self.emb)
            + get_num_params(self.unemb)
        )
        
    def num_params(self):
        return (
            self.num_params_emb()
            + self.num_params_no_emb()
        )
        
    def num_active(self):
        return (
            self.num_params_emb()
            + self.num_active_no_emb()
        )
            
            
            
            


    