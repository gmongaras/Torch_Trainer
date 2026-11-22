import torch
from torch import nn
import os
import wandb
from tqdm import tqdm
from contextlib import nullcontext
import sys
from safetensors.torch import save_file, load_file
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from typing_extensions import Self
from typing import Optional
import json
import time

from src.distributed_helpers import is_main_process
from src.model.Model import Model
from src.utils import (
    safe_int_div,
    init_distributed,
    get_model_num_params,
    find_highest_step,
)
from src.configs import (
    TrainerConfig,
    ModelConfig, 
    OptimConfig, 
) 
from src.types import LRSchedulerType
from src.dataloader import BSSDataloader
from src.configs import TrainerConfig
from src.model.Metrics import Metric, Metrics
from src.model.Losses import Losses
from torch.distributed import ReduceOp
from src.model.Timing import Timing


WANDB_PROJECT = "Torch_Trainer"
TORCH_CPU = torch.device("cpu")







def get_scheduler(
        optim_config: OptimConfig,
        optimizer: torch.optim.Optimizer, 
        warmup_steps: int, 
        total_steps: int
    ):
    # Map different lr scheduler types. The lambda
    # maps the step to a value that is multiplied by the lr
    # lr_new = lr_old * lr_lambda(step)
    if optim_config.lr_scheduler_type == LRSchedulerType.CONSTANT:
        lr_lambda = lambda step: 1.0
    elif optim_config.lr_scheduler_type == LRSchedulerType.LINEAR:
        lr_lambda = lambda step: (
            # Warmup
            step/warmup_steps if step < warmup_steps
            # Decrease from 1 to 0 from warmup_steps to total_steps
            else (1.0 - (step - warmup_steps) / (total_steps - warmup_steps))
        )
    elif optim_config.lr_scheduler_type == LRSchedulerType.WARMUP:
        lr_lambda = lambda step: (
            # Warmup
            step/warmup_steps if step < warmup_steps
            # Constant after warmup
            else 1.0
        )

    # Create the scheduler
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=lr_lambda
    )



class Trainer():
    def __init__(
            self, 
            model_config: ModelConfig,
            optim_config: OptimConfig,
            trainer_config: TrainerConfig,
            # True to load in a checkpoint, False
            # to reinitialize the model weights
            load_checkpoint_on_init: bool = False,
        ):
        self.trainer_config = trainer_config
        self.optim_config = optim_config
        self.model_config = model_config
        self.dataset_type = trainer_config.dataset_type
        self.run_name = trainer_config.run_name
        self.wandb_name = trainer_config.wandb_name
        self.log_steps = trainer_config.log_steps
        self.use_amp = trainer_config.use_amp
        self.num_devices = trainer_config.num_devices
        self.min_train_seqlen = trainer_config.min_train_seqlen
        self.model_save_path = trainer_config.model_save_path
        self.num_save_steps = trainer_config.num_save_steps
        self.dataset_test_per = trainer_config.dataset_test_per
        self.num_steps_test = trainer_config.num_steps_test
        self.test_loss = trainer_config.test_loss
        self.enable_timing = self.trainer_config.enable_timing
        self.grad_acc_steps = optim_config.grad_acc_steps
        
        assert self.min_train_seqlen < self.model_config.max_seqlen
        
        # Starting step is whatever the model step is
        # Plus 1 so we go to the next step
        self.starting_step = (
            0
            if model_config.checkpoint_step is None
            else model_config.checkpoint_step+1
        )
        
        # Initialize distributed env
        self.rank, self.world_size, self.local_rank = init_distributed()
        assert self.world_size == self.num_devices
        
        # Update the save paths
        self.update_save_paths()
        
        # Actual model to train
        self.model = Model(
            model_config=model_config
        )
        # Load in model checkpoint before DDP
        if load_checkpoint_on_init:
            self.model.load_state_dict(
                load_file(
                    self.model_save_path_model, 
                    device="cpu"
                ),
                strict=True,
                assign=True
            )
        
        # bss for this rank
        self.global_bss = self.optim_config.global_bss
        self.local_bss = safe_int_div(
            self.global_bss, 
            self.world_size
        )

        # Get number of params.
        (
            num_params,
            num_params_no_emb,
            num_active,
            num_active_no_emb
        ) = get_model_num_params(self.model)
        if is_main_process():
            print(f"Total number of params: {num_params:.2f}B")
            print(f"Total number of params no emb: {num_params_no_emb:.2f}B")
            print(f"Number of active params: {num_active:.2f}B")
            print(f"Number of active params no emb: {num_active_no_emb:.2f}B")

        # Initialize DDP model
        self.model = DDP(
            self.model.cuda(), 
            device_ids=[self.local_rank], 
            find_unused_parameters=False
        )
        
        # Optimizer
        optim_params = [{
            "params": list(self.model.parameters()),
            "name": "dense",
            "lr": self.optim_config.learning_rate,
        },]
        self.optimizer = torch.optim.AdamW(
            optim_params,
            lr=self.optim_config.learning_rate, 
            betas=(
                self.optim_config.beta1, 
                self.optim_config.beta2
            ), 
            weight_decay=self.optim_config.weight_decay, 
            eps=self.optim_config.eps,
            fused=True,
        )
        if load_checkpoint_on_init:
            def keep_fused(_optimizer, state_dict):
                for group in state_dict["param_groups"]:
                    group["fused"] = True
                    group["foreach"] = False
            self.optimizer.register_load_state_dict_pre_hook(keep_fused)
            self.optimizer.load_state_dict(
                torch.load(
                    self.model_save_path_optim, 
                    map_location=f"cuda:{self.local_rank}",
                    mmap=True
                )
            )

        # LR Scheduler
        self.scheduler = get_scheduler(
            self.optim_config,
            self.optimizer, 
            warmup_steps=self.optim_config.warmup_steps, 
            total_steps=self.optim_config.num_steps
        )
        if load_checkpoint_on_init:
            self.scheduler.load_state_dict(
                torch.load(
                    self.model_save_path_scheduler, 
                    map_location=f"cuda:{self.local_rank}",
                    mmap=True
                )
            )

        # Automatic mixed precision grad scaler
        if self.use_amp:
            self.grad_scaler = torch.amp.GradScaler(f"cuda:{self.local_rank}")
            if load_checkpoint_on_init:
                self.grad_scaler.load_state_dict(
                    torch.load(
                        self.model_save_path_grad_scaler,
                        map_location=f"cuda:{self.local_rank}",
                        mmap=True
                    )
                )
        
        # Wandb id from the trainer config. None on
        # init, likely not None when loading a checkpoint
        self.wandb_id = self.trainer_config.wandb_id
        
        # Base model ref for DDP
        self.model_ref = self.model.module 

        # Number of sequences and tokens we have trained for
        self.num_tokens = torch.tensor(
            (
                0
                if model_config.checkpoint_step is None
                else model_config.num_tokens_trained
            ),
            dtype=torch.long,
            device=f"cuda:{self.local_rank}"
        )
        self.num_sequences = torch.tensor(
            (
                0
                if model_config.checkpoint_step is None
                else model_config.num_sequences_trained
            ),
            dtype=torch.long,
            device=f"cuda:{self.local_rank}"
        )
        
        
        
        
    def __call__(self):
        self.train_model()
        
        
        
        
        
    def train_model(self):
        # Load in dataset
        dataset = BSSDataloader(
            dataset_name=self.dataset_type,
            local_bss=self.local_bss,
            max_seqlen=self.model_ref.max_seqlen,
            min_train_seqlen=self.min_train_seqlen,
            tokenizer=self.model_ref.tokenizer,
            dataset_test_per=self.dataset_test_per,
            max_buffer_size=4,
            num_workers=4,
            rank=self.rank,
            world_size=self.world_size,
            is_training=True
        )
        
        # Train mode
        self.model.train()
        
        # Initialize wandb run
        if is_main_process():
            wandb.init(
                project=WANDB_PROJECT,
                name=self.wandb_name,
                notes=None,
                # Resume training if checkpoint exists
                resume="must" if self.wandb_id is not None else None,
                id=self.wandb_id,
                config={
                    "model_config": self.model_config.to_dict(),
                    "trainer_config": self.trainer_config.to_dict(),
                    "optim_config": self.optim_config.to_dict(),
                }
            )
            
            # Save wandb run id
            self.wandb_id = wandb.run.id
    
        
        # Wandb metrics
        batch_loss = 0
        batch_ntp = 0
        num_nan_grad_steps = 0
        num_nan = 0
        avg_time_per_fwd = 0
        avg_time_per_iter = 0
        avg_time_per_bwd = 0
        ema_weight = 0.9
        def update_ema(cur_ema, new_val):
            return ema_weight*cur_ema + (1-ema_weight)*new_val
        
        # Training loop. Only tqdm on rank 0
        for step in (
                tqdm(
                    range(self.starting_step+1, self.optim_config.num_steps+1),
                    initial=self.starting_step+1,
                    total=self.optim_config.num_steps+1
                ) if is_main_process()
                else range(self.starting_step+1, self.optim_config.num_steps+1)
            ):
            # Timing metrics for this step
            timing = Timing(self.enable_timing)
            
            timing.start_span("step")
            
            # Zero gradients
            self.optimizer.zero_grad()

            # Iter over grad acc steps
            grad_acc_loss = 0.0
            grad_acc_ntp = 0.0
            grad_acc_timing = dict()
            for grad_acc_step in range(self.grad_acc_steps):
                timing.start_span("grad acc step")

                # Get next local batch
                batch = dataset.sample_batch()
                
                # Metrics for this batch
                metrics = Metrics(
                    "General",
                    "Losses",
                    "Timing",
                    "Grad_Norm"
                )
                
                # Losses for this batch
                losses = Losses()
                    
                # Get input and labels
                input_ids = batch.input_ids.to(self.model.device)
                labels = batch.labels.to(self.model.device)
                cu_seqlens = batch.cu_seqlens.to(self.model.device)
            
                
                timing.start_span("fwd")
                
                with (
                        torch.autocast(
                            device_type='cuda', 
                            dtype=torch.bfloat16,
                        ) 
                        if self.use_amp 
                        else nullcontext()
                    ):
                    
                    
                    # Forward pass without constructing the
                    # unembed output.
                    self.model(
                        inputs=input_ids, 
                        cu_seqlens=cu_seqlens,
                        return_loss=True,
                        target=labels,
                        metrics=metrics,
                        timing=timing,
                        losses=losses,
                        cache=dict(),
                    )
                    
                    # Get combined loss
                    loss = losses.get_final_loss()
                    losses.log(metrics, "Losses")
                    
                    # Rarely this loss will become nan. It's really annoying to debug and seems to be due
                    # to some random state of the GPUs. For now, I am just gonna multiply the loss on this GPU by
                    # zero. This means all gradients will be zero and the model will not be changed. I think this
                    # will technically cause gradients across ranks to be aff by a factor of 1/# gpus. However, I
                    # am not gonna worry about it.
                    if loss.isnan().item():
                        loss = loss * 0
                        num_nan += 1
                        print(f"loss was nan! Total nan losses: {num_nan}")
                
                timing.end_span("fwd")
                    

                # Backpropagate loss
                timing.start_span("bwd")
                if self.use_amp:
                    self.grad_scaler.scale(loss).backward()
                else:
                    loss.backward()
                    
                # Add to number of trainable sequences this rank
                # has gone through
                self.num_tokens += torch.tensor(
                    (labels != -100).sum().item(),
                    dtype=torch.long,
                    device=f"cuda:{self.local_rank}"
                )
                self.num_sequences += torch.tensor(
                    batch.n_seqs, 
                    dtype=torch.long,
                    device=f"cuda:{self.local_rank}"
                )
                    
                if True in [i.grad.isnan().any().item() for i in self.model.parameters() if i.requires_grad == True]:
                    num_nan_grad_steps += 1

                # Update accumulated grad acc loss
                grad_acc_loss += loss.item()/self.grad_acc_steps
                grad_acc_ntp += losses.get("ntp")/self.grad_acc_steps

                timing.end_span("grad acc step")
                timing.end_span("bwd")

                # Next step of grad acc

            timing.start_span("fwd_bwd_post")

            # Unscale grads
            if self.use_amp:
                self.grad_scaler.unscale_(self.optimizer)
                
            # Calculate grad norm
            with torch.no_grad():
                if is_main_process():
                    for w_n, w in self.model.named_parameters():
                        g = w.grad
                        if g is not None:
                            metrics.add_metric(
                                "Grad_Norm",
                                w_n,
                                w.grad.norm(p="fro")
                            )

                
            # Clip gradients
            if self.optim_config.grad_clipping_value is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.optim_config.grad_clipping_value
                )
            
            # Take optimizer step
            if self.use_amp:
                self.grad_scaler.step(self.optimizer)
            else:
                self.optimizer.step()

            # Update scheduler
            self.scheduler.step()
            
            # Step the gradient scaler
            if self.use_amp:
                self.grad_scaler.update()
            
            
            # Update batch losses
            batch_loss += grad_acc_loss/self.log_steps
            batch_ntp += grad_acc_ntp/self.log_steps

            timing.end_span("fwd_bwd_post")
            
            
            # Update emas
            timing.end_span("step")
            avg_time_per_fwd = update_ema(
                avg_time_per_fwd, 
                timing.get_span_total("fwd")
            )
            avg_time_per_bwd = update_ema(
                avg_time_per_bwd, 
                timing.get_span_total("bwd")
            )
            avg_time_per_iter = update_ema(
                avg_time_per_iter, 
                timing.get_span_total("fwd_bwd_post")
            )
            
            
            # Add wandb metrics
            if step % self.log_steps == 0:
                metrics.add_metrics(
                    "General",
                    loss=Metric(batch_loss, ReduceOp.AVG),
                    perplexity=Metric(torch.exp(torch.tensor(batch_loss)), ReduceOp.AVG),
                    loss_ntp=Metric(batch_ntp, ReduceOp.AVG),
                    lr=Metric(self.optimizer.param_groups[0]['lr'], ReduceOp.AVG),
                    num_sequences=Metric(self.num_sequences, ReduceOp.SUM),
                    num_tokens=Metric(self.num_tokens, ReduceOp.SUM),
                )
                metrics.add_metrics(
                    "Timing",
                    avg_time_per_fwd=Metric(avg_time_per_fwd, ReduceOp.AVG),
                    avg_time_per_bwd=Metric(avg_time_per_bwd, ReduceOp.AVG),
                    avg_time_per_iter=Metric(avg_time_per_iter, ReduceOp.AVG),
                )
                
                batch_loss = 0
                batch_ntp = 0
            
            
            # Save model
            if step % self.num_save_steps == 0:
                self.save_model(step)
            
            # Save timing metrics
            if step > 0:
                metrics.add_metrics(
                    "Timing",
                    **timing.get_all_span_totals(
                        return_as_metrics=True
                    )
                )
                
            # Log metrics to wandb
            metrics.log_dist(step=step)

            # Testing the model
            if step % self.num_steps_test == 0 and step > 1 and self.test_loss:
                with torch.no_grad():
                    # Put model in eval mode
                    self.model.eval()
                    
                    # Get test dataloader
                    test_dataset = dataset.get_test_dataloader()

                    # Average loss and ppl
                    avg_test_loss = 0
                    avg_test_ppl = 0
                    total_batches = 0
                    
                    # Test metrics
                    test_metrics = Metrics(
                        "Test"
                    )
                    
                    # Test timing
                    test_timing = Timing(enabled=False)

                    # Testing loop
                    batch = test_dataset.sample_batch()
                    pbar = tqdm(total=len(test_dataset.test_data_loader))
                    while batch is not None:
                        pbar.update(batch.n_seqs)
                        # Get input and labels
                        input_ids = batch.input_ids.to(self.model.device)
                        labels = batch.labels.to(self.model.device)
                        cu_seqlens = batch.cu_seqlens.to(self.model.device)
                    
                        # Losses for the fwd pass
                        test_losses = Losses()

                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16) if self.use_amp else nullcontext():
                            # Forward pass and get loss           
                            self.model(
                                inputs=input_ids, 
                                cu_seqlens=cu_seqlens,
                                return_loss=True,
                                target=labels,
                                metrics=test_metrics,
                                timing=test_timing,
                                losses=test_losses,
                            )
                            
                            # Loss is just the main loss
                            loss = test_losses.get("ntp")
                            # test_losses.log(test_metrics, "Test")

                        # Perplexity
                        ppl = loss.exp().item()
                        loss = loss.item()

                        # Accumulate loss and ppl
                        avg_test_loss += loss
                        avg_test_ppl += ppl
                        total_batches += 1
                        
                        # Sample next batch
                        batch = test_dataset.sample_batch()
                    pbar.close()

                    # Get the averages
                    avg_test_loss /= total_batches
                    avg_test_ppl /= total_batches

                    test_metrics.add_metrics(
                        "Test",
                        test_loss=Metric(avg_test_loss, ReduceOp.AVG),
                        test_perplexity=Metric(avg_test_ppl, ReduceOp.AVG),
                    )
                        
                    # Log metrics
                    test_metrics.log_dist(step=step)

                    del batch, input_ids, labels, cu_seqlens, loss, test_metrics

                    # Put model in train mode
                    self.model.train()
                    
                    
                    
            # Early stop
            if step == self.optim_config.num_steps_early_stop:
                break

           
            
            
    def update_save_paths(self):
        step_str = f"{self.optim_config.checkpoint_step}" if self.trainer_config.include_step_in_checkpoint else ""
        self.base_pth = os.path.join(self.model_save_path, step_str)
        suf = ""
        self.model_save_path_model = os.path.join(self.base_pth, f"model.safetensors")
        self.model_save_path_optim = os.path.join(self.base_pth, f"optimizer{suf}.pt")
        self.model_save_path_scheduler = os.path.join(self.base_pth, f"scheduler{suf}.pt")
        self.model_save_path_grad_scaler = os.path.join(self.base_pth, f"grad_scaler{suf}.pt")
        self.model_save_path_model_cfg = os.path.join(self.base_pth, f"model_cfg.json")
        self.model_save_path_optim_cfg = os.path.join(self.base_pth, f"optimizer_cfg.json")
        self.model_save_path_trn_cfg = os.path.join(self.base_pth, f"trainer_cfg.json")
                
                
    def save_model(
            self, 
            step
        ):
        # Update the starting step in the model and optim config
        self.model_config.checkpoint_step = step
        self.optim_config.checkpoint_step = step
        self.trainer_config.starting_step = step

        # Update the number of tokens/sequences this
        # model checkpoint has been trained for
        num_toks = self.num_tokens.clone()
        dist.all_reduce(
            num_toks,
            op=ReduceOp.SUM
        )
        num_seqs = self.num_sequences.clone()
        dist.all_reduce(
            num_seqs,
            op=ReduceOp.SUM
        )
        self.model_config.num_tokens_trained = num_toks.item()
        self.model_config.num_sequences_trained = num_seqs.item()
        
        # Update the wandb id
        self.trainer_config.wandb_id = self.wandb_id
        
        # Update the save paths
        self.update_save_paths()
            
        # Just save on the first rank
        if is_main_process():
            if not os.path.exists(self.base_pth) and is_main_process():
                os.makedirs(self.base_pth)
            
            # Save the model. Model is DDP replicated
            if is_main_process():
                save_file(self.model_ref.state_dict(), self.model_save_path_model)
            # Save optimizer
            torch.save(self.optimizer.state_dict(), self.model_save_path_optim)
            # Save scheduler
            torch.save(self.scheduler.state_dict(), self.model_save_path_scheduler)
            # Save grad scaler if using one
            if self.use_amp:
                torch.save(self.grad_scaler.state_dict(), self.model_save_path_grad_scaler)
            # Save configs. Configs only need to be saved on one rank
            if is_main_process():
                with open(self.model_save_path_model_cfg, "w") as f:
                    json.dump(self.model_config.to_json(), f)
                with open(self.model_save_path_optim_cfg, "w") as f:
                    json.dump(self.optim_config.to_json(), f)
                with open(self.model_save_path_trn_cfg, "w") as f:
                    json.dump(self.trainer_config.to_json(), f)
        
        dist.barrier()

            
    @staticmethod
    def load_checkpoint(
            model_save_path: str,
            step: Optional[int] = None,
            reset_wandb: bool = False,
            new_wandb_name: Optional[str] = None,
            num_devices: Optional[int] = None,
        ) -> Self:
        # Get step as a string if not supplied
        step_str = find_highest_step(
            step,
            model_save_path,
        )
        
        # Load in the configs
        base_pth = os.path.join(model_save_path, step_str)
        model_config_name = os.path.join(base_pth, "model_cfg.json")
        with open(model_config_name, "r") as f:
            model_config = ModelConfig.from_json(json.load(f))
        optim_config_name = os.path.join(base_pth, "optimizer_cfg.json")
        with open(optim_config_name, "r") as f:
            optim_config = OptimConfig.from_json(json.load(f))
        trainer_config_name = os.path.join(base_pth, "trainer_cfg.json")
        with open(trainer_config_name, "r") as f:
            trainer_config = TrainerConfig.from_json(json.load(f))
        assert model_config.checkpoint_step == optim_config.checkpoint_step
        
        # Get number of devices to use if not supplied. If
        # supplied, change the trainer config
        if num_devices is None:
            num_devices = trainer_config.num_devices
        else:
            trainer_config.num_devices = num_devices
            
        # Reset wandb in trainer config
        if reset_wandb:
            trainer_config.wandb_id = None
            if new_wandb_name is not None:
                trainer_config.run_name = new_wandb_name
        wandb_name = trainer_config.run_name
        
        if is_main_process():
            print(f"Loading in checkpoint at `{base_pth}`")
            print(
        f"""
-------------------------------------
Running with the following setup:
Model config: {model_config_name}:
{model_config}

Optim config: {optim_config_name}
{optim_config}

Trainer config: {trainer_config_name}
{trainer_config}

Starting from step {step}
Reset wandb: {reset_wandb}
Wandb name: {wandb_name}
Num devices: {num_devices}
-------------------------------------
""".strip())
        
        # Load in the model and set up the trainer
        return Trainer(
            model_config=model_config,
            optim_config=optim_config,
            trainer_config=trainer_config,
            load_checkpoint_on_init=True
        )






# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⡿⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⡟⠀⣠⣀⠙⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⣄⠈⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⡟⠀⣼⣿⣿⣿⣦⣄⠙⠻⣿⣿⣿⣿⣿⣿⣿⠀⢻⣷⣦⣈⠙⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠛⠛⠛⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⠃⢰⣿⣿⣿⣿⣿⣿⣿⣦⡍⠙⠉⣁⣠⣤⣤⣄⡀⢻⣿⣿⣿⣦⣄⣈⠙⠿⢿⣿⣿⣿⣿⣿⣿⣿⡿⠟⠋⣀⣠⣴⣶⣿⣷⡄⠘⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⡏⠀⣼⣿⣿⣿⣿⣿⣿⣿⣿⣄⣀⠛⢿⣿⣿⣿⣿⣷⣾⣿⣿⣿⣿⣿⣿⣷⣶⣄⠛⣿⣿⣿⡿⠟⠋⣠⣴⣾⣿⣿⣿⣿⣿⣿⡇⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⡇⢰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⣌⠻⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣄⠘⣿⠋⠀⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⡏⢠⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⠁⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣦⡄⠉⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣤⣦⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡏⠀⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠁⣸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⠀⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠁⢠⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⡇⠘⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢻⣿⡀⢻⣿⣿⣿⠏⢠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣷⡀⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡘⣿⠃⣸⣿⣿⠏⢀⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⡿⠿⠿⠛⠃⣠⣿⣿⡿⠟⠁⢀⣀⣀⡀⠉⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣶⣶⣿⡿⠋⢀⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣷⡈⢶⣶⣿⣿⣿⣿⣦⣤⣾⣿⣿⣿⣿⣷⣀⢘⣿⣿⣿⣿⣿⣿⣿⣿⡿⠛⠉⠀⣀⣀⣀⠀⠉⠻⣿⣿⣿⣿⣿⣿⠟⠀⠀⠛⠛⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣷⣄⡛⠟⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣆⣠⣶⣿⣿⣿⣿⣿⣷⣄⠈⣿⣿⣿⣿⣿⣶⣾⣿⡟⠁⣸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⡟⢁⣾⡟⠿⠛⠉⢻⣿⣿⣿⣿⣧⣀⡀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠟⠁⣠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⡿⠀⣿⣿⣿⣿⣿⡁⣉⣁⣤⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠛⠛⠛⢿⡿⠿⢿⣿⣿⡀⠠⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⡟⠀⠚⠛⣉⣉⣉⡉⠛⢿⣿⣿⣿⣿⣿⣿⡿⢿⣿⠿⢿⣿⣿⡏⣿⣿⣿⣿⣿⣧⣴⣶⣧⡀⢉⣠⣶⣿⣿⣿⣷⡀⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣷⣶⣿⣿⣿⣿⣿⣷⣦⡀⠙⠻⢿⣿⣿⣿⣧⣌⠉⣠⣬⣍⠋⢁⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣾⣿⠿⢿⣿⣿⣿⡇⠘⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣤⣀⣉⠙⠛⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠿⠟⠛⠋⠉⠠⠤⣤⣴⣶⣦⣤⣤⣄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⡀⠠⣤⣤⣤⣤⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠿⣿⣿⣿⠃⢀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⡀⠉⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠋⣉⣠⣤⣶⣶⣤⣤⣄⠀⠸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠀⣸⣿⣿⣿⣿⣿⣿⣿⣿⠛⢁⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣆⡈⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠁⣴⣿⣿⣿⣿⣿⣿⣿⡟⢁⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⡀⢹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⣀⣠⣤⡄⠸⣿⣿⣿⣿⣇⠸⣿⡏⢹⣿⣿⡿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣧⡄⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠈⢿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⠀⣿⣿⣿⣿⣿⣦⣈⠁⠘⠿⣿⡇⢸⠿⠟⢉⣠⣿⣿⣿⣿⣿⣷⡀⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⣧⡄⠹⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⣿⣿⣿⣿⣿⣿⣿⣿⣶⣦⣤⣤⣤⡆⣿⣿⣿⣿⣿⣿⣿⣿⣿⣇⠈⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠁⠹⣿⠃⢰⣿⣷⣄⠘⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠀⣰⡀⠈⠀⣿⣿⣿⣿⣄⠈⢻⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⣿⣿⣿⣿⣿⣿⣿⡏⢻⣿⣿⣿⣿⣇⠹⣿⣿⣿⣿⣿⣿⣿⡿⠁⣸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠁⣰⣿⣇⠀⢰⣿⣿⣿⣿⣿⣇⠈⢿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⣿⣿⣿⣿⣿⣿⣿⣧⠘⣿⣿⣿⣿⣿⣄⠻⣿⣿⣿⣿⡿⠟⢀⠰⠻⠿⠿⣿⣿⣿⣿⣿⣿⣿⡟⢀⣼⣿⣿⣿⢠⣿⣿⣿⣿⣿⣿⣿⣇⠈⢻⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠛⠀⣿⣿⣿⣿⣿⣿⣿⣿⡄⢹⣿⣿⣿⣿⣿⣶⣤⣤⣤⣤⣴⣾⣿⣶⡶⠂⣴⣿⣿⣿⣿⡿⠟⠉⣠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡆⠈⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠋⢀⣰⣶⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣀⠘⠿⢿⠿⠛⠁⣀⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣇⠀⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠁⣠⣿⣿⣿⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠛⢁⣠⣤⣤⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⢸⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⢸⣿⣿⣿⣿⡀⢿⣿⣿⣿⣿⣿⣿⣿⣿⡀⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⠘⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠀⣼⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⢼⣿⣿⣿⣿⡇⠘⣿⣿⣿⣿⣿⣿⣿⣿⡇⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣆⠙⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠁⣴⣿⣿
