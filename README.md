# What is this repo?

Most of my past research projects require training models from scratch. I have always opted for making the entire training loop from scratch so I know exactly what is going on everywhere. I am not a fan of trainers such as [PyTorch lightning](https://lightning.ai/docs/pytorch/stable/core-api/trainer). I should not be able to call train() and have something work.

My first Trainer was absolute garbage but got the job done. However I pretty much kept piling onto the garbage code cause why fix what isn't broken? This led to the terrible [repo](https://github.com/gmongaras/2Mamba2Furious) found in my 2Mamba2Furious paper. The code is quite bad for a lot of reasons and really embodies research code.

I wanted to rewrite my Trainer for all future projects I work on. This repo is the complete rewrite I did. Is it good code? Probably not. But I know what is going on everywhere, the code is mostly readable, and it is easy to edit.

The code is about 95% mine and 5% slop. Probably around 90% of the slop is in kernels I really couldn't bother spending a while trying to optimize. For example, the fused_enemb.py code is 100% slop. I do not know how it works nor do I care. I just cared that it was much faster than the original implementation. For almost everything else, I did care about code quality and did not want slop. I could've slopped this out in a day with cursor but then had something worse than PyTorch lightning and way less readable/understandable.

I also wanted to get evals working as submitting to any reasonable conference requires evals yet evals are very annoying to get unless you utilize some training framework, which is exactly what I am trying to avoid.

Anyway. To answer the question, "What is this repo?", it is a Trainer I mostly built for all my future pretraining research projects that I wanted to share in case others find it useful for their work.

Note that while I have an LLM optimization loop, it is not difficult to refactor the codebase for a diffusion model. I tried to keep things somewhat generic s.t. I can pretrain any type of model.


# Setup

I am assuming you have a linux machine with some GPUs. Hopefully more than 1, though the scripts work with a single GPU.

First download [python](https://www.python.org/downloads/) and [torch](https://pytorch.org/). For my setup, I am using Python 3.10.20 and torch 2.11.0 with Cuda 12.8 with A100 GPUs. I do not see why this repo wouldn't work on other GPUs and other version but who knows.

Then, create a venv for this repo with `python3 -m venv TTVenv`. Activate the venv with `source TTVenv/bin/activate` and download the necessary packages with `pip install -r requirements.txt`.

I also use wandb. Download wandb with `pip install wandb` then do `wandb login` to login to wandb.

I have performed a training run using the script described in [Model Training](#model-training). The wandb for this run is located [here](https://forge.coreweave.com/wandb/gmongaras1/Torch_Trainer/workspace) and the checkpoint is uploaded to huggingface [here]()


# Features

The current setup I have is a text-based transformer in a training loop. The base model setup comes from the small test model in [DeepSeekMoE](https://arxiv.org/abs/2401.06066) and the dataset is [fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb). In particular, I wanted training to be efficient so I use a varlen/cu_seqlens setup.

Features listed out:
1. [Model Training](#model-training)
2. [Checkpoint Reloading](#checkpoint-reloading)
3. [Checkpoint Inference](#checkpoint-inference)
4. [Checkpoint Eval](#checkpoint-eval)
5. [Metrics and Timing](#metrics-and-timing)
6. [Layer Interface](#layer-interface)


## Model Training

Training a model on 2 GPUs can be done with the following command:
```
nnodes=1
nproc_per_node=2
model_config="Tfm"
optim_config="Tfm"
trainer_config="Tfm"
torchrun \
    --nnodes $nnodes \
    --nproc_per_node $nproc_per_node \
    --rdzv_backend c10d \
    src/train.py \
    --model_config_name $model_config \
    --optim_config_name $optim_config \
    --trainer_config_name $trainer_config
```

If all goes well, stuff will download to `cache/` and, once done, the model should begin training.

If you are on a SLURM system, you can also edit the `runjob.sh` file and launch the training script with `sbatch runjob.sh`. As training progresses, you will see a `runjob.out` file with the runs progress.

By default, the trainer runs on 2 GPUs, for 25K steps at batch size of 128*2048 and 2048 sequence length. The default model is quite small at only 200M params. It is 9 layers and has a model dim of 1280. The hyperparameters follow [DeepSeekMoE](https://arxiv.org/abs/2401.06066), except for the batch size which is smaller than the specified batch size in DeepSeekMoE. A batch size of 4 million takes 8-16 A100 80GB GPUs. All configurations can be found in `src/exp_configs.py`.

The dataloader works by greedily taking sequences and shoving them in the current batch. By default, the batch size is `128*2048` tokens with max seq len of `2048`. This means all sequences are cut to 2048. The trivial case packs 128 sequences of length 2048 into the batch, thus having a batch size of 128 sequences. However, if a sequence is smaller than the `128*2048` batch size, we can either not worry about it and have a smaller batch, keeping a batch size of 128 sequences, or train on as many sequences that we can s.t. each sequence is at most 2048 tokens and we have a batch size of `128*2048` tokens. The dataloader does the latter. As such the min number of sequences is 128 and the max number of sequence sis `128*2048` (if we somehow have `128*2048` sequences of length 1).


### Adding losses

The current script has a single loss, which can be found under the `Losses` panel in wandb. Obviously this loss is good ol' NTP. Losses are managed through an object passed through each layer, found in `src/model/Losses.py`. If you want to add another loss, doing so is quite easy and can be done so by adding the following line in the code:

```
losses.add_loss(
    loss_name="...",
    loss_value=...
    loss_weight=1.0
)
```

Note that losses can overwrite each other. If you add a loss to a layer, you likely want to differentiate said loss with a layer specific indicator such as a suffix to the name `_layer{self.layer_num}`

Let's say I wanted to add a loss to minimize the L2 norm of the attention output for whatever reason. I could do so by adding the following right before returning in `src/model/layers/Attention.py`:

```
losses.add_loss(
    loss_name=f"attn_out_norm_l{self.layer_num}",
    loss_value=torch.linalg.vector_norm(out, dim=-1).mean()
    loss_weight=0.001
)
```

Note that all losses are automatically added together with their corresponding weights and backpropped at the end of each forward pass.







## Checkpoint Reloading














## Checkpoint Inference












## Checkpoint Eval












## Metrics and Timing


### Adding metrics

Metrics are useful to



### Adding timing spans

Timing spans allow you to see how long a section of code is taking. The timing interface can be found in ``.










## Layer Interface

The layer interface can be found in `src/model/layers/Layer.py`. All layers inherit from this class, mainly so one is able to independently checkpoint each layer and so the layer has the same set of inputs:
- `X` - Sequence of shape (N, d) for said layer to process.
- `cu_seqlens` - Sequence lens of shape (num_sequences+1) corresponding to the sequence lengths in X.
- `metrics` - Optional metrics as described in [Metrics and Timing](#metrics-and-timing)
- `timing` - Optional timing as described in [Metrics and Timing](#metrics-and-timing)
- `losses` - Optional losses as described in [Adding Losses](#adding-losses)
- `cache` - A dictionary that allows you to add stuff from one layer and use it in a future layer. For example, let's say I need to calculate something once for all layers and said computation is somewhat expensive and doesn't need to be differentiable. I can have the first layer do the expensive operation and add it to the cache, then all future layers can pull the cached output from the cache. Not used in this repo but it works and I found it quite useful.
- inf_cfg: Optional[InfConfig] = None - Object used for inference as described in [Checkpoint Inference](#checkpoint-inference).
