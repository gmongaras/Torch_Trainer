"""Mean linear cross-entropy for an unaligned vocabulary.

The training vocabulary, 50277, is not a multiple of 128. On this GPU cuBLAS
runs that shape at about 60 TFLOP/s instead of about 240. Liger also splits
the batch into chunks of 2048 and launches the logits, input-grad, and
weight-grad GEMMs once per chunk.

``weight`` is padded with zero rows up to the next multiple of 128, and the
batch is walked in tiles of ``_TOKEN_TILE`` tokens. Each tile runs the logits,
input-grad, and weight-grad GEMMs once. The Liger cross-entropy kernel writes
the logit gradient into the real vocabulary columns. The pad columns stay
zero, so they do not change the loss or the weight gradient. Backward only
scales the two gradients saved here.

The tile exists because one logits matrix for the whole batch is
``tokens x vocab`` in bf16. At 131,072 tokens that is about 13 GB, which the
8-GPU transformer step does not have. A larger tile is not faster.

This covers mean reduction with an ignore index. Bias, label smoothing,
softcap, z-loss, and token accuracy stay on Liger's fused linear
cross-entropy.
"""
import torch
import triton

from liger_kernel.ops.cross_entropy import liger_cross_entropy_kernel
from liger_kernel.ops.utils import amp_custom_bwd
from liger_kernel.ops.utils import amp_custom_fwd
from liger_kernel.ops.utils import is_hip


def _pad_amount(n, multiple=128):
    return (multiple - n % multiple) % multiple


# Logits for one tile are tile x vocab in bf16. 8192 tokens is about 0.8 GB
# and matches the speed of one GEMM over the whole batch.
_TOKEN_TILE = 8192


def _forward_tiles(hidden, weight, target, ignore_index, tokens, hidden_dim, vocab, pad):
    # The cast weight is bf16 under autocast. Pad rows are zero, so they
    # never win the softmax and never receive gradient.
    weight_pad = (
        torch.nn.functional.pad(weight, (0, 0, 0, pad)) if pad else weight
    )
    target = target.contiguous()
    n_non_ignore = int((target != ignore_index).sum().item())
    # Ignored labels return before the kernel stores a loss, so those
    # rows must already be zero. empty() leaves them as leftover memory.
    loss_1d = torch.zeros(tokens, dtype=torch.float32, device=hidden.device)
    grad_hidden = torch.empty_like(hidden)
    # fp32 accumulation across tiles. Summing the tiles in fp32 is closer
    # to a full-batch weight gradient than one bf16 GEMM.
    grad_weight = torch.zeros(
        vocab, hidden_dim, dtype=torch.float32, device=hidden.device,
    )
    block = min(32768, triton.next_power_of_2(vocab))
    for start in range(0, tokens, _TOKEN_TILE):
        end = min(start + _TOKEN_TILE, tokens)
        tile_hidden = hidden[start:end]
        logits = tile_hidden @ weight_pad.t()
        real = logits[:, :vocab]
        if real.stride(-1) != 1:
            real = real.contiguous()
        liger_cross_entropy_kernel[(end - start,)](
            X_ptr=real,
            X_stride=real.stride(-2),
            Y_ptr=target[start:end],
            Y_stride=target.stride(-1),
            weight_ptr=real,
            loss_ptr=loss_1d[start:end],
            z_loss_ptr=loss_1d[start:end],
            loss_stride=loss_1d.stride(-1),
            token_accuracy_ptr=loss_1d,
            token_accuracy_stride=0,
            predicted_tokens_ptr=loss_1d,
            predicted_tokens_stride=0,
            n_cols=vocab,
            n_non_ignore=n_non_ignore,
            sum_non_ignore_weight=n_non_ignore,
            weight_sum=0.0,
            ignore_index=ignore_index,
            lse_square_scale=0.0,
            label_smoothing=0.0,
            reduction="mean",
            softcap=0.0,
            RETURN_Z_LOSS=False,
            RETURN_TOKEN_ACCURACY=False,
            RETURN_PREDICTED_TOKENS=False,
            BLOCK_SIZE=block,
            HAS_WEIGHT=False,
            HAS_SOFTCAPPING=False,
            HAS_GRADIENTS=True,
            num_warps=32 if not is_hip() else 16,
        )
        if pad:
            logits[:, vocab:] = 0
        grad_hidden[start:end] = logits @ weight_pad
        grad_weight += torch.mm(logits.t(), tile_hidden)[:vocab].float()
    return grad_hidden, grad_weight, loss_1d.sum()


class _FastLinearCE(torch.autograd.Function):
    @staticmethod
    @amp_custom_fwd
    def forward(ctx, hidden, weight, target, ignore_index):
        tokens, hidden_dim = hidden.shape
        vocab = weight.shape[0]
        pad = _pad_amount(vocab)
        # no_grad: these GEMMs are the gradient. Recording them would keep
        # every tile's logits alive until the backward.
        with torch.no_grad():
            grad_hidden, grad_weight, loss = _forward_tiles(
                hidden, weight, target, ignore_index, tokens, hidden_dim, vocab, pad,
            )
        ctx.save_for_backward(grad_hidden, grad_weight)
        return loss

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output):
        grad_hidden, grad_weight = ctx.saved_tensors
        scale = grad_output.reshape(())
        return grad_hidden * scale, grad_weight * scale, None, None


def fast_linear_cross_entropy(hidden, weight, target, ignore_index=-100):
    """Mean cross-entropy of ``hidden @ weight.T`` against ``target``.

    ``weight`` is ``(vocab, hidden)``. Labels equal to ``ignore_index`` are
    dropped from the mean. Returns a scalar loss.
    """
    return _FastLinearCE.apply(hidden, weight, target, ignore_index)
