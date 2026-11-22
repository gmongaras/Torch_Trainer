# Torch has a fused RMSNorm kernel that only works when the activation
# and scale has the same dtype. This is just a wrapper to make sure
# they have the same dtype
import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedRMSNorm(nn.RMSNorm):
    def forward(self, x):
        # Cast weight
        weight = (
            self.weight.to(x.dtype)
            if self.weight is not None
            else None
        )
        # Do norm
        return F.rms_norm(
            x,
            self.normalized_shape,
            weight,
            self.eps
        )
