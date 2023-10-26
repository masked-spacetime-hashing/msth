"""
codes for maximum likelihood estimation of the distributions in ray sampling.
"""
import torch
import torch.nn as nn


class MLELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ts, est_mean, est_std):
        # ts B x Ns
        mle_mean = ts.mean(dim=-1).unsqueeze(dim=-1)
        mle_val = (ts - mle_mean.unsqueeze(-1) ** 2).mean(dim=1)
        loss_mean = ((mle_mean - est_mean) ** 2).mean()
        loss_std = ((mle_val - est_std**2) ** 2).mean()
        return loss_mean + loss_std
