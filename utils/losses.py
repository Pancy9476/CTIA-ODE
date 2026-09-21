import torch
import torch.nn as nn


class ORHL(nn.Module):
    """
    Outlier-Robust Hybrid Loss (ORHL)
    A robust M-estimator combining Elastic-Net-style local convergence
    with bounded Lipschitz gradients for large anomalies.
    """

    def __init__(self, delta=1.0, lambda_=5.0):
        super().__init__()
        self.delta = delta
        self.lambda_ = lambda_

    def forward(self, preds, target):
        abs_e = torch.abs(preds - target)
        mask_small = abs_e <= self.delta
        loss_small = 0.5 * (abs_e ** 2) + self.lambda_ * abs_e
        loss_large = (self.delta + self.lambda_) * abs_e - 0.5 * (self.delta ** 2)
        loss = torch.where(mask_small, loss_small, loss_large)
        return loss.mean()