"""Photometric loss: (1 - w) * MSE + w * (1 - SSIM)."""

import torch
import torch.nn.functional as F


def _gaussian_window(window_size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window_2d = g.unsqueeze(1) @ g.unsqueeze(0)
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """img1, img2: (H, W, C) float tensors in [0, 1]."""
    x1 = img1.permute(2, 0, 1).unsqueeze(0)
    x2 = img2.permute(2, 0, 1).unsqueeze(0)
    channels = x1.shape[1]
    window = _gaussian_window(window_size, 1.5, channels, x1.device, x1.dtype)

    mu1 = F.conv2d(x1, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(x2, window, padding=window_size // 2, groups=channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = F.conv2d(x1 * x1, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(x2 * x2, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(x1 * x2, window, padding=window_size // 2, groups=channels) - mu1_mu2

    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def photometric_loss(pred: torch.Tensor, gt: torch.Tensor, ssim_weight: float = 0.2) -> torch.Tensor:
    """L2 + SSIM combined loss, matching PhysDreamer's own combination (train_material.py's
    train_one_step): loss = l2 * (1 - ssim_weight) + (1 - ssim) * ssim_weight."""
    l2 = F.mse_loss(pred, gt, reduction="mean")
    s = ssim(pred, gt)
    return l2 * (1.0 - ssim_weight) + (1.0 - s) * ssim_weight
