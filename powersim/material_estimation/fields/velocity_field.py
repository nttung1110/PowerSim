"""Triplane field for the initial velocity."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sample_plane(plane: torch.Tensor, coords_2d: torch.Tensor) -> torch.Tensor:
    """plane: (1, C, res, res)"""
    grid = coords_2d.view(1, 1, -1, 2)
    sampled = F.grid_sample(plane, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled[0, :, 0, :].transpose(0, 1)


def _plane_tv_loss(plane: torch.Tensor) -> torch.Tensor:
    """Total-variation regularizer: mean squared difference between adjacent grid cells, horizontally
    and vertically."""
    diff_h = plane[:, :, 1:, :] - plane[:, :, :-1, :]
    diff_w = plane[:, :, :, 1:] - plane[:, :, :, :-1]
    return (diff_h**2).mean() + (diff_w**2).mean()


class TriplaneVelocityField(nn.Module):
    def __init__(
        self,
        aabb: torch.Tensor,
        resolution: int = 24,
        feat_dim: int = 32,
        hidden_dim: int = 64,
        num_hidden_layers: int = 2,
        output_scale: float = 0.1,
        device: str = "cuda:0",
    ):
        """Args: aabb: (2, 3) tensor, [min_xyz, max_xyz]."""
        super().__init__()
        self.register_buffer("aabb_min", aabb[0].clone().to(device))
        self.register_buffer("aabb_max", aabb[1].clone().to(device))
        self.output_scale = output_scale

        init_scale = 0.1
        self.plane_xy = nn.Parameter(torch.randn(1, feat_dim, resolution, resolution, device=device) * init_scale)
        self.plane_yz = nn.Parameter(torch.randn(1, feat_dim, resolution, resolution, device=device) * init_scale)
        self.plane_xz = nn.Parameter(torch.randn(1, feat_dim, resolution, resolution, device=device) * init_scale)

        layers = []
        in_dim = feat_dim
        for _ in range(num_hidden_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)]
            in_dim = hidden_dim
        final = nn.Linear(in_dim, 3)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.decoder = nn.Sequential(*layers).to(device)

    def _normalize(self, positions: torch.Tensor) -> torch.Tensor:
        extent = (self.aabb_max - self.aabb_min).clamp_min(1e-8)
        normed = 2.0 * (positions - self.aabb_min) / extent - 1.0
        return normed.clamp(-1.0, 1.0)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """positions: (n, 3) in the same space as `aabb`."""
        p = self._normalize(positions)
        feat_xy = _sample_plane(self.plane_xy, p[:, [0, 1]])
        feat_yz = _sample_plane(self.plane_yz, p[:, [1, 2]])
        feat_xz = _sample_plane(self.plane_xz, p[:, [0, 2]])
        feat = feat_xy + feat_yz + feat_xz
        return self.decoder(feat) * self.output_scale

    def compute_smoothness_loss(self) -> torch.Tensor:
        return _plane_tv_loss(self.plane_xy) + _plane_tv_loss(self.plane_yz) + _plane_tv_loss(self.plane_xz)
