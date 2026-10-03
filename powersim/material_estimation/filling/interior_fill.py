"""Fill an object's interior with particles (PhysGaussian's particle_filling on PowerFoam primitives)."""

import torch
import taichi as ti

from powersim.core.thirdparty_paths import ensure_physgaussian_on_path

ensure_physgaussian_on_path()

from particle_filling.filling import fill_particles  # noqa: E402

_TAICHI_INITIALIZED = False


def ensure_taichi_initialized(arch=None) -> None:
    """particle_filling/filling.py defines its taichi kernels but never calls ti.init() itself."""
    global _TAICHI_INITIALIZED
    if _TAICHI_INITIALIZED:
        return
    ti.init(arch=arch if arch is not None else ti.cuda)
    _TAICHI_INITIALIZED = True


def radii_to_cov_upper(radii: torch.Tensor) -> torch.Tensor:
    """Isotropic sphere of radius r -> upper-triangular covariance diag(r^2,r^2,r^2), in PhysGaussian's
    own (xx,xy,xz,yy,yz,zz) ordering."""
    r2 = (radii.reshape(-1) ** 2).to(torch.float32)
    zeros = torch.zeros_like(r2)
    return torch.stack([r2, zeros, zeros, r2, zeros, r2], dim=-1)


def nearest_neighbor_copy(new_pos: torch.Tensor, orig_pos: torch.Tensor, orig_attrs: dict, chunk: int = 4096) -> dict:
    """For each row of new_pos, find its closest row in orig_pos (chunked cdist to bound peak
    memory for large point counts) and copy the corresponding orig_attrs entries."""
    idx = torch.empty(new_pos.shape[0], dtype=torch.long, device=new_pos.device)
    for start in range(0, new_pos.shape[0], chunk):
        end = min(start + chunk, new_pos.shape[0])
        d = torch.cdist(new_pos[start:end], orig_pos)
        idx[start:end] = d.argmin(dim=-1)
    return {name: tensor[idx] for name, tensor in orig_attrs.items()}


def fill_interior(
    points: torch.Tensor,
    radii: torch.Tensor,
    density: torch.Tensor,
    quaternions: torch.Tensor,
    grid_n: int = 64,
    max_samples: int = 200_000,
    density_thres: float = 2.0,
    search_thres: float = 1.0,
    max_particles_per_cell: int = 1,
    search_exclude_dir: int = 5,
    ray_cast_dir: int = 4,
    padding: float = 0.1,
    smooth: bool = False,
):
    """Returns (points_filled, radii_filled, density_filled, quaternions_filled, is_interior)"""
    ensure_taichi_initialized()

    device = points.device
    n_orig = points.shape[0]

    pos_min = points.min(dim=0).values
    pos_max = points.max(dim=0).values
    extent = (pos_max - pos_min).max().item()
    pad = padding * extent
    origin = pos_min - pad
    grid_dx = (extent + 2 * pad) / grid_n

    points_shifted = points - origin
    cov_upper = radii_to_cov_upper(radii)

    filled_points_shifted = fill_particles(
        pos=points_shifted,
        opacity=density.reshape(-1),
        cov=cov_upper,
        grid_n=grid_n,
        max_samples=max_samples,
        grid_dx=grid_dx,
        density_thres=density_thres,
        search_thres=search_thres,
        max_particles_per_cell=max_particles_per_cell,
        search_exclude_dir=search_exclude_dir,
        ray_cast_dir=ray_cast_dir,
        boundary=None,
        smooth=smooth,
    )
    filled_points = filled_points_shifted + origin

    n_new = filled_points.shape[0] - n_orig
    new_points = filled_points[n_orig:]

    if n_new > 0:
        orig_attrs = {"radii": radii, "density": density, "quaternions": quaternions}
        new_attrs = nearest_neighbor_copy(new_points, points, orig_attrs)
        radii_filled = torch.cat([radii, new_attrs["radii"]], dim=0)
        density_filled = torch.cat([density, new_attrs["density"]], dim=0)
        quaternions_filled = torch.cat([quaternions, new_attrs["quaternions"]], dim=0)
    else:
        radii_filled, density_filled, quaternions_filled = radii, density, quaternions

    is_interior = torch.zeros(filled_points.shape[0], dtype=torch.bool, device=device)
    is_interior[n_orig:] = True

    return filled_points, radii_filled, density_filled, quaternions_filled, is_interior
