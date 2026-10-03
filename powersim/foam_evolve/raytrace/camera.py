"""Look-at camera for the ray-traced demos: {center, radius, azimuth_deg, elevation_deg, fov_deg,
world_up[, width, height]} as in config/cameras/*.json (PowerFoam's render_orbit.py conventions)."""

import math

import torch


def build_equatorial_basis(world_up: torch.Tensor):
    """Same as PowerFoam's render_orbit.py."""
    world_up = world_up.to(torch.float32).cpu()
    world_up = world_up / torch.norm(world_up)
    seed = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    if torch.dot(world_up, seed).abs() > 0.9:
        seed = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    u1 = torch.cross(seed, world_up, dim=-1)
    u1 = u1 / torch.norm(u1)
    u2 = torch.cross(world_up, u1, dim=-1)
    return u1, u2


def make_orbit_camera(
    center: torch.Tensor,
    radius: float,
    azimuth_deg: float,
    elevation_deg: float,
    fov_deg: float,
    width: int,
    height: int,
    device,
    world_up: torch.Tensor = None,
):
    from powerfoam.camera import TorchCamera

    if world_up is None:
        world_up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    world_up = world_up.to(torch.float32).cpu()
    world_up = world_up / torch.norm(world_up)
    u1, u2 = build_equatorial_basis(world_up)

    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)

    equator_dir = math.cos(az) * u1 + math.sin(az) * u2
    eye = center + radius * (math.cos(el) * equator_dir + math.sin(el) * world_up)

    fwd = center - eye
    fwd = fwd / torch.norm(fwd)

    up_ref = world_up
    if torch.dot(fwd, up_ref).abs() > 0.99:
        up_ref = u1
    right_dir = torch.cross(fwd, up_ref, dim=-1)
    right_dir = right_dir / torch.norm(right_dir)

    up_dir = torch.cross(right_dir, fwd, dim=-1)
    up_dir = up_dir / torch.norm(up_dir)

    fov_rad = math.radians(fov_deg)
    focal = 0.5 * width / math.tan(0.5 * fov_rad)
    right_scale = (width / 2.0 - 0.5) / focal
    up_scale = (height / 2.0 - 0.5) / focal

    return TorchCamera(
        eye=eye.to(device),
        right=(right_dir * right_scale).to(device),
        up=(up_dir * up_scale).to(device),
        width=width,
        height=height,
    )
