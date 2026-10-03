"""Differentiable primitive update (F to quaternions and radii) and rasterizer call."""

import torch

from powersim.core.primitive_state import (
    quaternion_multiply_wxyz,
    quaternion_wxyz_to_rotmat,
    rotmat_to_quaternion_wxyz,
)


def robust_rotation_from_F(F: torch.Tensor, num_iters: int = 9) -> torch.Tensor:
    """Rotational factor of F's polar decomposition (F = R S, R orthogonal, S symmetric positive-
    semidefinite), via Newton's iteration (R_{k+1} = 0.5*(R_k + inv(R_k)^T))"""
    R = F
    for _ in range(num_iters):
        R = 0.5 * (R + torch.linalg.inv(R).transpose(-1, -2))
    return R


def compute_deformed_quaternions_radii(
    F: torch.Tensor,
    base_quaternions: torch.Tensor,
    base_radii: torch.Tensor,
):
    """Non-mutating equivalent of primitive_state.apply_deformation_to_primitives' quaternion/ radii
    math (same q_delta = rotmat_to_quaternion_wxyz(R.transpose(-1, -2)) inverse-rotation convention
    documented there)"""
    R = robust_rotation_from_F(F)
    volume_scale = torch.linalg.det(F).abs().clamp_min(1e-12).pow(1.0 / 3.0)

    q_delta = rotmat_to_quaternion_wxyz(R.transpose(-1, -2))
    new_quaternions = quaternion_multiply_wxyz(base_quaternions, q_delta)
    new_radii = base_radii * volume_scale
    return new_quaternions, new_radii


def normals_tangents_from_quaternions(quaternions: torch.Tensor):
    """Reimplementation of PowerfoamScene.get_normals()/get_tangents(), parameterized on an arbitrary
    quaternions tensor instead of self.quaternions."""
    R = quaternion_wxyz_to_rotmat(quaternions)
    return R[..., 0, :], R[..., 1, :], R[..., 2, :]


def compute_texel_rgb(scene, points: torch.Tensor, radii: torch.Tensor, tangents: torch.Tensor,
                       bitangent: torch.Tensor, camera) -> torch.Tensor:
    """Appearance lookup, mirroring PowerfoamScene.forward()'s own texel_rgb computation."""
    offsets = scene.texel_sites * radii[:, None, None]
    offsets = (
        offsets[..., 0:1] * tangents[:, None, :]
        + offsets[..., 1:2] * bitangent[:, None, :]
    )
    texel_sites = points[:, None, :] + offsets

    att_sites, att_values, att_temps = scene.get_att_sv()
    texel_rgb = scene.sv.forward(
        texel_sites.view(-1, 3).detach(), camera, att_sites, att_values, att_temps
    )
    return texel_rgb.view(points.shape[0], scene.args.num_texel_sites, 3), texel_sites


def render_rgb_differentiable(
    scene,
    camera,
    points: torch.Tensor,
    radii: torch.Tensor,
    quaternions: torch.Tensor,
    depth_quantiles=None,
    indices: torch.Tensor = None,
    return_opacity: bool = False,
):
    """Gradient-tracked render of a PowerfoamScene with overridden points/radii/quaternions, bypassing
    scene.forward()'s hardcoded self.points/self.get_radii()/self.quaternions reads."""
    if indices is not None:
        points = scene.points.detach().clone().index_copy(0, indices, points)
        radii = scene.get_radii().detach().clone().index_copy(0, indices, radii)
        quaternions = scene.quaternions.detach().clone().index_copy(0, indices, quaternions)

    normals, tangents, bitangent = normals_tangents_from_quaternions(quaternions)
    texel_rgb, texel_sites = compute_texel_rgb(scene, points, radii, tangents, bitangent, camera)
    texel_height = scene.texel_height * radii[:, None]
    density = scene.get_density()

    rgb, opacity, *_ = scene.rasterizer.forward(
        camera,
        depth_quantiles,
        points,
        radii,
        density,
        normals,
        texel_sites,
        texel_rgb,
        texel_height,
        scene.adjacency,
        scene.adjacency_offsets,
        None,
        False,
    )
    if return_opacity:
        return rgb, opacity
    return rgb
