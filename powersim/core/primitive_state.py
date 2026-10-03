"""Map MPM particle state onto PowerFoam primitives: positions from x, radii from det(F)^(1/3),
dipole quaternions and appearance axes from the rotation of F = R S."""

import torch

_SOFTPLUS_BETA = 100.0


def compute_grid_occupancy_volume(
    points: torch.Tensor, n_grid: int, grid_dx: float, uniform: bool = False
) -> torch.Tensor:
    """Per-particle volume via the same grid-occupancy heuristic as PhysGaussian's
    particle_filling.filling.get_particle_volume, reimplemented in pure torch instead of taichi."""
    idx = (points / grid_dx).floor().long().clamp(0, n_grid - 1)
    cell_id = idx[:, 0] * n_grid * n_grid + idx[:, 1] * n_grid + idx[:, 2]
    _, inverse, counts = torch.unique(cell_id, return_inverse=True, return_counts=True)
    vol = (grid_dx**3) / counts[inverse].to(points.dtype)
    if uniform:
        vol = vol.mean().expand(points.shape[0]).contiguous()
    return vol


def softplus_beta100(x: torch.Tensor) -> torch.Tensor:
    """Matches PowerfoamScene.get_radii/get_density: F.softplus(x, beta=100)"""
    return torch.nn.functional.softplus(x, beta=_SOFTPLUS_BETA)


def inverse_softplus_beta100(y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Inverse of softplus_beta100, in the form y + log(1 - exp(-beta*y)) / beta, which does not
    overflow for large beta*y (log(expm1(beta*y)) does, from y of about 0.89 in float32)."""
    y = y.clamp_min(eps)
    return y + torch.log(-torch.expm1(-_SOFTPLUS_BETA * y)) / _SOFTPLUS_BETA


def rotmat_to_quaternion_wxyz(R: torch.Tensor) -> torch.Tensor:
    """Batched rotation matrix (..., 3, 3) -> unit quaternion (..., 4) in (w, x, y, z) order."""
    m = R
    batch_shape = m.shape[:-2]
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]

    trace = m00 + m11 + m22

    def _branch_trace(eps=1e-12):
        s = torch.sqrt((trace + 1.0).clamp_min(0.0) + eps) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
        return torch.stack([w, x, y, z], dim=-1)

    def _branch_x(eps=1e-12):
        s = torch.sqrt((1.0 + m00 - m11 - m22).clamp_min(0.0) + eps) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
        return torch.stack([w, x, y, z], dim=-1)

    def _branch_y(eps=1e-12):
        s = torch.sqrt((1.0 + m11 - m00 - m22).clamp_min(0.0) + eps) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
        return torch.stack([w, x, y, z], dim=-1)

    def _branch_z(eps=1e-12):
        s = torch.sqrt((1.0 + m22 - m00 - m11).clamp_min(0.0) + eps) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
        return torch.stack([w, x, y, z], dim=-1)

    use_trace = trace > 0
    use_x = (~use_trace) & (m00 >= m11) & (m00 >= m22)
    use_y = (~use_trace) & (~use_x) & (m11 >= m22)
    use_z = (~use_trace) & (~use_x) & (~use_y)

    q = torch.zeros(*batch_shape, 4, dtype=m.dtype, device=m.device)
    if use_trace.any():
        q = torch.where(use_trace[..., None], _branch_trace(), q)
    if use_x.any():
        q = torch.where(use_x[..., None], _branch_x(), q)
    if use_y.any():
        q = torch.where(use_y[..., None], _branch_y(), q)
    if use_z.any():
        q = torch.where(use_z[..., None], _branch_z(), q)

    return q / q.norm(dim=-1, keepdim=True)


def quaternion_wxyz_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternion (..., 4) in (w, x, y, z) order -> rotation matrix (..., 3, 3)"""
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    row0 = torch.stack([1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)], dim=-1)
    row1 = torch.stack([2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)], dim=-1)
    row2 = torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def quaternion_multiply_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product q1 * q2, both (..., 4) in (w, x, y, z) order."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def polar_decompose(F: torch.Tensor):
    """F (..., 3, 3) -> (R, volume_scale, max_stretch_scale, mean_stretch_scale)"""
    U, S, Vh = torch.linalg.svd(F)
    R = U @ Vh
    det_R = torch.linalg.det(R)
    # Flip the smallest singular vector's sign to keep R a proper rotation (det=+1) whenever SVD
    # handed us a reflection
    flip = det_R < 0
    if flip.any():
        U_fixed = U.clone()
        U_fixed[flip, :, -1] *= -1
        R = torch.where(flip[..., None, None], U_fixed @ Vh, R)

    det_F = torch.linalg.det(F)
    volume_scale = det_F.abs().clamp_min(1e-12).pow(1.0 / 3.0)
    # torch.linalg.svd returns S sorted descending, so S[..., 0] is already the max
    max_stretch_scale = S.max(dim=-1).values
    mean_stretch_scale = S.mean(dim=-1)
    return R, volume_scale, max_stretch_scale, mean_stretch_scale


def rotate_texel_sv_axis(base_texel_sv_axis: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """axis_new = R @ axis_old per (texel_site, sv_dof) direction."""
    num_texel_sites = base_texel_sv_axis.shape[1]
    sv_dof = base_texel_sv_axis.shape[-1] // 3
    axis = base_texel_sv_axis.to(dtype=torch.float32).view(-1, num_texel_sites, sv_dof, 3)
    axis_new = torch.einsum("kij,ktdj->ktdi", R, axis)
    return axis_new.reshape(base_texel_sv_axis.shape)


def apply_deformation_to_primitives(
    scene,
    indices: torch.Tensor,
    F: torch.Tensor,
    base_quaternions: torch.Tensor,
    base_radii: torch.Tensor,
    centroid: torch.Tensor = None,
    base_texel_sv_axis: torch.Tensor = None,
    radius_scale_mode: str = "volume",
    update_quaternions: bool = True,
    update_radii: bool = True,
) -> None:
    """Write MPM deformation gradient(s) F into scene.points/quaternions/radii (and, if given,
    texel_sv_axis) in place."""
    device = scene.points.device
    F = F.to(device=device, dtype=torch.float32)
    if F.dim() == 2:
        F = F.unsqueeze(0).expand(indices.shape[0], -1, -1)

    R, volume_scale, max_stretch_scale, mean_stretch_scale = polar_decompose(F)
    if radius_scale_mode == "volume":
        radius_scale = volume_scale
    elif radius_scale_mode == "max_stretch":
        radius_scale = max_stretch_scale
    elif radius_scale_mode == "mean_stretch":
        radius_scale = mean_stretch_scale
    else:
        raise ValueError(f"Unknown radius_scale_mode: {radius_scale_mode!r}")

    # get_normals()/get_tangents() read quaternions[:, 0] as w and return ROW 0 of the corresponding
    # rotation matrix (not "R @ local_x_axis", the more common convention)
    if update_quaternions:
        q_delta = rotmat_to_quaternion_wxyz(R.transpose(-1, -2))
        q_new = quaternion_multiply_wxyz(base_quaternions.to(device), q_delta)
        scene.quaternions.data[indices] = q_new.to(scene.quaternions.dtype)

    if update_radii:
        new_radii = base_radii.to(device) * radius_scale
        scene.radii.data[indices] = inverse_softplus_beta100(new_radii).to(scene.radii.dtype)

    if centroid is not None:
        centroid = centroid.to(device=device, dtype=torch.float32)
        points_old = scene.points.data[indices].to(torch.float32)
        points_new = centroid + torch.einsum("kij,kj->ki", F, points_old - centroid)
        scene.points.data[indices] = points_new.to(scene.points.dtype)

    if base_texel_sv_axis is not None:
        axis_new = rotate_texel_sv_axis(base_texel_sv_axis.to(device), R)
        scene.texel_sv_axis.data[indices] = axis_new.to(scene.texel_sv_axis.dtype)
