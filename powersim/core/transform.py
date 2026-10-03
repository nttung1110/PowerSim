"""World <-> MPM simulation-space transforms (recentre, scale to the grid, shift to the grid centre),
after PhysGaussian's transformation_utils but device-agnostic."""

import math

import torch


def generate_rotation_matrix(degree: float, axis: int, device) -> torch.Tensor:
    """degree in degrees; axis in {0, 1, 2} for x/y/z."""
    theta = degree / 180.0 * math.pi
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    if axis == 0:
        matrix = [[1, 0, 0], [0, cos_theta, -sin_theta], [0, sin_theta, cos_theta]]
    elif axis == 1:
        matrix = [[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]]
    elif axis == 2:
        matrix = [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]]
    else:
        raise ValueError(f"Invalid axis selection: {axis!r}")
    return torch.tensor(matrix, dtype=torch.float32, device=device)


def generate_rotation_matrices(degrees, axes, device) -> list:
    assert len(degrees) == len(axes)
    return [generate_rotation_matrix(d, a, device) for d, a in zip(degrees, axes)]


def apply_rotation(points: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    return points @ R.T


def apply_rotations(points: torch.Tensor, rotation_matrices: list) -> torch.Tensor:
    for R in rotation_matrices:
        points = apply_rotation(points, R)
    return points


def apply_inverse_rotation(points: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    return points @ R


def apply_inverse_rotations(points: torch.Tensor, rotation_matrices: list) -> torch.Tensor:
    for R in reversed(rotation_matrices):
        points = apply_inverse_rotation(points, R)
    return points


def compose_rotation_matrices(rotation_matrices: list) -> torch.Tensor:
    """Net single rotation equivalent to calling apply_rotations with this list, in the convention
    powersim.core.primitive_state.apply_deformation_to_primitives expects as a single F."""
    if not rotation_matrices:
        raise ValueError("rotation_matrices must be non-empty")
    R_net = rotation_matrices[0]
    for R in rotation_matrices[1:]:
        R_net = R @ R_net
    return R_net


def transform2origin(points: torch.Tensor, scale: float = 1.0):
    """Recenter points to their bbox center and rescale so the max bbox side length becomes `scale`."""
    min_pos = torch.min(points, 0)[0]
    max_pos = torch.max(points, 0)[0]
    max_diff = torch.max(max_pos - min_pos)
    original_mean_pos = (min_pos + max_pos) / 2.0
    scale_factor = scale / max_diff
    new_points = (points - original_mean_pos) * scale_factor
    return new_points, scale_factor, original_mean_pos


def undotransform2origin(points: torch.Tensor, scale_factor, original_mean_pos) -> torch.Tensor:
    return original_mean_pos + points / scale_factor


def shift2center111(points: torch.Tensor) -> torch.Tensor:
    offset = torch.ones(3, dtype=points.dtype, device=points.device)
    return points + offset


def undoshift2center111(points: torch.Tensor) -> torch.Tensor:
    offset = torch.ones(3, dtype=points.dtype, device=points.device)
    return points - offset


def undo_all_transforms(
    points: torch.Tensor,
    rotation_matrices: list,
    scale_factor,
    original_mean_pos,
) -> torch.Tensor:
    """Inverse of shift2center111 -> transform2origin -> apply_rotations, in that order."""
    unshifted = undoshift2center111(points)
    unscaled = undotransform2origin(unshifted, scale_factor, original_mean_pos)
    return apply_inverse_rotations(unscaled, rotation_matrices)
