"""Unit tests for the deformation-gradient to primitive mapping."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from powersim.core.primitive_state import (
    apply_deformation_to_primitives,
    inverse_softplus_beta100,
    polar_decompose,
    quaternion_multiply_wxyz,
    quaternion_wxyz_to_rotmat,
    rotate_texel_sv_axis,
    rotmat_to_quaternion_wxyz,
    softplus_beta100,
)


def _random_rotations(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, 3, 3, generator=g)
    q, r = torch.linalg.qr(a)
    d = torch.diagonal(r, dim1=-2, dim2=-1).sign()
    q = q * d[:, None, :]
    det = torch.linalg.det(q)
    q[..., -1] *= det.sign().unsqueeze(-1)
    return q


def test_softplus_inverse_roundtrip():
    y = torch.rand(1000) * 5.0 + 1e-3
    x = inverse_softplus_beta100(y)
    y_hat = softplus_beta100(x)
    assert torch.allclose(y, y_hat, atol=1e-4)


def test_rotmat_quaternion_roundtrip():
    R = _random_rotations(64, seed=1)
    q = rotmat_to_quaternion_wxyz(R)
    assert torch.allclose(q.norm(dim=-1), torch.ones(64), atol=1e-4)
    R_hat = quaternion_wxyz_to_rotmat(q)
    assert torch.allclose(R, R_hat, atol=1e-4)


def test_quaternion_multiply_identity():
    R = _random_rotations(16, seed=2)
    q = rotmat_to_quaternion_wxyz(R)
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand_as(q)
    assert torch.allclose(quaternion_multiply_wxyz(identity, q), q, atol=1e-5)
    assert torch.allclose(quaternion_multiply_wxyz(q, identity), q, atol=1e-5)


def test_quaternion_multiply_composes_rotations():
    theta = torch.pi / 2
    Rz90 = torch.tensor(
        [
            [torch.cos(torch.tensor(theta)), -torch.sin(torch.tensor(theta)), 0.0],
            [torch.sin(torch.tensor(theta)), torch.cos(torch.tensor(theta)), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    q90 = rotmat_to_quaternion_wxyz(Rz90)
    q180 = quaternion_multiply_wxyz(q90, q90)
    R180_hat = quaternion_wxyz_to_rotmat(q180)
    R180 = Rz90 @ Rz90
    assert torch.allclose(R180_hat, R180, atol=1e-5)


def test_quaternion_multiply_matches_matrix_product():
    # quaternion_multiply_wxyz(q1, q2) -> R(q1) @ R(q2) (q1 as the LEFT/outer matrix)
    theta = torch.pi / 2
    c, s = torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))
    Rx90 = torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    Rz90 = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    qx = rotmat_to_quaternion_wxyz(Rx90)
    qz = rotmat_to_quaternion_wxyz(Rz90)

    q_new = quaternion_multiply_wxyz(qx, qz)
    R_new = quaternion_wxyz_to_rotmat(q_new)
    assert torch.allclose(R_new, Rx90 @ Rz90, atol=1e-4)

    # reversed order gives a genuinely different (non-commuting) result
    q_new_rev = quaternion_multiply_wxyz(qz, qx)
    R_new_rev = quaternion_wxyz_to_rotmat(q_new_rev)
    assert not torch.allclose(R_new_rev, Rx90 @ Rz90, atol=1e-2)
    assert torch.allclose(R_new_rev, Rz90 @ Rx90, atol=1e-4)


def test_polar_decompose_pure_rotation_preserves_volume():
    R = _random_rotations(8, seed=3)
    R_hat, volume_scale, max_stretch_scale, mean_stretch_scale = polar_decompose(R)
    assert torch.allclose(volume_scale, torch.ones(8), atol=1e-4)
    assert torch.allclose(max_stretch_scale, torch.ones(8), atol=1e-4)
    assert torch.allclose(mean_stretch_scale, torch.ones(8), atol=1e-4)
    assert torch.allclose(R_hat, R, atol=1e-4)


def test_polar_decompose_isotropic_stretch():
    scale = 1.7
    F = torch.eye(3).unsqueeze(0) * scale
    R_hat, volume_scale, max_stretch_scale, mean_stretch_scale = polar_decompose(F)
    assert torch.allclose(R_hat, torch.eye(3).unsqueeze(0), atol=1e-4)
    assert torch.allclose(volume_scale, torch.tensor([scale]), atol=1e-4)
    assert torch.allclose(max_stretch_scale, torch.tensor([scale]), atol=1e-4)
    assert torch.allclose(mean_stretch_scale, torch.tensor([scale]), atol=1e-4)


def test_polar_decompose_anisotropic_stretch_scalar_summaries():
    # Anisotropic, volume-preserving stretch: singular values (2.0, 1/sqrt(2), 1/sqrt(2))
    stretch = 2.0
    compress = 1.0 / (stretch**0.5)
    F = torch.diag(torch.tensor([stretch, compress, compress])).unsqueeze(0)
    _, volume_scale, max_stretch_scale, mean_stretch_scale = polar_decompose(F)
    expected_mean = (stretch + 2 * compress) / 3.0
    assert torch.allclose(volume_scale, torch.tensor([1.0]), atol=1e-4)
    assert torch.allclose(max_stretch_scale, torch.tensor([stretch]), atol=1e-4)
    assert torch.allclose(mean_stretch_scale, torch.tensor([expected_mean]), atol=1e-4)
    assert (volume_scale < mean_stretch_scale).all()
    assert (mean_stretch_scale < max_stretch_scale).all()


def test_rotate_texel_sv_axis_specific_case():
    # 90deg about z: +x should end up at +y.
    theta = torch.tensor(torch.pi / 2)
    Rz90 = torch.tensor(
        [
            [torch.cos(theta), -torch.sin(theta), 0.0],
            [torch.sin(theta), torch.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    ).unsqueeze(0)

    # K=1 primitive, num_texel_sites=1, sv_dof=1 -> raw shape (1, 1, 3)
    axis = torch.tensor([[[1.0, 0.0, 0.0]]])
    axis_new = rotate_texel_sv_axis(axis, Rz90)
    assert torch.allclose(axis_new, torch.tensor([[[0.0, 1.0, 0.0]]]), atol=1e-5)


def test_rotate_texel_sv_axis_preserves_norms():
    K, num_texel_sites, sv_dof = 4, 3, 2
    axis = torch.randn(K, num_texel_sites, sv_dof * 3)
    R = _random_rotations(K, seed=6)
    axis_new = rotate_texel_sv_axis(axis, R)

    norms_before = axis.view(K, num_texel_sites, sv_dof, 3).norm(dim=-1)
    norms_after = axis_new.view(K, num_texel_sites, sv_dof, 3).norm(dim=-1)
    assert torch.allclose(norms_before, norms_after, atol=1e-4)


class _FakeScene:
    """Minimal stand-in for PowerfoamScene, exposing only what
    apply_deformation_to_primitives touches."""

    def __init__(self, points, quaternions, radii, texel_sv_axis=None):
        self.points = _Param(points)
        self.quaternions = _Param(quaternions)
        self.radii = _Param(radii)
        if texel_sv_axis is not None:
            self.texel_sv_axis = _Param(texel_sv_axis)


class _Param:
    def __init__(self, data):
        self.data = data
        self.device = data.device
        self.dtype = data.dtype


def test_apply_deformation_updates_selected_indices_only():
    n = 5
    points = torch.zeros(n, 3)
    quaternions = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(n, 4).clone()
    radii = inverse_softplus_beta100(torch.ones(n) * 0.1)
    scene = _FakeScene(points, quaternions, radii)

    indices = torch.tensor([1, 3])
    base_q = quaternions[indices].clone()
    base_r = softplus_beta100(radii[indices]).clone()

    F = torch.eye(3) * 1.5  # isotropic growth, no rotation
    apply_deformation_to_primitives(scene, indices, F, base_q, base_r)

    untouched = torch.tensor([0, 2, 4])
    assert torch.allclose(scene.radii.data[untouched], radii[untouched])
    assert torch.allclose(scene.quaternions.data[untouched], quaternions[untouched])

    new_radii = softplus_beta100(scene.radii.data[indices])
    assert torch.allclose(new_radii, torch.ones(2) * 0.1 * 1.5, atol=1e-4)


def test_apply_deformation_radius_scale_mode_max_stretch():
    # Anisotropic, volume-preserving stretch along x: singular values (2.0, 1/sqrt(2), 1/sqrt(2))
    n = 2
    points = torch.zeros(n, 3)
    quaternions = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(n, 4).clone()
    radii = inverse_softplus_beta100(torch.ones(n) * 0.1)
    stretch = 2.0
    compress = 1.0 / (stretch**0.5)
    F = torch.diag(torch.tensor([stretch, compress, compress]))

    scene_volume = _FakeScene(points.clone(), quaternions.clone(), radii.clone())
    indices = torch.tensor([0, 1])
    base_q = quaternions[indices].clone()
    base_r = softplus_beta100(radii[indices]).clone()
    apply_deformation_to_primitives(
        scene_volume, indices, F, base_q, base_r, radius_scale_mode="volume"
    )
    assert torch.allclose(
        softplus_beta100(scene_volume.radii.data), torch.ones(n) * 0.1, atol=1e-4
    )

    scene_max = _FakeScene(points.clone(), quaternions.clone(), radii.clone())
    apply_deformation_to_primitives(
        scene_max, indices, F, base_q, base_r, radius_scale_mode="max_stretch"
    )
    assert torch.allclose(
        softplus_beta100(scene_max.radii.data), torch.ones(n) * 0.1 * stretch, atol=1e-4
    )

    scene_mean = _FakeScene(points.clone(), quaternions.clone(), radii.clone())
    apply_deformation_to_primitives(
        scene_mean, indices, F, base_q, base_r, radius_scale_mode="mean_stretch"
    )
    expected_mean = (stretch + 2 * compress) / 3.0
    assert torch.allclose(
        softplus_beta100(scene_mean.radii.data), torch.ones(n) * 0.1 * expected_mean, atol=1e-4
    )
    # sits strictly between "volume" (1.0x) and "max_stretch" (2.0x)
    assert 0.1 < softplus_beta100(scene_mean.radii.data)[0].item() < 0.1 * stretch


def test_apply_deformation_rejects_unknown_radius_scale_mode():
    n = 1
    points = torch.zeros(n, 3)
    quaternions = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(n, 4).clone()
    radii = inverse_softplus_beta100(torch.ones(n) * 0.1)
    scene = _FakeScene(points, quaternions, radii)
    indices = torch.tensor([0])
    base_q = quaternions[indices].clone()
    base_r = softplus_beta100(radii[indices]).clone()
    try:
        apply_deformation_to_primitives(
            scene, indices, torch.eye(3), base_q, base_r, radius_scale_mode="bogus"
        )
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_apply_deformation_rotates_texel_sv_axis_when_given():
    n = 3
    points = torch.zeros(n, 3)
    quaternions = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(n, 4).clone()
    radii = inverse_softplus_beta100(torch.ones(n) * 0.1)
    # K=n primitives, num_texel_sites=1, sv_dof=1, each axis pointing along +x
    texel_sv_axis = torch.tensor([1.0, 0.0, 0.0]).expand(n, 1, 3).clone()
    scene = _FakeScene(points, quaternions, radii, texel_sv_axis=texel_sv_axis)

    indices = torch.tensor([0, 2])
    base_q = quaternions[indices].clone()
    base_r = softplus_beta100(radii[indices]).clone()
    base_axis = texel_sv_axis[indices].clone()

    theta = torch.tensor(torch.pi / 2)
    Rz90 = torch.tensor(
        [
            [torch.cos(theta), -torch.sin(theta), 0.0],
            [torch.sin(theta), torch.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    apply_deformation_to_primitives(
        scene, indices, Rz90, base_q, base_r, base_texel_sv_axis=base_axis
    )

    # untouched primitive (index 1) keeps its original +x axis
    assert torch.allclose(scene.texel_sv_axis.data[1], texel_sv_axis[1])
    # rotated primitives' axes moved from +x to +y
    expected = torch.tensor([0.0, 1.0, 0.0]).expand(2, 1, 3)
    assert torch.allclose(scene.texel_sv_axis.data[indices], expected, atol=1e-5)


def test_apply_deformation_rotates_normal_for_general_angle():
    # Regression test for a real bug: theta = pi specifically can't distinguish applying +theta from
    # -theta to the normal, since Rz(-pi) == Rz(pi) (same rotation)
    theta = 1.5
    c, s = torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))
    Rz = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    R_base = _random_rotations(1, seed=99)
    q_base = rotmat_to_quaternion_wxyz(R_base)
    normal_old = R_base[0, 0, :]  # row 0, matching get_normals()'s convention

    points = torch.zeros(1, 3)
    radii = inverse_softplus_beta100(torch.ones(1) * 0.1)
    scene = _FakeScene(points, q_base.clone(), radii)

    indices = torch.tensor([0])
    apply_deformation_to_primitives(
        scene, indices, Rz, q_base.clone(), softplus_beta100(radii).clone()
    )

    new_normal = quaternion_wxyz_to_rotmat(scene.quaternions.data)[0, 0, :]
    expected_normal = Rz @ normal_old
    assert torch.allclose(new_normal, expected_normal, atol=1e-4)
