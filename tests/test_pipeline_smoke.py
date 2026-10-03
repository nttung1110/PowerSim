"""Smoke tests of the MPM coupling. The full-pipeline test needs POWERSIM_TEST_CHECKPOINT_CONFIG."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

torch = pytest.importorskip("torch")

from powersim.core import transform
from powersim.core.primitive_state import (
    apply_deformation_to_primitives,
    compute_grid_occupancy_volume,
    polar_decompose,
    rotmat_to_quaternion_wxyz,
)


# ---------------------------------------------------------------------------
# Pure-math: multi-rotation composition (no GPU needed)
# ---------------------------------------------------------------------------


def test_compose_rotation_matrices_matches_chained_application():
    # Guards the same failure class that bit quaternion composition order twice in : a plain per-
    # item loop over apply_deformation_to_primitives would silently only leave the *last*
    device = torch.device("cpu")
    rotation_matrices = transform.generate_rotation_matrices([30.0, 45.0], [0, 2], device)
    R_net = transform.compose_rotation_matrices(rotation_matrices)

    from tests.test_primitive_state_roundtrip import _FakeScene, _random_rotations
    from powersim.core.primitive_state import softplus_beta100, inverse_softplus_beta100

    R_base = _random_rotations(1, seed=42)
    q_base = rotmat_to_quaternion_wxyz(R_base)
    points = torch.zeros(1, 3)
    radii = inverse_softplus_beta100(torch.ones(1) * 0.1)

    # Single call with the composed net rotation.
    scene_composed = _FakeScene(points.clone(), q_base.clone(), radii.clone())
    indices = torch.tensor([0])
    apply_deformation_to_primitives(
        scene_composed, indices, R_net, q_base.clone(), softplus_beta100(radii).clone()
    )

    # Manually chain: apply the first rotation, re-capture as the new base, apply the second on top
    # of that
    scene_chained = _FakeScene(points.clone(), q_base.clone(), radii.clone())
    apply_deformation_to_primitives(
        scene_chained,
        indices,
        rotation_matrices[0],
        q_base.clone(),
        softplus_beta100(radii).clone(),
    )
    q_after_first = scene_chained.quaternions.data.clone()
    r_after_first = softplus_beta100(scene_chained.radii.data).clone()
    apply_deformation_to_primitives(
        scene_chained, indices, rotation_matrices[1], q_after_first, r_after_first
    )

    assert torch.allclose(
        scene_composed.quaternions.data, scene_chained.quaternions.data, atol=1e-4
    )


# ---------------------------------------------------------------------------
# Tier 1: MPM-only smoke test
# ---------------------------------------------------------------------------


def _make_synthetic_cube(n_per_axis=6, low=0.7, high=1.3, device="cpu"):
    lin = torch.linspace(low, high, n_per_axis)
    grid = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), dim=-1)
    return grid.reshape(-1, 3).to(device)


def test_mpm_only_smoke():
    wp = pytest.importorskip("warp")
    if not torch.cuda.is_available():
        pytest.skip("MPM_Simulator_WARP requires CUDA")

    from powersim.foam_evolve.mpm.solver import build_mpm_solver, step_frame

    device = "cuda:0"
    n_grid = 32
    grid_lim = 2.0
    grid_dx = grid_lim / n_grid

    points = _make_synthetic_cube(device=device)
    mpm_init_vol = compute_grid_occupancy_volume(points, n_grid, grid_dx)
    assert mpm_init_vol.shape == (points.shape[0],)
    assert torch.isfinite(mpm_init_vol).all()
    assert (mpm_init_vol > 0).all()

    material_params = {
        "material": "jelly",
        "n_grid": n_grid,
        "grid_lim": grid_lim,
        "E": 2e5,
        "nu": 0.3,
        "density": 200.0,
        "g": [0.0, 0.0, -9.8],
        "rpic_damping": 0.0,
        "grid_v_damping_scale": 1.0,
        "additional_material_params": [],
    }
    bc_params = []
    time_params = {"substep_dt": 1e-4, "frame_dt": 4e-3, "frame_num": 2}

    mpm_solver = build_mpm_solver(points, mpm_init_vol, material_params, bc_params, time_params, device)

    step_per_frame = int(time_params["frame_dt"] / time_params["substep_dt"])
    for frame in range(time_params["frame_num"]):
        step_frame(mpm_solver, step_per_frame, time_params["substep_dt"], frame=frame, device=device)

        x = mpm_solver.export_particle_x_to_torch()
        F = mpm_solver.export_particle_F_to_torch().reshape(-1, 3, 3)
        assert x.shape == points.shape
        assert F.shape == (points.shape[0], 3, 3)
        assert torch.isfinite(x).all()
        assert torch.isfinite(F).all()

        _, volume_scale, _, _ = polar_decompose(F)
        if frame == 0:
            # F should start near-identity after only a few substeps
            assert torch.allclose(volume_scale, torch.ones_like(volume_scale), atol=0.1)

    # Under gravity (g=9.8, no holding boundary condition), the cloud should have net fallen by the
    # final frame
    final_x = mpm_solver.export_particle_x_to_torch()
    assert final_x[:, 2].mean().item() < points[:, 2].mean().item()


# ---------------------------------------------------------------------------
# Tier 2: full pipeline smoke test (needs a real trained checkpoint)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    "POWERSIM_TEST_CHECKPOINT_CONFIG" not in os.environ,
    reason="set POWERSIM_TEST_CHECKPOINT_CONFIG to a trained output/<experiment>/config.yaml to run",
)
def test_full_pipeline_smoke(tmp_path):
    wp = pytest.importorskip("warp")
    if not torch.cuda.is_available():
        pytest.skip("MPM_Simulator_WARP requires CUDA")

    from powersim.foam_evolve import simulate as foam_simulation

    checkpoint_config = os.environ["POWERSIM_TEST_CHECKPOINT_CONFIG"]
    sim_config = os.environ.get(
        "POWERSIM_TEST_SIM_CONFIG",
        str(Path(__file__).resolve().parents[1] / "configs" / "scenes" / "smoke_test.json"),
    )

    # Deliberately invoked from a cwd that is NOT third_parties/powerfoam, to confirm
    # foam_simulation.run's own os.chdir() handling works rather than relying on the caller to cd
    # first
    assert Path.cwd() != (Path(__file__).resolve().parents[1] / "third_parties" / "powerfoam")

    output_dir = tmp_path / "smoke_frames"
    foam_simulation.run(
        checkpoint_config=checkpoint_config,
        sim_config=sim_config,
        output_dir=str(output_dir),
        device="cuda:0",
        num_frames=2,
    )

    frame_files = sorted(output_dir.glob("frame_*.png"))
    assert len(frame_files) == 2

    import cv2
    import numpy as np

    for f in (frame_files[0], frame_files[-1]):
        img = cv2.imread(str(f))
        assert img is not None
        assert img.ndim == 3 and img.shape[2] == 3
        assert np.isfinite(img).all()
