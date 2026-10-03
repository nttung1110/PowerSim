"""Build PhysGaussian's MPM solver from a sim config and advance it frame by frame."""

import torch

from powersim.core.thirdparty_paths import (
    ensure_physgaussian_on_path,
    load_physgaussian_decode_param,
)
from powersim.core.warp_compat import patch_warp_array_owner_kwarg


def set_per_particle_material(mpm_solver, E=None, nu=None, density=None, yield_stress=None, device: str = "cuda:0"):
    """Override set_parameters_dict's scalar-broadcast E/nu/density/yield_stress with per-particle,
    spatially-varying values."""
    from mpm_solver_warp.warp_utils import torch2warp_float

    n = mpm_solver.n_particles
    if E is not None:
        assert E.shape[0] == n, f"E has {E.shape[0]} entries, solver has {n} particles"
        E = E.detach().contiguous().to(dtype=torch.float32, device=device)
        mpm_solver.mpm_model.E = torch2warp_float(E, dvc=device)
    if nu is not None:
        assert nu.shape[0] == n, f"nu has {nu.shape[0]} entries, solver has {n} particles"
        nu = nu.detach().contiguous().to(dtype=torch.float32, device=device)
        mpm_solver.mpm_model.nu = torch2warp_float(nu, dvc=device)
    if yield_stress is not None:
        assert yield_stress.shape[0] == n, f"yield_stress has {yield_stress.shape[0]} entries, solver has {n} particles"
        yield_stress = yield_stress.detach().contiguous().to(dtype=torch.float32, device=device)
        mpm_solver.mpm_model.yield_stress = torch2warp_float(yield_stress, dvc=device)
    if density is not None:
        assert density.shape[0] == n, f"density has {density.shape[0]} entries, solver has {n} particles"
        density = density.detach().contiguous().to(dtype=torch.float32, device=device)
        mpm_solver.reset_densities_and_update_masses(density, device=device)


def build_mpm_solver(
    mpm_init_pos,
    mpm_init_vol,
    material_params: dict,
    bc_params: list,
    time_params: dict,
    device: str = "cuda:0",
    per_particle_material: dict = None,
):
    """Construct and fully configure an MPM_Simulator_WARP from converted particle state."""
    ensure_physgaussian_on_path()
    patch_warp_array_owner_kwarg()
    from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP

    decode_param = load_physgaussian_decode_param()

    mpm_solver = MPM_Simulator_WARP(mpm_init_pos.shape[0])
    mpm_solver.load_initial_data_from_torch(
        mpm_init_pos,
        mpm_init_vol,
        tensor_cov=None,
        n_grid=material_params["n_grid"],
        grid_lim=material_params["grid_lim"],
        device=device,
    )
    mpm_solver.set_parameters_dict(material_params, device=device)
    if per_particle_material:
        set_per_particle_material(mpm_solver, device=device, **per_particle_material)
    # Must run after set_parameters_dict/set_per_particle_material
    from powersim.foam_evolve.mpm.colliders import apply_powersim_bcs, split_powersim_bcs

    upstream_bcs, powersim_bcs = split_powersim_bcs(bc_params)
    decode_param.set_boundary_conditions(mpm_solver, upstream_bcs, time_params)
    apply_powersim_bcs(mpm_solver, powersim_bcs)
    mpm_solver.finalize_mu_lam(device=device)
    return mpm_solver


def step_frame(mpm_solver, step_per_frame: int, substep_dt: float, frame: int = 0, device: str = "cuda:0") -> None:
    """Advance the solver by one rendered frame's worth of substeps."""
    for _ in range(step_per_frame):
        mpm_solver.p2g2p(frame, substep_dt, device=device)
