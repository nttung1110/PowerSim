"""Differentiable MPM substep: out-of-place state updates so wp.Tape can backpropagate."""

import warp as wp

from powersim.core.thirdparty_paths import ensure_physgaussian_on_path
from powersim.core.warp_compat import patch_warp_array_owner_kwarg

ensure_physgaussian_on_path()
patch_warp_array_owner_kwarg()

from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP  # noqa: E402  (also puts mpm_solver_warp's own directory on sys.path)
from warp_utils import MPMModelStruct, MPMStateStruct  # noqa: E402
from mpm_utils import (  # noqa: E402
    compute_mu_lam_from_E_nu,
    zero_grid,
    compute_stress_from_F_trial,
    p2g_apic_with_stress,
    grid_normalization_and_gravity,
    compute_dweight,
)

__all__ = [
    "MPM_Simulator_WARP",
    "MPMModelStruct",
    "MPMStateStruct",
    "MODEL_SCALAR_FIELDS",
    "g2p_out_of_place",
    "grid_normalization_gravity_and_sticky_floor",
    "grid_normalization_gravity_damping_and_cuboid_pin",
    "apply_particle_impulse_kernel",
    "copy_vec3_kernel",
    "copy_mat33_kernel",
    "diff_copy_vec3",
    "diff_copy_mat33",
    "clone_model_with_grad",
    "diff_state_from",
    "next_state_buffers",
    "run_substep_inplace_prefix",
    "run_rollout",
]

# MPMModelStruct fields PhysGaussian's own solver populates as plain scalars/vec3 (not warp arrays)
MODEL_SCALAR_FIELDS = (
    "grid_lim", "n_particles", "n_grid", "dx", "inv_dx", "grid_dim_x",
    "grid_dim_y", "grid_dim_z", "material", "friction_angle", "alpha",
    "gravitational_accelaration", "hardening", "xi", "plastic_viscosity",
    "softening", "rpic_damping", "grid_v_damping_scale", "update_cov_with_F",
)


@wp.kernel
def copy_vec3_kernel(src: wp.array(dtype=wp.vec3), dst: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    dst[i] = src[i]


@wp.kernel
def copy_mat33_kernel(src: wp.array(dtype=wp.mat33), dst: wp.array(dtype=wp.mat33)):
    i = wp.tid()
    dst[i] = src[i]


def diff_copy_vec3(src: wp.array, n: int, device: str) -> wp.array:
    """A fresh requires_grad=True vec3 array, populated from src via an explicit copy KERNEL (not
    wp.clone()/wp.copy."""
    dst = wp.zeros(n, dtype=wp.vec3, device=device, requires_grad=True)
    wp.launch(kernel=copy_vec3_kernel, dim=n, inputs=[src, dst], device=device)
    return dst


def diff_copy_mat33(src: wp.array, n: int, device: str) -> wp.array:
    dst = wp.zeros(n, dtype=wp.mat33, device=device, requires_grad=True)
    wp.launch(kernel=copy_mat33_kernel, dim=n, inputs=[src, dst], device=device)
    return dst


@wp.kernel
def g2p_out_of_place(
    state: MPMStateStruct, next_state: MPMStateStruct, model: MPMModelStruct, dt: float
):
    """PhysGaussian's g2p (mpm_solver_warp/mpm_utils.py), unmodified except that particle_v/x/C/F_trial
    are written into next_state instead of state."""
    p = wp.tid()
    if state.particle_selection[p] == 0:
        grid_pos = state.particle_x[p] * model.inv_dx
        base_pos_x = wp.int(grid_pos[0] - 0.5)
        base_pos_y = wp.int(grid_pos[1] - 0.5)
        base_pos_z = wp.int(grid_pos[2] - 0.5)
        fx = grid_pos - wp.vec3(
            wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
        )
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)
        w = wp.mat33(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
            wp.cw_mul(wc, wc) * 0.5,
        )
        dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5))
        new_v = wp.vec3(0.0, 0.0, 0.0)
        new_C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        new_F = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    ix = base_pos_x + i
                    iy = base_pos_y + j
                    iz = base_pos_z + k
                    dpos = wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                    weight = w[0, i] * w[1, j] * w[2, k]
                    grid_v = state.grid_v_out[ix, iy, iz]
                    new_v = new_v + grid_v * weight
                    new_C = new_C + wp.outer(grid_v, dpos) * (weight * model.inv_dx * 4.0)
                    dweight = compute_dweight(model, w, dw, i, j, k)
                    new_F = new_F + wp.outer(grid_v, dweight)

        next_state.particle_v[p] = new_v
        next_state.particle_x[p] = state.particle_x[p] + dt * new_v
        next_state.particle_C[p] = new_C
        I33 = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        F_tmp = (I33 + new_F * dt) * state.particle_F[p]
        next_state.particle_F_trial[p] = F_tmp


def clone_model_with_grad(model: MPMModelStruct, E_wp: wp.array, nu_wp: wp.array, device: str) -> MPMModelStruct:
    """A fresh MPMModelStruct sharing `model`'s scalar config, with mu/lam as fresh requires_grad=True
    buffers (recomputed once by compute_mu_lam_from_E_nu before the rollout starts) driven by
    caller-supplied E_wp/nu_wp."""
    m = MPMModelStruct()
    for f in MODEL_SCALAR_FIELDS:
        setattr(m, f, getattr(model, f))
    n = model.mu.shape[0]
    m.E = E_wp
    m.nu = nu_wp
    m.mu = wp.zeros(n, dtype=float, device=device, requires_grad=True)
    m.lam = wp.zeros(n, dtype=float, device=device, requires_grad=True)
    m.yield_stress = wp.clone(model.yield_stress, device=device, requires_grad=False)
    return m


def diff_state_from(state: MPMStateStruct, n: int, device: str) -> MPMStateStruct:
    """The rollout's starting state: fresh requires_grad=True buffers for every field a kernel will
    ever write to across the chain (x, v, F, F_trial, C, stress, grid_*), copied via kernel from
    `state`'s current values."""
    s = MPMStateStruct()
    s.particle_x = diff_copy_vec3(state.particle_x, n, device)
    s.particle_v = diff_copy_vec3(state.particle_v, n, device)
    s.particle_F = diff_copy_mat33(state.particle_F, n, device)
    s.particle_F_trial = diff_copy_mat33(state.particle_F_trial, n, device)
    s.particle_C = diff_copy_mat33(state.particle_C, n, device)
    s.particle_stress = diff_copy_mat33(state.particle_stress, n, device)
    s.particle_init_cov = wp.clone(state.particle_init_cov, device=device, requires_grad=False)
    s.particle_cov = wp.clone(state.particle_cov, device=device, requires_grad=False)
    s.particle_R = wp.clone(state.particle_R, device=device, requires_grad=False)
    s.particle_vol = wp.clone(state.particle_vol, device=device, requires_grad=False)
    s.particle_mass = wp.clone(state.particle_mass, device=device, requires_grad=False)
    s.particle_density = wp.clone(state.particle_density, device=device, requires_grad=False)
    s.particle_Jp = wp.clone(state.particle_Jp, device=device, requires_grad=False)
    s.particle_selection = wp.clone(state.particle_selection, device=device, requires_grad=False)
    n_grid = state.grid_m.shape[0]
    s.grid_m = wp.zeros((n_grid, n_grid, n_grid), dtype=float, device=device, requires_grad=True)
    s.grid_v_in = wp.zeros((n_grid, n_grid, n_grid), dtype=wp.vec3, device=device, requires_grad=True)
    s.grid_v_out = wp.zeros((n_grid, n_grid, n_grid), dtype=wp.vec3, device=device, requires_grad=True)
    return s


def next_state_buffers(prev: MPMStateStruct, n: int, device: str) -> MPMStateStruct:
    """Fresh write-target buffers for g2p_out_of_place."""
    s = MPMStateStruct()
    s.particle_x = wp.zeros(n, dtype=wp.vec3, device=device, requires_grad=True)
    s.particle_v = wp.zeros(n, dtype=wp.vec3, device=device, requires_grad=True)
    s.particle_F = wp.zeros(n, dtype=wp.mat33, device=device, requires_grad=True)
    s.particle_F_trial = wp.zeros(n, dtype=wp.mat33, device=device, requires_grad=True)
    s.particle_C = wp.zeros(n, dtype=wp.mat33, device=device, requires_grad=True)
    s.particle_stress = wp.zeros(n, dtype=wp.mat33, device=device, requires_grad=True)
    s.particle_init_cov = prev.particle_init_cov
    s.particle_cov = prev.particle_cov
    s.particle_R = prev.particle_R
    s.particle_vol = prev.particle_vol
    s.particle_mass = prev.particle_mass
    s.particle_density = prev.particle_density
    s.particle_Jp = prev.particle_Jp
    s.particle_selection = prev.particle_selection
    n_grid = prev.grid_m.shape[0]
    s.grid_m = wp.zeros((n_grid, n_grid, n_grid), dtype=float, device=device, requires_grad=True)
    s.grid_v_in = wp.zeros((n_grid, n_grid, n_grid), dtype=wp.vec3, device=device, requires_grad=True)
    s.grid_v_out = wp.zeros((n_grid, n_grid, n_grid), dtype=wp.vec3, device=device, requires_grad=True)
    return s


@wp.kernel
def grid_normalization_gravity_and_sticky_floor(
    state: MPMStateStruct, model: MPMModelStruct, dt: float,
    damping_scale: float, point: wp.vec3, normal: wp.vec3, floor_mode: int,
):
    """grid_normalization_and_gravity's own logic (PhysGaussian's kernel, mpm_utils.py) fused with
    PhysGaussian's own add_damping_via_grid (a per-substep grid_v_out *= damping_scale) and a
    single, always-on, STICKY."""
    grid_x, grid_y, grid_z = wp.tid()
    if state.grid_m[grid_x, grid_y, grid_z] > 1e-15:
        v_out = state.grid_v_in[grid_x, grid_y, grid_z] * (1.0 / state.grid_m[grid_x, grid_y, grid_z])
        v_out = v_out + dt * model.gravitational_accelaration
        if damping_scale < 1.0:
            v_out = v_out * damping_scale
        offset = wp.vec3(
            float(grid_x) * model.dx - point[0],
            float(grid_y) * model.dx - point[1],
            float(grid_z) * model.dx - point[2],
        )
        if wp.dot(offset, normal) < 0.0:
            if floor_mode == 0:
                v_out = wp.vec3(0.0, 0.0, 0.0)
            else:
                normal_component = wp.dot(v_out, normal)
                if floor_mode == 1:
                    v_out = v_out - normal_component * normal
                else:
                    v_out = v_out - wp.min(normal_component, 0.0) * normal
        state.grid_v_out[grid_x, grid_y, grid_z] = v_out


@wp.kernel
def grid_normalization_gravity_damping_and_cuboid_pin(
    state: MPMStateStruct, model: MPMModelStruct, dt: float,
    damping_scale: float, pin_point: wp.vec3, pin_size: wp.vec3,
):
    """grid_normalization_and_gravity fused with damping and a fixed (velocity=[0,0,0]) cuboid pin."""
    grid_x, grid_y, grid_z = wp.tid()
    if state.grid_m[grid_x, grid_y, grid_z] > 1e-15:
        v_out = state.grid_v_in[grid_x, grid_y, grid_z] * (1.0 / state.grid_m[grid_x, grid_y, grid_z])
        v_out = v_out + dt * model.gravitational_accelaration
        if damping_scale < 1.0:
            v_out = v_out * damping_scale
        offset = wp.vec3(
            float(grid_x) * model.dx - pin_point[0],
            float(grid_y) * model.dx - pin_point[1],
            float(grid_z) * model.dx - pin_point[2],
        )
        if wp.abs(offset[0]) < pin_size[0] and wp.abs(offset[1]) < pin_size[1] and wp.abs(offset[2]) < pin_size[2]:
            v_out = wp.vec3(0.0, 0.0, 0.0)
        state.grid_v_out[grid_x, grid_y, grid_z] = v_out


@wp.kernel
def apply_particle_impulse_kernel(
    state: MPMStateStruct, dt: float, impulse_force: wp.array(dtype=wp.vec3),
):
    """Ports PhysGaussian's own add_impulse_on_particles / apply_force kernel (mpm_solver_warp.py
    ~L885-899): state.particle_v[p] += (force[p] / mass[p]) * dt."""
    p = wp.tid()
    f = impulse_force[p]
    mass = state.particle_mass[p]
    # atomic_add, NOT `particle_v[p] = particle_v[p] + delta`: the tape adjoint of that read-modify-
    # write self-assignment doubles adj_particle_v every substep the impulse is applied
    wp.atomic_add(state.particle_v, p, wp.vec3(f[0] / mass, f[1] / mass, f[2] / mass) * dt)


def run_substep_inplace_prefix(
    state: MPMStateStruct, model: MPMModelStruct, n: int, dt: float, device: str,
    floor_point: wp.vec3 = None, floor_normal: wp.vec3 = None, damping_scale: float = 1.0,
    pin_point: wp.vec3 = None, pin_size: wp.vec3 = None, impulse_force: wp.array = None,
    floor_mode: int = 0,
) -> None:
    """Every substep kernel except g2p."""
    grid_size = (model.grid_dim_x, model.grid_dim_y, model.grid_dim_z)
    wp.launch(kernel=zero_grid, dim=grid_size, inputs=[state, model], device=device)
    if impulse_force is not None:
        wp.launch(kernel=apply_particle_impulse_kernel, dim=n, inputs=[state, dt, impulse_force], device=device)
    wp.launch(kernel=compute_stress_from_F_trial, dim=n, inputs=[state, model, dt], device=device)
    wp.launch(kernel=p2g_apic_with_stress, dim=n, inputs=[state, model, dt], device=device)
    if floor_point is not None:
        wp.launch(
            kernel=grid_normalization_gravity_and_sticky_floor, dim=grid_size,
            inputs=[state, model, dt, damping_scale, floor_point, floor_normal, floor_mode], device=device,
        )
    elif pin_point is not None:
        wp.launch(
            kernel=grid_normalization_gravity_damping_and_cuboid_pin, dim=grid_size,
            inputs=[state, model, dt, damping_scale, pin_point, pin_size], device=device,
        )
    else:
        wp.launch(kernel=grid_normalization_and_gravity, dim=grid_size, inputs=[state, model, dt], device=device)


def run_rollout(
    state: MPMStateStruct,
    model: MPMModelStruct,
    n: int,
    num_substeps: int,
    dt: float,
    device: str,
    recompute_mu_lam: bool = True,
    floor_point: wp.vec3 = None,
    floor_normal: wp.vec3 = None,
    damping_scale: float = 1.0,
    pin_point: wp.vec3 = None,
    pin_size: wp.vec3 = None,
    impulse_force: wp.array = None,
    impulse_substeps: int = 0,
    floor_mode: int = 0,
) -> MPMStateStruct:
    """Runs num_substeps of MPM starting from `state` (already requires_grad-configured, e.g. via
    diff_state_from), returning the final state."""
    if recompute_mu_lam:
        wp.launch(kernel=compute_mu_lam_from_E_nu, dim=n, inputs=[state, model], device=device)
    for step_idx in range(num_substeps):
        this_step_impulse = impulse_force if (impulse_force is not None and step_idx < impulse_substeps) else None
        run_substep_inplace_prefix(
            state, model, n, dt, device, floor_point, floor_normal, damping_scale,
            pin_point, pin_size, this_step_impulse, floor_mode,
        )
        next_state = next_state_buffers(state, n, device)
        wp.launch(kernel=g2p_out_of_place, dim=n, inputs=[state, next_state, model, dt], device=device)
        state = next_state
    return state
