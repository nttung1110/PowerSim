"""Initial MPM state from primitive positions."""

import torch

from powersim.core import transform
from powersim.core.primitive_state import compute_grid_occupancy_volume
from powersim.core.warp_compat import patch_warp_array_owner_kwarg

from powersim.material_estimation.mpm.differentiable_mpm import MPM_Simulator_WARP

_IDENTITY_CACHE = {}


def build_mpm_state_from_points(
    points_world: torch.Tensor,
    material_params: dict,
    device: str,
    scale: float = 0.6,
):
    """Args: points_world: (n, 3) primitive positions in the checkpoint's own world space."""
    patch_warp_array_owner_kwarg()

    n = points_world.shape[0]
    transformed_pos, scale_factor, original_mean_pos = transform.transform2origin(points_world, scale)
    mpm_init_pos = transform.shift2center111(transformed_pos)

    grid_dx = material_params["grid_lim"] / material_params["n_grid"]
    mpm_init_vol = compute_grid_occupancy_volume(
        mpm_init_pos, material_params["n_grid"], grid_dx, uniform=(material_params["material"] == "sand")
    )

    sim = MPM_Simulator_WARP(n, n_grid=material_params["n_grid"], grid_lim=material_params["grid_lim"], device=device)
    sim.load_initial_data_from_torch(
        mpm_init_pos,
        mpm_init_vol,
        tensor_cov=None,
        n_grid=material_params["n_grid"],
        grid_lim=material_params["grid_lim"],
        device=device,
    )
    sim.set_parameters_dict(material_params, device=device)

    x0 = mpm_init_pos
    F_trial0 = torch.eye(3, dtype=torch.float32, device=device).expand(n, 3, 3).contiguous()

    return sim.mpm_state, sim.mpm_model, x0, F_trial0, n, (scale_factor, original_mean_pos)
