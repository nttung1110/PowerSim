"""Draw the impulse arrow on rendered frames."""

import math

import cv2
import numpy as np
import torch

from powersim.foam_evolve.viz.material_field_viz import project_to_pixels


def compute_impulse_indicator_world_space(
    bc_params: list, rotation_matrices: list, scale_factor, original_mean_pos, device: str,
    step_per_frame: int = None, substep_dt: float = None,
):
    """Find the particle_impulse boundary condition (if any) and return its application point and force
    direction in WORLD space."""
    from powersim.core import transform

    impulse_bc = next((bc for bc in bc_params if bc["type"] == "particle_impulse"), None)
    if impulse_bc is None:
        # powersim extension: a push made of velocity-drive stages
        drives = [bc for bc in bc_params if bc.get("indicator") and "velocity" in bc]
        if not drives:
            return None
        main = max(drives, key=lambda bc: sum(v*v for v in bc["velocity"]))
        point_sim = torch.tensor([main["point"]], dtype=torch.float32, device=device)
        point_world = transform.undo_all_transforms(point_sim, rotation_matrices, scale_factor, original_mean_pos)[0]
        vel = torch.tensor(main["velocity"], dtype=torch.float32, device=device)
        direction_world = transform.apply_inverse_rotations((vel / vel.norm()).unsqueeze(0), rotation_matrices)[0]
        active_frames = None
        if step_per_frame is not None and substep_dt is not None:
            end_time = max(bc.get("end_time", 0.0) for bc in drives)
            active_frames = max(1, math.ceil(end_time / (step_per_frame * substep_dt)))
        return point_world, direction_world, point_sim[0], active_frames

    point_sim = torch.tensor([impulse_bc.get("point", [1, 1, 1])], dtype=torch.float32, device=device)
    point_world = transform.undo_all_transforms(point_sim, rotation_matrices, scale_factor, original_mean_pos)[0]

    force = torch.tensor(impulse_bc["force"], dtype=torch.float32, device=device)
    direction_sim = force / force.norm()
    # Direction vectors only need the rotation undone, not the translation/recentering that
    # undo_all_transforms also applies (which is meaningless for a direction)
    direction_world = transform.apply_inverse_rotations(direction_sim.unsqueeze(0), rotation_matrices)[0]

    active_frames = None
    if step_per_frame is not None and substep_dt is not None:
        start_time = impulse_bc.get("start_time", 0.0)
        num_dt = impulse_bc.get("num_dt", 1)
        end_time = start_time + substep_dt * num_dt
        frame_time = step_per_frame * substep_dt  # the solver's actual per-frame time increment,
        # which can be a hair less than time_params["frame_dt"] itself since step_per_frame
        # truncates frame_dt/substep_dt to an int
        active_frames = max(1, math.ceil(end_time / frame_time))

    return point_world, direction_world, point_sim[0], active_frames


def find_tracked_particle_index(mpm_init_pos: torch.Tensor, point_sim: torch.Tensor) -> int:
    """Index (into mpm_init_pos's own ordering, i.e. into --selection) of the simulated particle
    closest to the impulse's rest-space point."""
    d = (mpm_init_pos - point_sim.to(mpm_init_pos.device)).norm(dim=1)
    return int(d.argmin().item())


def draw_force_indicator(
    img: np.ndarray,
    camera,
    point_world: torch.Tensor,
    direction_world: torch.Tensor,
    radius_px: int = 16,
    arrow_len_px: int = 26,
    circle_color=(255, 255, 255),
    arrow_color=(255, 200, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Overlay a circle at point_world's projected pixel position, plus an arrow pointing into it along
    direction_world's projected screen-space direction."""
    eps = 0.02  # world-space offset for the direction probe point, small relative to
    probe = torch.stack([point_world, point_world + eps * direction_world], dim=0)
    rows, cols, in_front = project_to_pixels(camera, probe)
    if not (in_front[0] and in_front[1]):
        return img

    row, col = rows[0], cols[0]
    dr, dc = rows[1] - rows[0], cols[1] - cols[0]
    norm = (dr**2 + dc**2) ** 0.5
    if norm < 1e-6:
        return img
    dr, dc = dr / norm, dc / norm

    center = (int(round(col)), int(round(row)))
    cv2.circle(img, center, radius_px, circle_color, thickness, lineType=cv2.LINE_AA)

    tail = (int(round(col - dc * (radius_px + arrow_len_px))), int(round(row - dr * (radius_px + arrow_len_px))))
    head = (int(round(col - dc * radius_px)), int(round(row - dr * radius_px)))
    cv2.arrowedLine(img, tail, head, arrow_color, thickness + 1, tipLength=0.3, line_type=cv2.LINE_AA)
    return img
