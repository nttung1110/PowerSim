"""Rigidly driven primitive groups (BC type `powersim_kinematic_rotation`), posed analytically
instead of by MPM."""

import math

import torch

from powersim.core import transform
from powersim.core.primitive_state import apply_deformation_to_primitives

KINEMATIC_BC_TYPE = "powersim_kinematic_rotation"


def rodrigues(axis: torch.Tensor, angle: float) -> torch.Tensor:
    """3x3 right-handed rotation by `angle` (rad) about unit `axis`."""
    k = axis / torch.linalg.norm(axis)
    K = torch.tensor(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]], dtype=axis.dtype, device=axis.device
    )
    eye = torch.eye(3, dtype=axis.dtype, device=axis.device)
    return eye + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def _cylinder_mask(pos: torch.Tensor, bc: dict) -> torch.Tensor:
    point = torch.tensor(bc["point"], dtype=pos.dtype, device=pos.device)
    normal = torch.tensor(bc["normal"], dtype=pos.dtype, device=pos.device)
    normal = normal / torch.linalg.norm(normal)
    half_height, radius = bc["half_height_and_radius"]
    off = pos - point
    along = off @ normal
    radial = torch.linalg.norm(off - along[:, None] * normal[None, :], dim=1)
    mask = (along.abs() < half_height) & (radial < radius)
    for box in bc.get("exclude_boxes", []):
        bp = torch.tensor(box["point"], dtype=pos.dtype, device=pos.device)
        bs = torch.tensor(box["size"], dtype=pos.dtype, device=pos.device)
        mask &= ~((pos - bp).abs() < bs).all(dim=1)
    return mask


def split_kinematic_groups(bc_params, mpm_init_pos, indices, base_quaternions, base_radii, base_texel_sv_axis):
    """Pop every powersim_kinematic_rotation BC out of `bc_params` (in place, so the upstream decoder
    never sees it) and resolve its primitives against the selection's sim-space initial positions."""
    kin = [bc for bc in bc_params if bc["type"] == KINEMATIC_BC_TYPE]
    bc_params[:] = [bc for bc in bc_params if bc["type"] != KINEMATIC_BC_TYPE]
    keep = torch.ones(mpm_init_pos.shape[0], dtype=torch.bool, device=mpm_init_pos.device)
    groups = []
    for bc in kin:
        mask = _cylinder_mask(mpm_init_pos, bc) & keep
        if bc.get("mask_file"):
            full = torch.load(bc["mask_file"], map_location="cpu").to(torch.bool)
            mask &= full[indices.cpu()].to(mask.device)
        keep &= ~mask
        normal = torch.tensor(bc["normal"], dtype=torch.float32, device=mpm_init_pos.device)
        groups.append({
            "bc": bc,
            "indices": indices[mask],
            "p_sim0": mpm_init_pos[mask].clone(),
            "base_quaternions": base_quaternions[mask].clone(),
            "base_radii": base_radii[mask].clone(),
            "base_texel_sv_axis": base_texel_sv_axis[mask].clone(),
            "point": torch.tensor(bc["point"], dtype=torch.float32, device=mpm_init_pos.device),
            "axis": -normal / torch.linalg.norm(normal),  # upstream sign convention, see module docstring
            "normal": normal / torch.linalg.norm(normal),
            "omega": float(bc["rotation_scale"]),
            # translation_scale: velocity along +normal (same field as upstream), so a group can
            # ride along with a body that is itself driven by an enforce_particle_translation BC
            "v_normal": float(bc.get("translation_scale", 0.0)),
            "start": float(bc.get("start_time", 0.0)),
            "end": float(bc.get("end_time", 1e3)),
            # optional later start for the axial translation (a body that spins up before lifting)
            "t_start": float(bc.get("translation_start_time", bc.get("start_time", 0.0))),
        })
        print(
            f"kinematic rotation group: {int(mask.sum())} primitives in cylinder at {bc['point']} "
            f"(normal {bc['normal']}, half-height/radius {bc['half_height_and_radius']}, "
            f"{len(bc.get('exclude_boxes', []))} exclude boxes{', mask ' + bc['mask_file'] if bc.get('mask_file') else ''}) removed from MPM, "
            f"{bc['rotation_scale']} rad/s for t in [{bc.get('start_time', 0.0)}, {bc.get('end_time', 1e3)}]"
        )
    return groups, keep


def advance_kinematic_group(scene, group, t, rotation_matrices, scale_factor, original_mean_pos, update_texel_axis=True):
    """Pose the group at solver time `t`: rigid rotation in sim space, written back to the scene in
    world space exactly like the MPM-driven primitives are."""
    tau = max(0.0, min(t, group["end"]) - group["start"])
    tau_t = max(0.0, min(t, group["end"]) - group.get("t_start", group["start"]))
    theta = group["omega"] * tau
    R = rodrigues(group["axis"], theta)
    p_sim = group["point"] + (group["p_sim0"] - group["point"]) @ R.T + group["normal"] * (group["v_normal"] * tau_t)
    apply_deformation_to_primitives(
        scene,
        group["indices"],
        R,
        group["base_quaternions"],
        group["base_radii"],
        centroid=None,
        base_texel_sv_axis=group["base_texel_sv_axis"] if update_texel_axis else None,
    )
    scene.points.data[group["indices"]] = transform.undo_all_transforms(
        p_sim, rotation_matrices, scale_factor, original_mean_pos
    ).to(scene.points.dtype)
