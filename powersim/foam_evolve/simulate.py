"""Simulate a PowerFoam scene with MPM and render every frame.

    python -m powersim.foam_evolve.simulate --checkpoint-config <ckpt>/config.yaml \
        --sim-config config/<scene>.json --output-dir <dir> [--selection mask.pt] [--compile-video]

The sim config follows PhysGaussian's JSON schema. Primitives outside --selection stay static."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm

from powersim.core import transform
from powersim.core.primitive_state import (
    apply_deformation_to_primitives,
    compute_grid_occupancy_volume,
    softplus_beta100,
)
from powersim.core.checkpoint import load_powerfoam_checkpoint
from powersim.foam_evolve.mpm.solver import build_mpm_solver, step_frame
from powersim.foam_evolve.mpm.kinematic import advance_kinematic_group, split_kinematic_groups
from powersim.foam_evolve.viz.force_indicator import compute_impulse_indicator_world_space, draw_force_indicator, find_tracked_particle_index
from powersim.core.frame_renderer import render_rgb, rgb_to_uint8_image, save_uint8_rgb_png
from powersim.core.video import ffmpeg_exe
from powersim.foam_evolve.viz.material_field_viz import render_material_field_triptych
from powersim.core.thirdparty_paths import (
    POWERFOAM_ROOT,
    ensure_physgaussian_on_path,
    load_physgaussian_decode_param,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-config", required=True, help="PowerFoam checkpoint config.yaml (model.pt next to it)")
    parser.add_argument("--sim-config", required=True, help="simulation config JSON (PhysGaussian schema)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--selection",
        default=None,
        help="boolean mask (.pt) over the checkpoint's primitives: the ones driven by MPM (default: all)",
    )
    parser.add_argument(
        "--recenter-selection", default=None,
        help="mask whose bbox defines the MPM frame instead of --selection's, e.g. the unextended selection a sim config was written for",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", default="train", help="camera split: train or test")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="camera index within the split (default: the sim config's default_camera_index)",
    )
    parser.add_argument("--num-frames", type=int, default=None, help="number of frames (default: the sim config's frame_num)")
    parser.add_argument(
        "--drop-height",
        type=float,
        default=None,
        help="raise the object by this much (MPM units) before the simulation starts",
    )
    parser.add_argument(
        "--compile-video",
        action="store_true",
        help="also write output.mp4 from the frames",
    )
    parser.add_argument(
        "--material-field",
        default=None,
        help="per-primitive material file (.pt with indices, E, nu, density); uncovered primitives use the config's scalars",
    )
    parser.add_argument(
        "--fill-uncovered-nearest",
        action="store_true",
        help="primitives the material field does not cover take the values of their nearest covered primitive",
    )
    parser.add_argument(
        "--show-force-indicator",
        action="store_true",
        help="draw the impulse as an arrow on the frames while it acts",
    )
    parser.add_argument(
        "--force-indicator-frames",
        type=int,
        default=None,
        help="number of frames to draw the force indicator on (default: while the impulse lasts)",
    )
    parser.add_argument(
        "--freeze",
        nargs="+",
        default=[],
        choices=["dipole", "radii", "texel_axis"],
        help="ablation: keep these attributes at their rest values instead of updating them from F",
    )
    parser.add_argument(
        "--save-positions",
        action="store_true",
        help="also save the simulated primitives' world positions every frame (positions/positions_%%04d.pt)",
    )
    parser.add_argument(
        "--background",
        type=float, nargs=3, default=None, metavar=("R", "G", "B"),
        help="composite frames over this colour (0-1) instead of black",
    )
    parser.add_argument(
        "--passive-follow",
        action="store_true",
        help="primitives outside --selection follow their nearest simulated primitive instead of staying static",
    )
    parser.add_argument(
        "--cull-density",
        type=float, default=None,
        help="remove simulated primitives whose density is below this value from the simulation and the render",
    )
    parser.add_argument(
        "--hide",
        default=None,
        help="boolean mask (.pt) of primitives to make invisible for the whole run",
    )
    parser.add_argument(
        "--displace",
        default=None,
        help="(N, 3) world-space displacements (.pt) added to the primitives before the simulation",
    )
    parser.add_argument(
        "--appearance-source",
        default=None,
        help="boolean mask (.pt) of primitives with trustworthy appearance; other simulated primitives copy the look of their nearest one",
    )
    parser.add_argument(
        "--radius-mode",
        default="deformation",
        choices=["deformation", "neighbor"],
        help="how radii follow the simulation: 'deformation' = det(F)^(1/3); 'neighbor' = from nearest-neighbour distances (for plastic materials, whose F is elastic only)",
    )
    return parser.parse_args()


def build_per_particle_material(
    n_scene_points: int,
    indices: torch.Tensor,
    material_field_path: str,
    material_params: dict,
    device: str,
    positions: torch.Tensor = None,
    fill_uncovered_nearest: bool = False,
):
    """Build (E, nu, density, covered) tensors aligned to `indices`'s own order, using
    material_field_path's predicted values where covered and material_params's scalar defaults
    everywhere else."""
    data = torch.load(material_field_path, map_location=device)
    full_E = torch.full((n_scene_points,), float(material_params["E"]), device=device)
    full_nu = torch.full((n_scene_points,), float(material_params["nu"]), device=device)
    full_density = torch.full((n_scene_points,), float(material_params["density"]), device=device)
    full_covered = torch.zeros(n_scene_points, dtype=torch.bool, device=device)

    covered = data["covered"].to(device)
    field_indices = data["indices"].to(device)[covered]
    # yield_stress is optional: only the plastic materials read it
    full_yield = None
    if "yield_stress" in data:
        full_yield = torch.full((n_scene_points,), float(material_params.get("yield_stress", 0.0)), device=device)
        full_yield[field_indices] = data["yield_stress"].to(device)[covered]
    full_E[field_indices] = data["E"].to(device)[covered]
    full_nu[field_indices] = data["nu"].to(device)[covered]
    full_density[field_indices] = data["density"].to(device)[covered]
    full_covered[field_indices] = True

    E, nu, density, is_covered = full_E[indices], full_nu[indices], full_density[indices], full_covered[indices]
    yield_pp = None if full_yield is None else full_yield[indices]

    if fill_uncovered_nearest and (~is_covered).any():
        if positions is None:
            raise ValueError("positions is required when fill_uncovered_nearest=True")
        covered_idx = is_covered.nonzero(as_tuple=True)[0]
        uncovered_idx = (~is_covered).nonzero(as_tuple=True)[0]
        if covered_idx.numel() > 0:
            covered_pos = positions[covered_idx].detach().cpu().numpy()
            uncovered_pos = positions[uncovered_idx].detach().cpu().numpy()
            tree = cKDTree(covered_pos)
            _, nn = tree.query(uncovered_pos, k=1)
            nn_idx = covered_idx[torch.as_tensor(nn, device=device)]
            E = E.clone()
            nu = nu.clone()
            density = density.clone()
            is_covered = is_covered.clone()
            E[uncovered_idx] = E[nn_idx]
            nu[uncovered_idx] = nu[nn_idx]
            density[uncovered_idx] = density[nn_idx]
            if yield_pp is not None:
                yield_pp = yield_pp.clone()
                yield_pp[uncovered_idx] = yield_pp[nn_idx]
            is_covered[uncovered_idx] = True

    return E, nu, density, is_covered, yield_pp


def run(
    checkpoint_config: str,
    sim_config: str,
    output_dir: str,
    device: str = "cuda:0",
    split: str = "train",
    camera_index: int = None,
    num_frames: int = None,
    drop_height: float = None,
    compile_video: bool = False,
    selection: str = None,
    recenter_selection: str = None,
    material_field: str = None,
    fill_uncovered_nearest: bool = False,
    show_force_indicator: bool = False,
    force_indicator_frames: int = None,
    save_positions: bool = False,
    freeze: tuple = (),
    radius_mode: str = "deformation",
    appearance_source: str = None,
    displace: str = None,
    hide: str = None,
    passive_follow: bool = False,
    background=None,
    cull_density: float = None,
):
    # load_powerfoam_checkpoint needs cwd == third_party/powerfoam for the checkpoint's relative
    # data_path to resolve
    checkpoint_config = str(Path(checkpoint_config).resolve())
    original_cwd = os.getcwd()
    os.chdir(POWERFOAM_ROOT)
    try:
        checkpoint = load_powerfoam_checkpoint(checkpoint_config, device=device, split=split)
    finally:
        os.chdir(original_cwd)

    scene = checkpoint.scene

    # decode_param.py itself does `from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP`
    # at its top level, so mpm_solver_warp must already be importable before
    ensure_physgaussian_on_path()
    decode_param = load_physgaussian_decode_param()
    material_params, bc_params, time_params, preprocessing_params, camera_params = (
        decode_param.decode_param_json(sim_config)
    )
    # Boundary conditions may name a per-primitive mask (`mask_file`); a relative path is taken
    # relative to the sim config's own directory so released configs stay portable
    _cfg_dir = Path(sim_config).resolve().parent
    for _bc in bc_params:
        if _bc.get("mask_file") and not os.path.isabs(_bc["mask_file"]):
            _bc["mask_file"] = str((_cfg_dir / _bc["mask_file"]).resolve())
    if num_frames is not None:
        time_params["frame_num"] = num_frames

    # "powersim_drop_height" is a powersim-specific extension, not part of PhysGaussian's schema
    with open(sim_config) as f:
        raw_sim_config = json.load(f)
    # powersim_extra_material_params: extra keys handed straight to
    # MPM_Simulator_WARP.set_parameters_dict
    extra_mat = raw_sim_config.get("powersim_extra_material_params", {})
    if extra_mat:
        material_params.update(extra_mat)
        print(f"extra material params passed to the solver: {extra_mat}")
    resolved_drop_height = (
        drop_height if drop_height is not None else raw_sim_config.get("powersim_drop_height", 0.0)
    )

    # --camera-index (when explicitly passed) overrides the sim config's own
    # camera_params["default_camera_index"] (PhysGaussian schema field, decode_param_json default 0)
    resolved_camera_index = (
        camera_index if camera_index is not None else camera_params["default_camera_index"]
    )
    num_cameras = len(checkpoint.data_handler.cameras)
    if not (0 <= resolved_camera_index < num_cameras):
        raise ValueError(
            f"camera index {resolved_camera_index} out of range for {num_cameras} "
            f"cameras in split={split!r}"
        )
    camera = checkpoint.data_handler.cameras[resolved_camera_index]

    if selection is not None:
        selection_mask = torch.load(selection, map_location="cpu")
        if selection_mask.shape[0] != scene.points.shape[0]:
            raise ValueError(
                f"--selection has {selection_mask.shape[0]} entries but the checkpoint has "
                f"{scene.points.shape[0]} primitives"
            )
        indices = selection_mask.nonzero(as_tuple=True)[0].to(scene.points.device)
        print(f"simulating {indices.shape[0]}/{scene.points.shape[0]} primitives (from --selection)")
    else:
        indices = torch.arange(scene.points.shape[0], device=scene.points.device)

    if displace is not None:
        d = torch.load(displace, map_location="cpu").to(dtype=scene.points.dtype, device=scene.points.device)
        if d.shape != scene.points.shape:
            raise ValueError(f"--displace has shape {tuple(d.shape)}, checkpoint points are {tuple(scene.points.shape)}")
        with torch.no_grad():
            scene.points.data += d
        moved = int((d.norm(dim=1) > 0).sum())
        print(f"--displace: {moved} primitives moved, max |d| {float(d.norm(dim=1).max()):.5f} world units")

    if hide is not None:
        hmask = torch.load(hide, map_location="cpu").to(torch.bool)
        if hmask.shape[0] != scene.points.shape[0]:
            raise ValueError(f"--hide mask has {hmask.shape[0]} entries, checkpoint has {scene.points.shape[0]}")
        hmask = hmask.to(scene.density.device)
        with torch.no_grad():
            scene.density.data[hmask] = -50.0
        overlap = hmask[indices]
        if overlap.any():
            indices = indices[~overlap]
        print(f"--hide: {int(hmask.sum())} primitives made invisible ({int(overlap.sum())} of them removed from the selection)")

    if cull_density is not None:
        dens = scene.get_density().detach()[indices]
        q = torch.quantile(dens.float(), torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], device=dens.device)).tolist()
        cull = dens < cull_density
        print(f"--cull-density {cull_density}: simulated density p5/25/50/75/95 = {[round(v, 4) for v in q]}; "
              f"culling {int(cull.sum())}/{indices.numel()} ({100 * float(cull.float().mean()):.1f}%)")
        with torch.no_grad():
            scene.density.data[indices[cull]] = -1.0   # softplus(-100)/100 ~ 0: invisible
        indices = indices[~cull]

    if appearance_source is not None:
        src_mask = torch.load(appearance_source, map_location="cpu")
        if src_mask.shape[0] != scene.points.shape[0]:
            raise ValueError(f"--appearance-source mask has {src_mask.shape[0]} entries, checkpoint has {scene.points.shape[0]}")
        src_idx = src_mask.nonzero(as_tuple=True)[0].to(scene.points.device)
        sel_cpu = indices.cpu(); src_cpu = src_idx.cpu()
        fill = sel_cpu[~torch.isin(sel_cpu, src_cpu)]
        if fill.numel() and src_cpu.numel():
            tree = cKDTree(scene.points.data[src_idx].detach().float().cpu().numpy())
            _, nn = tree.query(scene.points.data[fill.to(scene.points.device)].detach().float().cpu().numpy(), k=1)
            src_of = src_idx[torch.from_numpy(nn).to(scene.points.device)]
            fill_dev = fill.to(scene.points.device)
            with torch.no_grad():
                for name in ("texel_sv_rgb", "texel_sv_axis", "texel_height", "texel_sites", "density"):
                    getattr(scene, name).data[fill_dev] = getattr(scene, name).data[src_of]
            print(f"--appearance-source: {fill.numel()} simulated primitives took the texel attributes/density of their nearest of {src_cpu.numel()} source primitives")

    passive = None
    if passive_follow:
        all_idx = torch.arange(scene.points.shape[0], device=scene.points.device)
        passive = all_idx[~torch.isin(all_idx, indices)]
        if passive.numel():
            sel0 = scene.points.data[indices].detach().float()
            tree = cKDTree(sel0.cpu().numpy())
            _, nn = tree.query(scene.points.data[passive].detach().float().cpu().numpy(), k=1)
            passive_nn = torch.as_tensor(nn, device=scene.points.device)
            passive_pos0 = scene.points.data[passive].detach().clone()
            sel_pos0 = sel0.clone()
            print(f"--passive-follow: {passive.numel()} passive primitives tied to their nearest of {indices.numel()} simulated ones")
        else:
            passive = None

    base_points = scene.points.data[indices].clone()
    base_quaternions = scene.quaternions.data[indices].clone()
    base_radii = softplus_beta100(scene.radii.data[indices]).clone()
    base_texel_sv_axis = scene.texel_sv_axis.data[indices].clone()

    rotation_matrices = transform.generate_rotation_matrices(
        preprocessing_params["rotation_degree"], preprocessing_params["rotation_axis"], device
    )
    if rotation_matrices:
        R_net = transform.compose_rotation_matrices(rotation_matrices)
        apply_deformation_to_primitives(
            scene,
            indices,
            R_net,
            base_quaternions,
            base_radii,
            centroid=None,
            base_texel_sv_axis=base_texel_sv_axis,
        )
        scene.points.data[indices] = transform.apply_rotations(base_points, rotation_matrices)

    # base_* is the F=I reference for every subsequent MPM-driven call
    base_quaternions = scene.quaternions.data[indices].clone()
    base_radii = softplus_beta100(scene.radii.data[indices]).clone()
    base_texel_sv_axis = scene.texel_sv_axis.data[indices].clone()

    sim_scale = preprocessing_params["scale"]
    if recenter_selection is not None:
        recenter_mask = torch.load(recenter_selection, map_location="cpu").to(torch.bool)
        if recenter_mask.shape[0] != scene.points.shape[0]:
            raise ValueError(f"--recenter-selection has {recenter_mask.shape[0]} entries but the checkpoint has {scene.points.shape[0]} primitives")
        recenter_idx = recenter_mask.nonzero(as_tuple=True)[0].to(scene.points.device)
        _, scale_factor, original_mean_pos = transform.transform2origin(scene.points.data[recenter_idx], sim_scale)
        transformed_pos = (scene.points.data[indices] - original_mean_pos) * scale_factor
        print(f"MPM frame recentred on --recenter-selection ({recenter_idx.shape[0]} primitives)")
    else:
        transformed_pos, scale_factor, original_mean_pos = transform.transform2origin(
            scene.points.data[indices], sim_scale
        )
    mpm_init_pos = transform.shift2center111(transformed_pos)

    # powersim_kinematic_rotation groups (powersim/mpm/kinematic.py): resolved against the
    # selection's sim-space start positions, then removed from the MPM particle set
    kinematic_groups, keep_mask = split_kinematic_groups(
        bc_params, mpm_init_pos, indices, base_quaternions, base_radii, base_texel_sv_axis
    )
    if kinematic_groups:
        indices = indices[keep_mask]
        base_quaternions = base_quaternions[keep_mask]
        base_radii = base_radii[keep_mask]
        base_texel_sv_axis = base_texel_sv_axis[keep_mask]
        mpm_init_pos = mpm_init_pos[keep_mask]
        print(f"{int(keep_mask.sum())} primitives remain MPM-driven")
    if resolved_drop_height != 0.0:
        # Purely a starting-condition offset fed into the solver
        drop_offset = torch.tensor([0.0, 0.0, resolved_drop_height], device=device, dtype=mpm_init_pos.dtype)
        mpm_init_pos = mpm_init_pos + drop_offset

    grid_dx = material_params["grid_lim"] / material_params["n_grid"]
    mpm_init_vol = compute_grid_occupancy_volume(
        mpm_init_pos, material_params["n_grid"], grid_dx, uniform=(material_params["material"] == "sand")
    )

    per_particle_material = None
    material_field_data = None
    if material_field is not None:
        n_before_fill = None
        if fill_uncovered_nearest:
            _, _, _, covered_before, _ = build_per_particle_material(
                scene.points.shape[0], indices, material_field, material_params, device
            )
            n_before_fill = int(covered_before.sum())
        E_pp, nu_pp, density_pp, covered_pp, yield_pp = build_per_particle_material(
            scene.points.shape[0], indices, material_field, material_params, device,
            positions=scene.points.data[indices], fill_uncovered_nearest=fill_uncovered_nearest,
        )
        per_particle_material = {"E": E_pp, "nu": nu_pp, "density": density_pp}
        if yield_pp is not None:
            per_particle_material["yield_stress"] = yield_pp
            print(f"material field: per-particle yield_stress {float(yield_pp.min()):.1f} .. {float(yield_pp.max()):.1f}")
        material_field_data = (E_pp, nu_pp, density_pp, covered_pp)
        if fill_uncovered_nearest:
            print(
                f"material field: {n_before_fill}/{indices.shape[0]} simulated primitives directly "
                f"covered by {material_field}, {int(covered_pp.sum()) - n_before_fill} filled from "
                f"their nearest covered primitive, {indices.shape[0] - int(covered_pp.sum())} still "
                f"fall back to scalar material_params"
            )
        else:
            print(
                f"material field: {int(covered_pp.sum())}/{indices.shape[0]} simulated primitives "
                f"covered by {material_field}, rest fall back to scalar material_params"
            )

    mpm_solver = build_mpm_solver(
        mpm_init_pos, mpm_init_vol, material_params, bc_params, time_params, device, per_particle_material
    )
    # powersim extension: an enforce_particle_translation / enforce_particle_velocity_rotation BC
    # may carry "mask_file" (bool .pt over the checkpoint's primitives)
    _vm_types = ("enforce_particle_translation", "enforce_particle_velocity_rotation")
    _vm_bcs = [bc for bc in bc_params if bc.get("type") in _vm_types]
    for k, bc in enumerate(_vm_bcs):
        if not bc.get("mask_file"):
            continue
        import warp as wp
        full = torch.load(bc["mask_file"], map_location="cpu").to(torch.bool)
        sub = full[indices.cpu()].to(device)
        param = mpm_solver.particle_velocity_modifier_params[k]
        box_mask = wp.to_torch(param.mask).to(torch.bool)
        combined = (box_mask & sub).to(torch.int32).contiguous()
        param.mask = wp.from_torch(combined, dtype=wp.int32)
        print(f"BC {k} ({bc['type']}): mask_file {Path(bc['mask_file']).name} -> {int(combined.sum())} particles (box alone had {int(box_mask.sum())})")

    step_per_frame = int(time_params["frame_dt"] / time_params["substep_dt"])
    substep_dt = time_params["substep_dt"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    direction_world = None
    tracked_idx = None
    indicator_frames = 0
    if show_force_indicator:
        indicator = compute_impulse_indicator_world_space(
            bc_params, rotation_matrices, scale_factor, original_mean_pos, device,
            step_per_frame=step_per_frame, substep_dt=substep_dt,
        )
        if indicator is None:
            print("--show-force-indicator given but sim config has no particle_impulse BC or indicator-marked drive; skipping.")
        else:
            _, direction_world, point_sim, auto_active_frames = indicator
            # Track the actual simulated particle nearest the impulse's rest-space point, rather
            # than freezing the marker at its frame-0 world position
            tracked_idx = find_tracked_particle_index(mpm_init_pos, point_sim)
            indicator_frames = force_indicator_frames if force_indicator_frames is not None else auto_active_frames
            print(f"force indicator: showing for {indicator_frames} frames (until the impulse is released)")

    if material_field_data is not None:
        E_pp, nu_pp, density_pp, covered_pp = material_field_data
    else:
        # Constant-material run: no per-primitive variation to show, but render the same triptych
        # anyway
        n_sim = indices.shape[0]
        E_pp = torch.full((n_sim,), float(material_params["E"]), device=device)
        nu_pp = torch.full((n_sim,), float(material_params["nu"]), device=device)
        density_pp = torch.full((n_sim,), float(material_params["density"]), device=device)
        covered_pp = None

    # A rerun with fewer frames than a previous run in the same directory would otherwise leave the
    # old tail in place, and --compile-video's frame_%04d pattern would splice it onto the
    stale = sorted(output_path.glob("frame_*.png"))
    if stale:
        for f_ in stale:
            f_.unlink()
        print(f"removed {len(stale)} frame_*.png left in {output_path} by a previous run")

    backdrop = rgb_to_uint8_image(render_rgb(scene, camera, background))
    render_material_field_triptych(
        backdrop,
        camera,
        scene.points.data[indices],
        E_pp,
        nu_pp,
        density_pp,
        output_path=str(output_path / "material_field.png"),
        covered=covered_pp,
        title_prefix=f"{Path(checkpoint_config).parent.name.split('@')[0]}: ",
    )
    print(f"Saved {output_path / 'material_field.png'}")

    def _knn_mean_dist(points_world, k=8):
        tree = cKDTree(points_world)
        d, _ = tree.query(points_world, k=k + 1)
        return d[:, 1:].mean(axis=1)

    if radius_mode == "neighbor":
        from powersim.core.primitive_state import inverse_softplus_beta100
        knn_d0 = _knn_mean_dist(scene.points.data[indices].detach().float().cpu().numpy())
        knn_d0 = np.maximum(knn_d0, 1e-6)
        print(f"radius mode 'neighbor': initial mean 8-NN distance {knn_d0.mean():.5f} (world units)")

    for frame in tqdm(range(time_params["frame_num"]), desc="Simulating"):
        step_frame(mpm_solver, step_per_frame, substep_dt, frame=frame, device=device)

        x = mpm_solver.export_particle_x_to_torch()
        # export_particle_R_to_torch() is deliberately not called: polar_decompose(F) below derives
        # its own R via SVD and needs the full singular-value spectrum S anyway
        F = mpm_solver.export_particle_F_to_torch().reshape(-1, 3, 3)

        apply_deformation_to_primitives(
            scene,
            indices,
            F,
            base_quaternions,
            base_radii,
            centroid=None,
            base_texel_sv_axis=None if "texel_axis" in freeze else base_texel_sv_axis,
            update_quaternions="dipole" not in freeze,
            update_radii="radii" not in freeze,
        )
        scene.points.data[indices] = transform.undo_all_transforms(
            x, rotation_matrices, scale_factor, original_mean_pos
        )
        if radius_mode == "neighbor" and "radii" not in freeze:
            d = _knn_mean_dist(scene.points.data[indices].detach().float().cpu().numpy())
            scale_r = torch.from_numpy(np.clip(d / knn_d0, 0.5, 4.0)).to(device=device, dtype=torch.float32)
            scene.radii.data[indices] = inverse_softplus_beta100(base_radii.to(device) * scale_r).to(scene.radii.dtype)
        if passive is not None:
            with torch.no_grad():
                scene.points.data[passive] = (passive_pos0 + (scene.points.data[indices].float()[passive_nn] - sel_pos0[passive_nn])).to(scene.points.dtype)
        for group in kinematic_groups:
            advance_kinematic_group(
                scene, group, (frame + 1) * time_params["frame_dt"],
                rotation_matrices, scale_factor, original_mean_pos,
                update_texel_axis="texel_axis" not in freeze,
            )
        scene.rebuild_adjacency()

        rgb = render_rgb(scene, camera, background)
        img = rgb_to_uint8_image(rgb)
        if tracked_idx is not None and frame < indicator_frames:
            # scene.points.data[indices] was just updated above to this frame's deformed world
            # positions
            current_point_world = scene.points.data[indices][tracked_idx]
            draw_force_indicator(img, camera, current_point_world, direction_world)
        save_uint8_rgb_png(img, str(output_path / f"frame_{frame:04d}.png"))
        if save_positions:
            (output_path / "positions").mkdir(exist_ok=True)
            torch.save(scene.points.data[indices].detach().to(torch.float16).cpu(),
                       output_path / "positions" / f"positions_{frame:04d}.pt")
            torch.save({"quaternions": scene.quaternions.data[indices].detach().float().cpu(),
                        "radii_raw": scene.radii.data[indices].detach().float().cpu()},
                       output_path / "positions" / f"state_{frame:04d}.pt")

    if compile_video:
        # fps follows frame_dt so playback speed matches simulated time
        height, width = rgb.shape[0], rgb.shape[1]
        # libx264 + yuv420p needs even width/height
        even_width, even_height = width - (width % 2), height - (height % 2)
        fps = int(1.0 / time_params["frame_dt"])
        video_path = output_path / "output.mp4"
        subprocess.run(
            [
                ffmpeg_exe(),
                "-framerate", str(fps),
                "-i", str(output_path / "frame_%04d.png"),
                "-c:v", "libx264",
                "-vf", f"crop={even_width}:{even_height}:0:0",
                "-y",
                "-pix_fmt", "yuv420p",
                str(video_path),
            ],
            check=True,
        )
        print(f"Compiled {video_path}")


def main():
    args = parse_args()
    run(
        checkpoint_config=args.checkpoint_config,
        sim_config=args.sim_config,
        output_dir=args.output_dir,
        device=args.device,
        split=args.split,
        camera_index=args.camera_index,
        num_frames=args.num_frames,
        drop_height=args.drop_height,
        compile_video=args.compile_video,
        selection=args.selection,
        recenter_selection=args.recenter_selection,
        material_field=args.material_field,
        fill_uncovered_nearest=args.fill_uncovered_nearest,
        show_force_indicator=args.show_force_indicator,
        force_indicator_frames=args.force_indicator_frames,
        save_positions=args.save_positions,
        freeze=tuple(args.freeze),
        radius_mode=args.radius_mode,
        appearance_source=args.appearance_source,
        displace=args.displace,
        hide=args.hide,
        passive_follow=args.passive_follow,
        background=args.background,
        cull_density=args.cull_density,
    )


if __name__ == "__main__":
    main()
