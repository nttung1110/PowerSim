"""Recover a Young's modulus field from one video of an object deforming, then resimulate.

Stage 1 fits the initial velocity on the first frames with E frozen; stage 2 freezes the velocity
and fits a triplane E field on the whole video. Both differentiate through MPM, the primitive
update and the rasterizer.

    python -m powersim.material_estimation.run_video --checkpoint-config <ckpt>/config.yaml \
        --sim-config config/<scene>.json --selection mask.pt --video poke.mp4 --output-dir <dir>"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from powersim.core import transform
from powersim.core.checkpoint import load_powerfoam_checkpoint
from powersim.core.frame_renderer import render_rgb, rgb_to_uint8_image, save_uint8_rgb_png
from powersim.foam_evolve.viz.force_indicator import (
    compute_impulse_indicator_world_space, draw_force_indicator, find_tracked_particle_index,
)
from powersim.foam_evolve.viz.material_field_viz import render_material_field_triptych
from powersim.material_estimation.export_field import export_from_field
from powersim.material_estimation.fields.material_field import TriplaneMaterialField
from powersim.material_estimation.fields.velocity_field import TriplaneVelocityField
from powersim.material_estimation.mpm.checkpoint_to_mpm import build_mpm_state_from_points
from powersim.material_estimation.optimize import optimize_field_E, optimize_velocity_only, simulate_window
from powersim.material_estimation.velocity_viz import plot_velocity_field
from powersim.material_estimation.video import compile_video, hstack_videos, load_video_frames


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-config", required=True)
    ap.add_argument("--sim-config", required=True, help="scene JSON: material, grid, time stepping, floor collider, impulse")
    ap.add_argument("--selection", required=True, help="boolean mask (.pt) of the object's primitives")
    ap.add_argument("--video", required=True, help="the observed video")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--fill-cache", default=None, help="interior-fill cache (.pt); default <output-dir>/interior_fill_cache.pt, computed if missing")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--split", default="train")
    ap.add_argument("--camera-index", type=int, default=None, help="camera the video was taken from (default: the sim config's default_camera_index)")
    ap.add_argument("--max-frames", type=int, default=25, help="frames of the video to use")
    # paper / PhysDreamer defaults
    ap.add_argument("--stage1-frames", type=int, default=3)
    ap.add_argument("--stage1-iters", type=int, default=30)
    ap.add_argument("--stage1-lr", type=float, default=1e-2)
    ap.add_argument("--stage2-iters", type=int, default=20)
    ap.add_argument("--stage2-lr", type=float, default=5e-3)
    ap.add_argument("--E-init", type=float, default=1e7, help="stage-2 initial Young's modulus (Pa)")
    ap.add_argument("--random-E-log10-range", type=float, nargs=2, default=(6.0, 7.5), help="stage-1 frozen random E, log10 range")
    ap.add_argument("--ssim-weight", type=float, default=0.2)
    ap.add_argument("--smoothness-weight", type=float, default=1e-4)
    ap.add_argument("--velocity-smoothness-weight", type=float, default=1e-4)
    ap.add_argument("--chunk-size", type=int, default=150, help="substeps per checkpointed autograd chunk")
    ap.add_argument("--ckpt-every", type=int, default=5)
    ap.add_argument("--field-resolution", type=int, default=24)
    ap.add_argument("--field-feat-dim", type=int, default=32)
    ap.add_argument("--velocity-output-scale", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def ensure_fill_cache(path: Path, checkpoint_config, selection, device):
    if path.exists():
        return
    print(f"[run_video] no interior-fill cache at {path}, computing (taichi) in a subprocess ...", flush=True)
    subprocess.run(
        [sys.executable, "-m", "powersim.material_estimation.filling.precompute_interior_fill",
         "--checkpoint-config", str(checkpoint_config), "--selection", str(selection),
         "--output", str(path), "--device", device],
        check=True,
    )


def main():
    a = parse_args()
    device = a.device
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(a.sim_config) as f:
        cfg = json.load(f)
    camera_index = a.camera_index if a.camera_index is not None else int(cfg.get("default_camera_index", 0))
    print(f"[run_video] camera {camera_index}, max_frames {a.max_frames}, stage1 {a.stage1_frames} frames x "
          f"{a.stage1_iters} it @ {a.stage1_lr}, stage2 {a.stage2_iters} it @ {a.stage2_lr}, E_init {a.E_init:.2e}", flush=True)

    checkpoint = load_powerfoam_checkpoint(a.checkpoint_config, device=device, split=a.split)
    scene = checkpoint.scene
    camera = checkpoint.data_handler.cameras[camera_index]
    print(f"[run_video] camera {camera_index}: {camera.width}x{camera.height}", flush=True)

    # ---- the video, resampled to the simulation's frame rate and the camera's size ----
    fps = int(round(1.0 / cfg["frame_dt"]))
    gt_frames = load_video_frames(a.video, target_hw=(camera.height, camera.width), fps=fps, device=device, max_frames=a.max_frames)
    gt_dir = out / "gt_video"
    gt_dir.mkdir(exist_ok=True)
    for i, frame in enumerate(gt_frames):
        save_uint8_rgb_png(rgb_to_uint8_image(frame), str(gt_dir / f"frame_{i:04d}.png"))
    gt_video = out / "gt_video.mp4"
    compile_video(gt_dir, fps, gt_video)
    num_windows = len(gt_frames)
    print(f"[run_video] {num_windows} frames at {fps} fps; stage 1 fits the first {a.stage1_frames}", flush=True)

    # ---- object primitives + interior filling ----
    selection_mask = torch.load(a.selection, map_location=device).to(torch.bool)
    indices = selection_mask.nonzero(as_tuple=True)[0]
    print(f"[run_video] selection: {indices.shape[0]}/{scene.points.shape[0]} primitives", flush=True)
    fill_cache_path = Path(a.fill_cache) if a.fill_cache else out / "interior_fill_cache.pt"
    ensure_fill_cache(fill_cache_path, a.checkpoint_config, a.selection, device)
    fill = torch.load(fill_cache_path, map_location=device)
    points, radii, quaternions = fill["points"], fill["radii"], fill["quaternions"]
    n_orig, points_filled = fill["n_orig"], fill["points_filled"]
    print(f"[run_video] after interior fill: {points_filled.shape[0]} particles ({points_filled.shape[0] - n_orig} interior)", flush=True)

    # ---- MPM state ----
    material_params = {k: cfg[k] for k in ("material", "n_grid", "grid_lim", "density", "g", "rpic_damping")}
    base_state, base_model, x0, _F0, n, transform_info = build_mpm_state_from_points(points_filled, material_params, device, scale=cfg["scale"])
    scale_factor, original_mean_pos = transform_info
    bcs = {bc["type"]: bc for bc in cfg["boundary_conditions"]}
    floor_bc = bcs["surface_collider"]
    floor_point, floor_normal = tuple(floor_bc["point"]), tuple(floor_bc["normal"])
    dt = cfg["substep_dt"]
    substeps_per_window = int(round(cfg["frame_dt"] / cfg["substep_dt"]))
    nu_fixed = torch.full((n,), float(cfg["nu"]), device=device)
    damping_scale = cfg["grid_v_damping_scale"]

    # the scripted impulse is used only by the final resimulation (training learns v0 instead)
    impulse_force, impulse_substeps = None, 0
    if "particle_impulse" in bcs:
        ib = bcs["particle_impulse"]
        ipoint = torch.tensor(ib["point"], device=device, dtype=x0.dtype)
        isize = torch.tensor(ib["size"], device=device, dtype=x0.dtype)
        imask = ((x0 - ipoint).abs() < isize).all(dim=-1)
        impulse_force = torch.zeros_like(x0)
        impulse_force[imask] = torch.tensor(ib["force"], device=device, dtype=x0.dtype)
        impulse_substeps = int(ib.get("num_dt", 1))
        print(f"[run_video] scripted impulse hits {int(imask.sum())}/{n} particles (resimulation only)", flush=True)

    backdrop = rgb_to_uint8_image(render_rgb(scene, camera))
    nu_panel = nu_fixed[:n_orig]
    density_panel = torch.full((n_orig,), float(cfg["density"]), device=device)
    x0_min, x0_max = x0.min(dim=0).values, x0.max(dim=0).values
    margin = 0.1 * (x0_max - x0_min).max()
    aabb = torch.stack([x0_min - margin, x0_max + margin], dim=0)

    # ================= Stage 1: initial velocity, E frozen at random values =================
    torch.manual_seed(a.seed)
    E_fixed = torch.pow(10.0, torch.empty(n, device=device).uniform_(*a.random_E_log10_range))
    velocity_field = TriplaneVelocityField(aabb, resolution=a.field_resolution, feat_dim=a.field_feat_dim,
                                           output_scale=a.velocity_output_scale, device=device)
    s1 = out / "stage1"
    s1.mkdir(exist_ok=True)

    def on_iteration_stage1(it, vfield):
        if it % a.ckpt_every == 0 or it == a.stage1_iters - 1:
            with torch.no_grad():
                v0_vals = vfield(x0[:n_orig])
            plot_velocity_field(backdrop, camera, points, v0_vals, str(s1 / f"velocity_{it:04d}.png"), title_prefix=f"stage 1 iter {it}: ")
            torch.save(vfield.state_dict(), str(s1 / f"velocity_{it:04d}.pt"))

    print(f"\n[run_video] --- Stage 1: velocity field, {a.stage1_frames} frames, E frozen ---", flush=True)
    velocity_field, history1 = optimize_velocity_only(
        scene, camera, base_state, base_model, x0, n_orig, quaternions, radii, transform_info,
        gt_frames[:a.stage1_frames], nu_fixed, E_fixed, substeps_per_window, dt, device,
        velocity_field=velocity_field, n_iters=a.stage1_iters, lr=a.stage1_lr, ssim_weight=a.ssim_weight,
        velocity_smoothness_weight=a.velocity_smoothness_weight, floor_point=floor_point, floor_normal=floor_normal,
        checkpoint_chunk_size=a.chunk_size, damping_scale=damping_scale, on_iteration=on_iteration_stage1,
        render_indices=indices,
    )
    with open(out / "loss_curve_stage1.txt", "w") as f:
        for it, total_loss, window_count in history1:
            f.write(f"{it}\t{total_loss:.6e}\t{window_count}\n")
    torch.save(velocity_field.state_dict(), out / "velocity_field.pt")
    with torch.no_grad():
        v0_recovered = velocity_field(x0[:n_orig])
    speeds = v0_recovered.norm(dim=-1)
    print(f"[run_video] stage 1 done: |v0| min {speeds.min():.3e} max {speeds.max():.3e} mean {speeds.mean():.3e}", flush=True)
    plot_velocity_field(backdrop, camera, points, v0_recovered, str(out / "final_velocity.png"), title_prefix="recovered: ")

    # ================= Stage 2: material field, velocity frozen, full video =================
    for p in velocity_field.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        v0_frozen = velocity_field(x0)
    torch.manual_seed(a.seed + 1)
    material_field = TriplaneMaterialField(aabb, init_E=a.E_init, resolution=a.field_resolution, feat_dim=a.field_feat_dim,
                                           residual_scale=1.0, log_space=True, device=device)
    with torch.no_grad():
        for p in material_field.decoder.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    s2 = out / "stage2"
    s2.mkdir(exist_ok=True)

    def on_iteration_stage2(it, field):
        if it % a.ckpt_every == 0 or it == a.stage2_iters - 1:
            with torch.no_grad():
                E_vals = field(x0[:n_orig])
            render_material_field_triptych(backdrop, camera, points, E_vals, nu_panel, density_panel,
                                           output_path=str(s2 / f"iter_{it:04d}.png"), title_prefix=f"stage 2 iter {it}: ")
            torch.save(field.state_dict(), str(s2 / f"iter_{it:04d}.pt"))

    print(f"\n[run_video] --- Stage 2: Young's modulus field, {num_windows} frames, velocity frozen ---", flush=True)
    material_field, history2 = optimize_field_E(
        scene, camera, base_state, base_model, x0, v0_frozen, n_orig, quaternions, radii, transform_info,
        gt_frames, nu_fixed, substeps_per_window, dt, device,
        material_field=material_field, n_iters=a.stage2_iters, lr=a.stage2_lr, ssim_weight=a.ssim_weight,
        smoothness_weight=a.smoothness_weight, floor_point=floor_point, floor_normal=floor_normal,
        checkpoint_chunk_size=a.chunk_size, damping_scale=damping_scale, window_schedule=None,
        on_iteration=on_iteration_stage2, render_indices=indices,
    )
    with open(out / "loss_curve_stage2.txt", "w") as f:
        for it, total_loss, window_count in history2:
            f.write(f"{it}\t{total_loss:.6e}\t{window_count}\n")
    torch.save(material_field.state_dict(), out / "triplane_fields.pt")
    with torch.no_grad():
        E_recovered = material_field(x0[:n_orig])
    print(f"[run_video] stage 2 done: E min {E_recovered.min():.3e} max {E_recovered.max():.3e} mean {E_recovered.mean():.3e}", flush=True)
    render_material_field_triptych(backdrop, camera, points, E_recovered, nu_panel, density_panel,
                                   output_path=str(out / "final_mat.png"), title_prefix="recovered: ")
    material_field.eval()
    export_from_field(material_field, scene.points.data[indices].detach(), indices, cfg, out / "material_field.pt",
                      source=f"run_video stage 2 ({a.stage2_iters} it), {a.video}")

    # ================= Resimulations =================
    def resimulate(dir_name, v_init, use_impulse):
        d = out / dir_name
        d.mkdir(exist_ok=True)
        indicator = compute_impulse_indicator_world_space(cfg["boundary_conditions"], [], scale_factor, original_mean_pos, device,
                                                          step_per_frame=substeps_per_window, substep_dt=dt) if use_impulse else None
        tracked_idx, direction_world, indicator_frames = None, None, 0
        if indicator is not None:
            _, direction_world, point_sim, indicator_frames = indicator
            tracked_idx = find_tracked_particle_index(x0[:n_orig], point_sim)
        x, v = x0, v_init
        F_trial = torch.eye(3, device=device, dtype=torch.float32).expand(n, 3, 3).contiguous()
        C = torch.zeros_like(F_trial)
        with torch.no_grad():
            E_tensor = material_field(x0)
            for w in range(num_windows):
                rgb, x, v, F_trial, C = simulate_window(
                    scene, camera, base_state, base_model, x, v, F_trial, C, n_orig, quaternions, radii, transform_info,
                    E_tensor, nu_fixed, substeps_per_window, dt, device, floor_point, floor_normal, a.chunk_size, damping_scale,
                    None, None, impulse_force if (use_impulse and w == 0) else None, impulse_substeps if (use_impulse and w == 0) else 0,
                    indices,
                )
                img = rgb_to_uint8_image(rgb)
                if tracked_idx is not None and w < indicator_frames:
                    x_world = transform.undotransform2origin(transform.undoshift2center111(x[:n_orig]), scale_factor, original_mean_pos)
                    draw_force_indicator(img, camera, x_world[tracked_idx], direction_world)
                save_uint8_rgb_png(img, str(d / f"frame_{w:04d}.png"))
        compile_video(d, fps, d / "video.mp4")
        return d / "video.mp4"

    print("\n[run_video] --- resimulating with the recovered E field + the scripted impulse ---", flush=True)
    rec_video = resimulate("resim_impulse", torch.zeros_like(x0), use_impulse=impulse_force is not None)
    hstack_videos(rec_video, gt_video, out / "compare.mp4")
    print("[run_video] --- resimulating with the recovered E field + the recovered initial velocity ---", flush=True)
    resimulate("resim_learned_velocity", v0_frozen, use_impulse=False)
    print(f"\n[run_video] all results under {out}", flush=True)


if __name__ == "__main__":
    main()
