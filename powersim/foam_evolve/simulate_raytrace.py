"""Simulate with MPM and ray trace every frame, optionally with an inserted mirror or chrome ball.

    python -m powersim.foam_evolve.simulate_raytrace --checkpoint-config <ckpt>/config.yaml \
        --sim-config config/<scene>.json --selection mask.pt --transform-json config/mirrors/<m>.json \
        --camera-json config/cameras/<c>.json --output-dir <dir> --compile-video

The power-diagram adjacency is rebuilt exactly each frame (third_party/geogram_psm), because
primary rays walk the diagram cell by cell."""

import argparse
import json
import subprocess
from pathlib import Path

import torch
from tqdm import tqdm

from powersim.core import transform
from powersim.core.checkpoint import load_powerfoam_checkpoint
from powersim.core.frame_renderer import rgb_to_uint8_image, save_uint8_rgb_png
from powersim.core.primitive_state import apply_deformation_to_primitives, compute_grid_occupancy_volume, softplus_beta100
from powersim.core.thirdparty_paths import ensure_physgaussian_on_path, load_physgaussian_decode_param
from powersim.core.video import ffmpeg_exe
from powersim.foam_evolve.mpm.solver import build_mpm_solver, step_frame
from powersim.foam_evolve.raytrace.camera import make_orbit_camera
from powersim.foam_evolve.raytrace.material import RayTraceMaterial
from powersim.foam_evolve.raytrace.mirror import ChromeBall, MirrorPlane
from powersim.foam_evolve.raytrace.renderer import RayTraceRenderer
from powersim.foam_evolve.raytrace.scene_prep import build_exact_power_adjacency_geogram, geogram_available
from powersim.foam_evolve.simulate import build_per_particle_material


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-config", required=True)
    p.add_argument("--sim-config", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--selection", default=None, help="boolean mask of the simulated primitives (default: all)")
    p.add_argument("--recenter-selection", default=None, help="mask whose bbox defines the MPM frame")
    p.add_argument("--material-field", default=None, help="per-primitive E / nu / density file")
    p.add_argument("--material-config", default=None, help="JSON {reflectivity, ior, refraction_enabled, background_color} for the whole scene (default: no secondary rays)")
    p.add_argument("--camera-index", type=int, default=None)
    p.add_argument("--camera-json", default=None, help="look-at camera JSON (overrides --camera-index)")
    p.add_argument("--transform-json", default=None, help="inserted mirror / ball JSON")
    p.add_argument("--reflectivity", type=float, default=1.0, help="inserted object's reflectivity")
    p.add_argument("--ior", type=float, default=1.10, help="inserted object's refractive index (with --refraction)")
    p.add_argument("--refraction", action="store_true")
    p.add_argument("--thickness", type=float, default=0.05, help="mirror slab depth")
    p.add_argument("--num-frames", type=int, default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--compile-video", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    device = a.device
    if not geogram_available():
        raise RuntimeError("the exact-adjacency shim is not built: run `bash third_party/geogram_psm/build.sh`")
    checkpoint = load_powerfoam_checkpoint(a.checkpoint_config, device=device, split=a.split)
    scene = checkpoint.scene

    ensure_physgaussian_on_path()
    decode_param = load_physgaussian_decode_param()
    material_params, bc_params, time_params, preprocessing_params, camera_params = decode_param.decode_param_json(a.sim_config)
    if a.num_frames is not None:
        time_params["frame_num"] = a.num_frames

    camera_index = a.camera_index if a.camera_index is not None else camera_params["default_camera_index"]
    camera = checkpoint.data_handler.cameras[camera_index]
    if a.camera_json is not None:
        cj = json.load(open(a.camera_json))
        camera = make_orbit_camera(
            center=torch.tensor(cj["center"], dtype=torch.float32), radius=float(cj["radius"]),
            azimuth_deg=float(cj["azimuth_deg"]), elevation_deg=float(cj["elevation_deg"]), fov_deg=float(cj["fov_deg"]),
            width=int(cj.get("width", camera.width)), height=int(cj.get("height", camera.height)), device=device,
            world_up=torch.tensor(cj["world_up"], dtype=torch.float32),
        )
        print(f"camera from {a.camera_json}: {camera.width}x{camera.height}")

    if a.selection is not None:
        mask = torch.load(a.selection, map_location="cpu").to(torch.bool)
        if mask.shape[0] != scene.points.shape[0]:
            raise ValueError(f"--selection has {mask.shape[0]} entries, the checkpoint {scene.points.shape[0]} primitives")
        indices = mask.nonzero(as_tuple=True)[0].to(scene.points.device)
        print(f"simulating {indices.shape[0]}/{scene.points.shape[0]} primitives (from --selection)")
    else:
        indices = torch.arange(scene.points.shape[0], device=scene.points.device)

    base_quaternions = scene.quaternions.data[indices].clone()
    base_radii = softplus_beta100(scene.radii.data[indices]).clone()
    base_texel_sv_axis = scene.texel_sv_axis.data[indices].clone()

    sim_scale = preprocessing_params["scale"]
    if a.recenter_selection is not None:
        rmask = torch.load(a.recenter_selection, map_location="cpu").to(torch.bool)
        ridx = rmask.nonzero(as_tuple=True)[0].to(scene.points.device)
        _, scale_factor, original_mean_pos = transform.transform2origin(scene.points.data[ridx], sim_scale)
        transformed_pos = (scene.points.data[indices] - original_mean_pos) * scale_factor
        print(f"MPM frame recentred on --recenter-selection ({ridx.shape[0]} primitives)")
    else:
        transformed_pos, scale_factor, original_mean_pos = transform.transform2origin(scene.points.data[indices], sim_scale)
    mpm_init_pos = transform.shift2center111(transformed_pos)
    grid_dx = material_params["grid_lim"] / material_params["n_grid"]
    mpm_init_vol = compute_grid_occupancy_volume(mpm_init_pos, material_params["n_grid"], grid_dx, uniform=(material_params["material"] == "sand"))

    per_particle_material = None
    if a.material_field is not None:
        E_pp, nu_pp, density_pp, covered = build_per_particle_material(
            scene.points.shape[0], indices, a.material_field, material_params, device, positions=scene.points.data[indices])
        per_particle_material = {"E": E_pp, "nu": nu_pp, "density": density_pp}
        print(f"material field: {int(covered.sum())}/{indices.shape[0]} simulated primitives covered, rest use the config's scalars")

    mpm_solver = build_mpm_solver(mpm_init_pos, mpm_init_vol, material_params, bc_params, time_params, device, per_particle_material)
    step_per_frame = int(time_params["frame_dt"] / time_params["substep_dt"])
    substep_dt = time_params["substep_dt"]

    material = RayTraceMaterial.load(a.material_config) if a.material_config else RayTraceMaterial()
    renderer = RayTraceRenderer(scene)
    mirror, ball = None, None
    if a.transform_json is not None:
        t = json.load(open(a.transform_json))
        kind = t.get("type", "mirror")
        if kind == "ball":
            ball = ChromeBall.from_transform(t["translate"], t["radius"], device, reflectivity=a.reflectivity, ior=a.ior, refraction_enabled=a.refraction)
            print(f"chrome ball at {ball.center.cpu().numpy().round(3)} radius {ball.radius:.3f}")
        elif kind == "mirror":
            mirror = MirrorPlane.from_transform(
                t["translate"], t["rotate_deg"], t["scale"], device, reflectivity=a.reflectivity, ior=a.ior,
                refraction_enabled=a.refraction, thickness=a.thickness,
                border_width=t.get("border_width", 0.0), border_color=t.get("border_color", (0.12, 0.08, 0.05)))
            print(f"mirror at {mirror.center.cpu().numpy().round(3)}, reflectivity {mirror.reflectivity}, border {mirror.border_width}")
        else:
            raise ValueError(f"--transform-json type must be 'mirror' or 'ball', got {kind!r}")

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for frame in tqdm(range(time_params["frame_num"]), desc="Simulating + ray tracing"):
        step_frame(mpm_solver, step_per_frame, substep_dt, frame=frame, device=device)
        x = mpm_solver.export_particle_x_to_torch()
        F = mpm_solver.export_particle_F_to_torch().reshape(-1, 3, 3)
        apply_deformation_to_primitives(scene, indices, F, base_quaternions, base_radii, centroid=None, base_texel_sv_axis=base_texel_sv_axis)
        scene.points.data[indices] = transform.undo_all_transforms(x, [], scale_factor, original_mean_pos)
        adjacency, adjacency_offsets = build_exact_power_adjacency_geogram(scene.points.detach(), scene.get_radii().detach())
        rgb = renderer.render(scene, camera, material=material, mirror=mirror, ball=ball, adjacency=adjacency, adjacency_offsets=adjacency_offsets)
        if not torch.isfinite(rgb).all():
            raise RuntimeError(f"frame {frame}: ray-traced image has NaN/Inf")
        save_uint8_rgb_png(rgb_to_uint8_image(rgb), str(out / f"frame_{frame:04d}.png"))

    if a.compile_video:
        fps = int(round(1.0 / time_params["frame_dt"]))
        subprocess.run([ffmpeg_exe(), "-framerate", str(fps), "-i", str(out / "frame_%04d.png"),
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-y", "-loglevel", "error",
                        "-pix_fmt", "yuv420p", str(out / "output.mp4")], check=True)
        print(f"Compiled {out / 'output.mp4'}")


if __name__ == "__main__":
    main()
