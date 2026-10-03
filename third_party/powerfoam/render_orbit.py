#!/usr/bin/env python3
"""
Render an orbit video from a trained Power Foam checkpoint.

Must be run from the powerfoam/ directory:
    cd /path/to/powerfoam
    python render_orbit.py --checkpoint output/job_000_raw_10k/model.pt

Key orbit parameters:
  --azimuth_start / --azimuth_end  horizontal sweep in degrees
  --elevation                      fixed camera height angle in degrees
  --radius                         camera distance from scene center
  --fov                            horizontal field of view in degrees

--radius, --fov, --azimuth_start/--azimuth_end, and the orbit's own up axis all default to
values DERIVED from the training cameras (config.yaml's data_path/scene, loaded the same way
view.py's own interactive viewer does) rather than fixed generic numbers:
  - up axis / radius / fov: matters for scenes whose coordinate frame isn't Z-up by convention
    (e.g. an independently-run COLMAP reconstruction, whose up axis can point anywhere) or
    whose absolute scale isn't ~1 (COLMAP's own reconstruction scale is arbitrary/unconstrained).
  - azimuth range: a real capture very often only covers a partial range of viewpoints (e.g.
    walking around the front of a desk against a wall, never behind it) -- unlike a synthetic
    scene rendered from every angle, sweeping a full 360 deg by default would orbit into
    unphotographed regions where the reconstruction was never constrained and looks broken.
Pass --radius/--fov/--azimuth_start/--azimuth_end explicitly to override any of these. Falls
back to fixed generic defaults (radius=1.3, fov=50, world-Z up, -35 deg to 325 deg) if the
training dataset isn't available on disk.
"""

import sys
import math
import argparse
import subprocess
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warp as wp
import yaml
from PIL import Image

# Must be run from powerfoam/ so imports resolve correctly.
POWERFOAM_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(POWERFOAM_DIR))

from configs import Params
from data_loader import DataHandler
from powerfoam.scene import PowerfoamScene
from powerfoam.camera import TorchCamera
from powerfoam.rasterize import Rasterizer, VisOptions
from powerfoam.raytrace import RayTracer
from powerfoam.color_fn import SphericalVoronoi


# ---------------------------------------------------------------------------
# Config / model loading
# ---------------------------------------------------------------------------

def load_args(config_yaml: Path) -> Params:
    with open(config_yaml) as f:
        cfg = yaml.safe_load(f)
    param_fields = {fld.name for fld in fields(Params)}
    return Params(**{k: v for k, v in cfg.items() if k in param_fields})


def load_model(checkpoint: Path, args: Params, device: torch.device) -> PowerfoamScene:
    """
    Build a PowerfoamScene entirely from a saved checkpoint.
    Avoids needing a DataHandler or the training dataset on disk.
    """
    ckpt = torch.load(str(checkpoint), map_location="cpu")

    model = PowerfoamScene(args)

    def _param(key):
        return nn.Parameter(ckpt[key].to(torch.float32).to(device))

    model.points        = _param("points")
    model.density       = _param("density")
    model.radii         = _param("radii")
    model.quaternions   = _param("quaternions")
    model.texel_sites   = _param("texel_sites")
    model.texel_sv_axis = _param("texel_sv_axis")
    model.texel_sv_rgb  = _param("texel_sv_rgb")
    model.texel_height  = _param("texel_height")

    model.adjacency         = ckpt["adjacency"].to(device)
    model.adjacency_offsets = ckpt["adjacency_offsets"].to(device)

    model.rasterizer = Rasterizer(args, device, model.attr_dtype)
    model.raytracer  = RayTracer(args, device, model.attr_dtype)
    model.sv         = SphericalVoronoi(args, device, model.attr_dtype)

    return model


# ---------------------------------------------------------------------------
# Deriving orbit defaults from the actual training cameras
# ---------------------------------------------------------------------------

def derive_azimuth_range(eyes: torch.Tensor, center: torch.Tensor, u1: torch.Tensor, u2: torch.Tensor):
    """Default azimuth sweep covering the range actually photographed by the training cameras,
    not a blind full 360 deg loop -- a real capture (as opposed to a synthetic scene rendered
    from every angle) very often only covers a partial range of viewpoints (e.g. walking around
    the front of a desk against a wall, never behind it). Orbiting into the uncovered gap
    inevitably passes through poorly- or un-reconstructed geometry (no photos ever constrained
    it), which looks broken -- not a bug in the orbit math itself, just outside what SfM could
    ever have reconstructed reliably.

    Standard circular-range technique: find the single largest gap between the (sorted, mod 360)
    training-camera azimuths and take the default sweep as everything OUTSIDE that gap, trimmed
    inward by a small margin so the sweep doesn't ride the very edge of coverage.
    """
    rel = eyes - center[None, :]
    az_rad = torch.atan2(rel @ u2, rel @ u1)
    azs = sorted((math.degrees(a) % 360.0) for a in az_rad.tolist())
    n = len(azs)
    gaps = [(azs[(i + 1) % n] - azs[i]) % 360.0 for i in range(n)]
    max_gap_idx = max(range(n), key=lambda i: gaps[i])
    max_gap = gaps[max_gap_idx]
    start = azs[(max_gap_idx + 1) % n]
    span = 360.0 - max_gap
    margin = 0.05 * span
    return start + margin, start + span - margin


def derive_orbit_defaults_from_training_data(args: Params, center: torch.Tensor):
    """
    Loads the training dataset (config.yaml's own data_path/scene) and derives:
      - world_up: the scene's real up direction (DataHandler.viewer_up, the same
        quantity view.py's own interactive viewer uses) -- NOT assumed to be Z.
        A from-scratch COLMAP reconstruction (e.g. alocasia_sfm) has an
        arbitrarily-oriented coordinate frame; there's no reason its up axis
        would coincide with world Z the way a Blender scene's would.
      - default_radius: median distance from the training cameras' own eye
        positions to `center` -- i.e. roughly how far the real photos were
        actually taken from the object.
      - default_fov: the first training camera's own horizontal FOV, recovered
        from its `right` vector magnitude via the exact inverse of
        make_orbit_camera's own focal-length formula.
      - (default_azimuth_start, default_azimuth_end): the azimuth range actually covered by
        the training cameras (see derive_azimuth_range) -- not a blind full-360 sweep.

    Returns None (with a printed reason) if the dataset isn't available on disk
    -- render_orbit.py can still run checkpoint-only, just falling back to
    fixed generic defaults in that case.
    """
    try:
        data_handler = DataHandler(args)
        data_handler.reload("train", downsample=args.downsample[-1])
    except Exception as e:
        print(f"Could not load training dataset to derive orbit defaults ({e}); "
              f"falling back to fixed generic defaults.")
        return None

    world_up = data_handler.viewer_up.to(torch.float32)
    u1, u2 = build_equatorial_basis(world_up)

    eyes = data_handler.c2ws[:, :3, 3].to(torch.float32)
    dists = torch.norm(eyes - center[None, :], dim=-1)
    default_radius = dists.median().item()

    cam0 = data_handler.cameras[0]
    right_scale = torch.norm(cam0.right).item()
    focal = (cam0.width / 2.0 - 0.5) / right_scale
    default_fov = math.degrees(2.0 * math.atan(0.5 * cam0.width / focal))

    az_start, az_end = derive_azimuth_range(eyes, center, u1, u2)

    print(f"Derived from training cameras: world_up={world_up.tolist()}, "
          f"radius={default_radius:.4f} (median cam-to-center distance), "
          f"fov={default_fov:.2f}° (from camera 0's own intrinsics), "
          f"azimuth range={az_start:.1f}°→{az_end:.1f}° (span of actual training-camera coverage)")
    return world_up, default_radius, default_fov, az_start, az_end


# ---------------------------------------------------------------------------
# Camera utilities
# ---------------------------------------------------------------------------

def build_equatorial_basis(world_up: torch.Tensor):
    """Orthonormal (u1, u2) perpendicular to world_up, spanning the orbit's "equatorial" plane
    -- shared by make_orbit_camera (to build the orbit path) and
    derive_orbit_defaults_from_training_data (to measure training cameras' azimuths in that
    same plane, so the two use an identical azimuth convention). Seeds with whichever world axis
    is least parallel to world_up, to avoid a degenerate cross product."""
    world_up = world_up.to(torch.float32).cpu()
    world_up = world_up / torch.norm(world_up)
    seed = torch.tensor([1., 0., 0.], dtype=torch.float32)
    if torch.dot(world_up, seed).abs() > 0.9:
        seed = torch.tensor([0., 1., 0.], dtype=torch.float32)
    u1 = torch.cross(seed, world_up, dim=-1)
    u1 = u1 / torch.norm(u1)
    u2 = torch.cross(world_up, u1, dim=-1)
    return u1, u2


def fov_cos_cutoff(fov_deg: float, width: int, height: int) -> float:
    """Matches SphericalVoronoi.compute_fov_cos_cutoff for a pinhole camera."""
    fov_rad = math.radians(fov_deg)
    focal   = 0.5 * width / math.tan(0.5 * fov_rad)
    cx, cy  = width / 2.0, height / 2.0
    x_max   = max(cx, width  - cx) / focal
    y_max   = max(cy, height - cy) / focal
    tan_th  = math.sqrt(x_max**2 + y_max**2) * 1.1
    return 1.0 / math.sqrt(1.0 + tan_th**2)


def make_orbit_camera(
    center: torch.Tensor,
    radius: float,
    azimuth_deg: float,
    elevation_deg: float,
    fov_deg: float,
    width: int,
    height: int,
    device: torch.device,
    world_up: torch.Tensor = None,
) -> TorchCamera:
    """
    Build a TorchCamera whose eye sits on a sphere of the given radius around
    `center`.  Follows the exact right/up scaling convention of BlenderDataset
    so the FOV is interpreted identically to training.

    world_up: the scene's real up direction (see derive_orbit_defaults_from_training_data).
    Defaults to world Z if not given -- only a safe assumption for scenes that are actually
    Z-up (e.g. this project's synthetic Blender scenes); a from-scratch COLMAP reconstruction's
    up axis can point anywhere, so passing the real one matters there. This is used for BOTH
    the orbit path itself (azimuth sweeps around world_up, not necessarily around Z) and each
    frame's camera right/up basis.
    """
    if world_up is None:
        world_up = torch.tensor([0., 0., 1.], dtype=torch.float32)
    world_up = world_up.to(torch.float32).cpu()
    world_up = world_up / torch.norm(world_up)
    u1, u2 = build_equatorial_basis(world_up)

    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)

    # Eye position: az sweeps around world_up in the (u1, u2) equatorial plane; el tilts
    # toward/away from world_up.
    equator_dir = math.cos(az) * u1 + math.sin(az) * u2
    eye = center + radius * (math.cos(el) * equator_dir + math.sin(el) * world_up)

    # Forward: eye → center
    fwd = center - eye
    fwd = fwd / torch.norm(fwd)

    # Right: forward × world_up  (fall back to u1 if looking nearly along world_up, i.e. near
    # the orbit's own poles)
    up_ref = world_up
    if torch.dot(fwd, up_ref).abs() > 0.99:
        up_ref = u1
    right_dir = torch.cross(fwd, up_ref, dim=-1)
    right_dir = right_dir / torch.norm(right_dir)

    # Camera up: right × forward
    up_dir = torch.cross(right_dir, fwd, dim=-1)
    up_dir = up_dir / torch.norm(up_dir)

    # Scale by FOV — mirrors the BlenderDataset formula exactly:
    #   focal = 0.5 * w / tan(fov/2)
    #   right_scale = (w/2 - 0.5) / focal
    fov_rad     = math.radians(fov_deg)
    focal       = 0.5 * width  / math.tan(0.5 * fov_rad)
    right_scale = (width  / 2.0 - 0.5) / focal
    up_scale    = (height / 2.0 - 0.5) / focal  # same focal → same angle per pixel

    return TorchCamera(
        eye   = eye.to(device),
        right = (right_dir * right_scale).to(device),
        up    = (up_dir    * up_scale   ).to(device),
        width = width,
        height= height,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_frame(
    model: PowerfoamScene,
    camera: TorchCamera,
    bg: list,
) -> np.ndarray:
    """Returns a uint8 H×W×3 numpy RGB array."""
    vis_opts = VisOptions()
    vis_opts.transmittance_threshold = 1e-3
    vis_opts.max_intersections       = 1024
    vis_opts.depth_quantile          = 0.5
    vis_opts.bkgd_color              = wp.vec3f(float(bg[0]), float(bg[1]), float(bg[2]))

    with torch.no_grad():
        color, _depth, _normal, _alpha, _ = model.forward_visualization(
            camera.to_device(model.device),
            vis_options=vis_opts,
        )

    rgb = color.clamp(0.0, 1.0).cpu().numpy()
    return (rgb * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Video compilation
# ---------------------------------------------------------------------------

def compile_video(frames_dir: Path, fps: int, out: Path) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(out),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print("ffmpeg stderr:", result.stderr.decode(errors="replace"))
        print("Frames left in:", frames_dir)
    else:
        print(f"Video saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Render a Power Foam checkpoint as an orbit video."
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model.pt  (config.yaml must be in the same dir)")
    parser.add_argument("--out", default=None,
                        help="Output .mp4 (default: <checkpoint_dir>/orbit.mp4)")
    parser.add_argument("--frames", type=int, default=120,
                        help="Number of frames  [120]")
    parser.add_argument("--azimuth_start", type=float, default=None,
                        help="Start azimuth in degrees "
                             "[default: derived from the training cameras' own coverage, if "
                             "the training dataset is available; else -35]")
    parser.add_argument("--azimuth_end", type=float, default=None,
                        help="End azimuth in degrees (exclusive) "
                             "[default: derived from the training cameras' own coverage, if "
                             "the training dataset is available; else 325 -> full loop]")
    parser.add_argument("--elevation", type=float, default=30.0,
                        help="Fixed elevation angle in degrees  [30]")
    parser.add_argument("--radius", type=float, default=None,
                        help="Camera distance from scene center "
                             "[default: derived from training cameras' own distance to the "
                             "scene center, if the training dataset is available; else 1.3]")
    parser.add_argument("--fov", type=float, default=None,
                        help="Horizontal FOV in degrees "
                             "[default: derived from the training cameras' own intrinsics, "
                             "if the training dataset is available; else 50]")
    parser.add_argument("--width",  type=int, default=768, help="Render width   [768]")
    parser.add_argument("--height", type=int, default=768, help="Render height  [768]")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS  [30]")
    parser.add_argument("--bg", nargs=3, type=float, default=[0.0, 0.0, 0.0],
                        metavar=("R", "G", "B"),
                        help="Background colour in [0,1]  [0 0 0]")
    parser.add_argument("--center", nargs=3, type=float, default=None,
                        metavar=("X", "Y", "Z"),
                        help="Scene centre override (default: density-weighted mean of points)")
    parser.add_argument("--device", default="cuda", help="PyTorch device  [cuda]")
    parser.add_argument("--keep_frames", action="store_true",
                        help="Keep individual PNG frames after video compilation")
    opt = parser.parse_args()

    checkpoint  = Path(opt.checkpoint)
    config_yaml = checkpoint.parent / "config.yaml"
    out_path    = Path(opt.out) if opt.out else checkpoint.parent / "orbit.mp4"
    device      = torch.device(opt.device)

    # ---- load ----------------------------------------------------------------
    print(f"Config  : {config_yaml}")
    args = load_args(config_yaml)

    wp.init()

    print(f"Checkpoint : {checkpoint}")
    model = load_model(checkpoint, args, device)
    model.eval()

    # ---- scene centre --------------------------------------------------------
    if opt.center is not None:
        center = torch.tensor(opt.center, dtype=torch.float32)
        print(f"Centre (user) : {center.tolist()}")
    else:
        with torch.no_grad():
            pts     = model.points.detach().cpu()
            weights = F.softplus(model.density.detach().cpu(), beta=100)
            weights = weights / weights.sum()
            center  = (pts * weights[:, None]).sum(dim=0)
        print(f"Centre (density-weighted mean) : {center.tolist()}")

    # ---- derive orbit defaults (world_up, radius, fov, azimuth range) from the training
    # cameras, if the training dataset is available -- see
    # derive_orbit_defaults_from_training_data's own docstring for why this matters (an
    # independently-run COLMAP reconstruction's up axis doesn't necessarily coincide with world
    # Z the way a Blender scene's does, and a real capture very often only covers a partial
    # range of viewpoints, unlike a synthetic scene rendered from every angle). Explicit
    # --radius/--fov/--azimuth_start/--azimuth_end on the command line always win over the
    # derived values.
    derived = derive_orbit_defaults_from_training_data(args, center)
    if derived is not None:
        world_up, derived_radius, derived_fov, derived_az_start, derived_az_end = derived
    else:
        world_up = None  # make_orbit_camera falls back to world Z
        derived_radius, derived_fov = 1.3, 50.0
        derived_az_start, derived_az_end = -35.0, 325.0

    radius = opt.radius if opt.radius is not None else derived_radius
    fov = opt.fov if opt.fov is not None else derived_fov
    azimuth_start = opt.azimuth_start if opt.azimuth_start is not None else derived_az_start
    azimuth_end = opt.azimuth_end if opt.azimuth_end is not None else derived_az_end

    # ---- SphericalVoronoi cutoff for this camera -----------------------------
    model.sv.fov_cos_cutoff = fov_cos_cutoff(fov, opt.width, opt.height)

    # ---- render frames -------------------------------------------------------
    frames_dir = out_path.parent / "_orbit_frames_tmp"
    frames_dir.mkdir(parents=True, exist_ok=True)

    azimuths = np.linspace(azimuth_start, azimuth_end, opt.frames, endpoint=False)
    print(f"Rendering {opt.frames} frames  "
          f"(az {azimuth_start:.1f}° → {azimuth_end:.1f}°, "
          f"el {opt.elevation}°, r {radius:.4f}, fov {fov:.2f}°) …")

    for i, az in enumerate(azimuths):
        cam = make_orbit_camera(
            center       = center,
            radius       = radius,
            azimuth_deg  = float(az),
            elevation_deg= opt.elevation,
            fov_deg      = fov,
            width        = opt.width,
            height       = opt.height,
            device       = device,
            world_up     = world_up,
        )
        frame = render_frame(model, cam, opt.bg)
        Image.fromarray(frame).save(frames_dir / f"{i:04d}.png")
        if (i + 1) % 10 == 0 or (i + 1) == opt.frames:
            print(f"  {i + 1:4d}/{opt.frames}")

    # ---- compile video -------------------------------------------------------
    compile_video(frames_dir, opt.fps, out_path)

    # ---- cleanup -------------------------------------------------------------
    if not opt.keep_frames:
        for f in frames_dir.glob("*.png"):
            f.unlink()
        try:
            frames_dir.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    main()
