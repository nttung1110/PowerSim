"""Flat buffers for the ray tracer and the exact power-diagram adjacency (geogram)."""

import os
import subprocess
import tempfile
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RaytraceBuffers:
    points: torch.Tensor  # (N, 3)
    radii: torch.Tensor  # (N,)
    density: torch.Tensor  # (N,)
    normals: torch.Tensor  # (N, 3)
    texel_sites: torch.Tensor  # (N, num_texel_sites, 3), world-space
    texel_height: torch.Tensor  # (N, num_texel_sites)
    att_sites: torch.Tensor
    att_values: torch.Tensor
    att_temps: torch.Tensor


def prepare_raytrace_buffers(scene) -> RaytraceBuffers:
    """Everything `PowerfoamScene.forward()` computes before handing off to a renderer."""
    normals = scene.get_normals()
    tangents, bitangent = scene.get_tangents()
    radii = scene.get_radii()
    offsets = scene.texel_sites * radii[:, None, None]
    offsets = (
        offsets[..., 0:1] * tangents[:, None, :] + offsets[..., 1:2] * bitangent[:, None, :]
    )
    texel_sites = scene.points[:, None, :] + offsets
    texel_height = scene.texel_height * radii[:, None]
    att_sites, att_values, att_temps = scene.get_att_sv()

    return RaytraceBuffers(
        points=scene.points,
        radii=radii,
        density=scene.get_density(),
        normals=normals,
        texel_sites=texel_sites,
        texel_height=texel_height,
        att_sites=att_sites,
        att_values=att_values,
        att_temps=att_temps,
    )


def compute_texel_rgb(scene, buffers: RaytraceBuffers, camera) -> torch.Tensor:
    """View-dependent texel color: must be recomputed per camera/frame, so it's kept out of
    `RaytraceBuffers` (which is otherwise safe to reuse across cameras for a static scene)"""
    texel_rgb = scene.sv.forward(
        buffers.texel_sites.view(-1, 3).detach(),
        camera,
        buffers.att_sites,
        buffers.att_values,
        buffers.att_temps,
    )
    return texel_rgb.view(buffers.points.shape[0], scene.args.num_texel_sites, 3)


def compute_adjacency_diff(
    points: torch.Tensor,
    radii: torch.Tensor,
    adjacency: torch.Tensor,
    adjacency_offsets: torch.Tensor,
) -> torch.Tensor:
    """Per-edge power-face difference, as in PowerFoam's benchmark.py."""
    num_adjs = adjacency_offsets.diff()
    pm = 0.5 * (points.norm(dim=-1) ** 2 - radii**2)
    self_points = points.repeat_interleave(num_adjs, dim=0)
    diff = points[adjacency] - self_points
    pm_diff = pm[adjacency] - pm.repeat_interleave(num_adjs, dim=0)
    adjacency_diff = torch.cat([diff, pm_diff[:, None]], dim=-1)
    return adjacency_diff.to(torch.float16)


def compute_start_point_idx(points: torch.Tensor, radii: torch.Tensor, camera) -> int:
    """Primary ray's seed primitive: nearest to the camera eye by power distance."""
    camera_eye = camera.eye.to(points.device)
    dists = torch.linalg.norm(points - camera_eye[None, :], dim=-1)
    return int(torch.argmin(dists**2 - radii**2))


# The exact power-diagram adjacency shim (third_party/geogram_psm/build.sh builds it).
_GEOGRAM_EXE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "third_party", "geogram_psm", "geo_power_adj",
)


def geogram_available() -> bool:
    """Whether the Geogram shim has been built."""
    return os.path.exists(_GEOGRAM_EXE) and os.access(_GEOGRAM_EXE, os.X_OK)


def build_exact_power_adjacency_geogram(points: torch.Tensor, radii: torch.Tensor, method: str = "PDEL"):
    """Exact power-diagram adjacency via Geogram's RegularWeightedDelaunay3d ("BPOW")"""
    if method not in ("PDEL", "BPOW"):
        raise ValueError(f"method must be PDEL or BPOW, got {method!r}")
    if not geogram_available():
        raise RuntimeError(f"geogram shim not built at {_GEOGRAM_EXE}; run `bash third_party/geogram_psm/build.sh`")
    pts = points.detach().cpu().numpy().astype(np.float64, copy=False)
    w = (radii.detach().cpu().numpy().astype(np.float64, copy=False)) ** 2
    n = pts.shape[0]

    with tempfile.TemporaryDirectory() as td:
        fin, fout = os.path.join(td, "in.bin"), os.path.join(td, "out.bin")
        with open(fin, "wb") as f:
            np.int64(n).tofile(f)
            np.ascontiguousarray(pts).tofile(f)
            np.ascontiguousarray(w).tofile(f)
        # PDEL (ParallelDelaunay3d) accepts dimension 4 and carries weighted_/heights_, so it
        # handles the weighted case
        r = subprocess.run([_GEOGRAM_EXE, fin, fout, method], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"geogram shim failed: {r.stderr[-500:]}")
        with open(fout, "rb") as f:
            n_out = int(np.fromfile(f, np.int64, 1)[0])
            n_edges = int(np.fromfile(f, np.int64, 1)[0])
            offsets = np.fromfile(f, np.int64, n_out + 1)
            adjacency = np.fromfile(f, np.int32, n_edges)

    if n_out != n:
        raise RuntimeError(f"geogram returned {n_out} rows for {n} points")

    # int32 for both, matching what the Warp kernels expect (an int64 adjacency raises "Could not
    # convert array interface with typestr='<i8' to Warp array with dtype=int32")
    adj_t = torch.from_numpy(adjacency.astype(np.int32)).to(points.device)
    off_t = torch.from_numpy(offsets.astype(np.int32)).to(points.device)
    return adj_t, off_t
