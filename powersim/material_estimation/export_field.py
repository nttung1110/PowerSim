"""Write a triplane material field as the per-primitive file the simulator reads (--material-field).

    python -m powersim.material_estimation.export_field --checkpoint-config <ckpt>/config.yaml \
        --selection mask.pt --sim-config config/<scene>.json --field-checkpoint stage2.pt --output field.pt

--random writes a random spatially correlated E field instead."""

import argparse
import json
import math
from pathlib import Path

import torch

from powersim.core import transform
from powersim.core.checkpoint import load_powerfoam_checkpoint
from powersim.material_estimation.fields.material_field import TriplaneMaterialField


def sim_space_positions(points_world: torch.Tensor, scale: float) -> torch.Tensor:
    """The forward simulator's MPM-space coordinates of `points_world` (bbox-based)"""
    transformed, _scale_factor, _mean = transform.transform2origin(points_world, scale)
    return transform.shift2center111(transformed)


def load_material_field(state_or_path, device, resolution=24, feat_dim=32, residual_scale=1.0, log_space=True):
    """Rebuild a TriplaneMaterialField from a saved state dict (aabb / init_E are buffers in it)"""
    state = torch.load(state_or_path, map_location=device) if isinstance(state_or_path, (str, Path)) else state_or_path
    field = TriplaneMaterialField(
        torch.zeros(2, 3, device=device), init_E=1.0, resolution=resolution, feat_dim=feat_dim,
        residual_scale=residual_scale, log_space=log_space, device=device,
    )
    field.load_state_dict(state)
    field.eval()
    return field


def save_material_file(path, indices, E, nu, density, source) -> None:
    n = indices.shape[0]
    payload = {
        "indices": indices.cpu(),
        "E": E.detach().cpu().float(),
        "nu": torch.full((n,), float(nu)),
        "density": torch.full((n,), float(density)),
        "covered": torch.ones(n, dtype=torch.bool),
        "source": source,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(f"[export_field] {path}: n={n} E min={E.min().item():.3e} median={E.median().item():.3e} "
          f"mean={E.mean().item():.3e} max={E.max().item():.3e}", flush=True)


def export_from_field(field, points_world, indices, sim_config: dict, out_path, source="TriplaneMaterialField"):
    """Evaluate `field` at the selected primitives' sim-space positions and write the material file."""
    x_sim = sim_space_positions(points_world, sim_config["scale"])
    inside = ((x_sim >= field.aabb_min) & (x_sim <= field.aabb_max)).all(dim=1)
    print(f"[export_field] {int(inside.sum())}/{x_sim.shape[0]} selected primitives inside the field's "
          f"aabb (outside ones take the boundary value)", flush=True)
    with torch.no_grad():
        E = field(x_sim)
    save_material_file(out_path, indices, E, sim_config["nu"], sim_config["density"], source)
    return E


def smooth_random_field(positions, correlation_length, n_modes, gen):
    """Isotropic Gaussian random field via random Fourier features; returns (n,) values."""
    device = positions.device
    dirs = torch.randn(n_modes, 3, generator=gen)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    base_freq = 2.0 * math.pi / correlation_length
    freqs = base_freq * torch.exp(0.4 * torch.randn(n_modes, generator=gen))
    k = (dirs * freqs[:, None]).to(device)
    phases = (2.0 * math.pi * torch.rand(n_modes, generator=gen)).to(device)
    amps = torch.randn(n_modes, generator=gen).to(device)
    return (amps[None, :] * torch.cos(positions @ k.T + phases[None, :])).sum(dim=1) / math.sqrt(n_modes)


def rank_to_log_uniform(values, lo, hi):
    """Monotone remap of `values` to a log-uniform distribution over [lo, hi] by rank."""
    n = values.shape[0]
    ranks = torch.empty(n, device=values.device, dtype=torch.float64)
    ranks[values.argsort()] = torch.arange(n, device=values.device, dtype=torch.float64)
    u = ranks / max(n - 1, 1)
    return torch.pow(10.0, math.log10(lo) + u * (math.log10(hi) - math.log10(lo))).float()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-config", required=True)
    ap.add_argument("--selection", required=True, help="boolean mask over the checkpoint's primitives")
    ap.add_argument("--sim-config", required=True, help="sim config JSON (scale, nu, density)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--field-checkpoint", default=None, help="TriplaneMaterialField state dict (stage-2 .pt)")
    ap.add_argument("--field-resolution", type=int, default=24)
    ap.add_argument("--field-feat-dim", type=int, default=32)
    ap.add_argument("--random", action="store_true", help="write a random E field instead")
    ap.add_argument("--E-lo", type=float, default=1e4)
    ap.add_argument("--E-hi", type=float, default=1e6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--correlation-length", type=float, default=0.25, help="fraction of the selection's max bbox side")
    ap.add_argument("--n-modes", type=int, default=64)
    ap.add_argument("--iid", action="store_true", help="independent per-primitive draws instead of a smooth field")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    with open(a.sim_config) as f:
        cfg = json.load(f)
    scene = load_powerfoam_checkpoint(a.checkpoint_config, device=a.device, split="train").scene
    mask = torch.load(a.selection, map_location=a.device).to(torch.bool)
    indices = mask.nonzero(as_tuple=True)[0]
    points = scene.points.data[indices].detach()
    print(f"[export_field] selection: {indices.shape[0]}/{scene.points.shape[0]} primitives", flush=True)

    if a.random:
        gen = torch.Generator().manual_seed(a.seed)
        if a.iid:
            raw = torch.rand(points.shape[0], generator=gen).to(a.device)
            kind = "iid"
        else:
            extent = float((points.max(0)[0] - points.min(0)[0]).max())
            raw = smooth_random_field(points, a.correlation_length * extent, a.n_modes, gen)
            kind = f"smooth(corr_len={a.correlation_length}*bbox, n_modes={a.n_modes})"
        E = rank_to_log_uniform(raw, a.E_lo, a.E_hi)
        save_material_file(a.output, indices, E, cfg["nu"], cfg["density"],
                           f"random {kind}, log-uniform [{a.E_lo:g},{a.E_hi:g}], seed={a.seed}")
        return
    if a.field_checkpoint is None:
        ap.error("--field-checkpoint is required unless --random")
    field = load_material_field(a.field_checkpoint, a.device, a.field_resolution, a.field_feat_dim)
    print(f"[export_field] field aabb (sim space): min={field.aabb_min.tolist()} max={field.aabb_max.tolist()}", flush=True)
    export_from_field(field, points, indices, cfg, a.output, source=f"TriplaneMaterialField {a.field_checkpoint}")


if __name__ == "__main__":
    main()
