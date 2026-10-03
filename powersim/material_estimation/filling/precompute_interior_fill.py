"""Compute and cache the interior fill of a selection."""

import argparse

import torch

from powersim.core.checkpoint import load_powerfoam_checkpoint

from powersim.material_estimation.filling.interior_fill import fill_interior


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-config", required=True)
    parser.add_argument("--selection", required=True, help="Boolean/index .pt mask over the checkpoint's full primitive ordering")
    parser.add_argument("--output", required=True, help="Where to save the filled-primitive cache (.pt)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--grid-n", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=200_000)
    parser.add_argument(
        "--density-thres", type=float, default=None,
        help="Default: the selection's own median density",
    )
    args = parser.parse_args()

    checkpoint = load_powerfoam_checkpoint(args.checkpoint_config, device=args.device, split="train")
    scene = checkpoint.scene

    selection_mask = torch.load(args.selection, map_location=args.device)
    indices = selection_mask.nonzero(as_tuple=True)[0]
    print(f"selection: {indices.shape[0]}/{scene.points.shape[0]} primitives")

    points = scene.points.detach()[indices]
    radii = scene.get_radii().detach()[indices]
    density = scene.get_density().detach()[indices]
    quaternions = scene.quaternions.detach()[indices]
    n_orig = points.shape[0]

    density_thres = args.density_thres if args.density_thres is not None else torch.quantile(density, 0.5).item()
    points_filled, radii_filled, density_filled, quat_filled, is_interior = fill_interior(
        points, radii, density, quaternions,
        grid_n=args.grid_n, max_samples=args.max_samples, density_thres=density_thres,
    )
    print(f"after fill_interior: {points_filled.shape[0]} primitives "
          f"({points_filled.shape[0] - n_orig} interior), density_thres={density_thres:.4g}")

    torch.save(
        {
            "points": points.cpu(),
            "radii": radii.cpu(),
            "density": density.cpu(),
            "quaternions": quaternions.cpu(),
            "indices": indices.cpu(),
            "n_orig": n_orig,
            "density_thres": density_thres,
            "points_filled": points_filled.cpu(),
            "radii_filled": radii_filled.cpu(),
            "density_filled": density_filled.cpu(),
            "quaternions_filled": quat_filled.cpu(),
            "is_interior": is_interior.cpu(),
        },
        args.output,
    )
    print(f"saved interior-fill cache to {args.output}")


if __name__ == "__main__":
    main()
