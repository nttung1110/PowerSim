"""Plot per-primitive E / nu / density over a rendered frame."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def project_to_pixels(camera, positions: torch.Tensor):
    """Invert the eye/right/up pinhole model to get (row, col, in_front) for world points."""
    eye = camera.eye.to(positions.device)
    right = camera.right.to(positions.device)
    up = camera.up.to(positions.device)
    forward = torch.linalg.cross(up, right)
    forward = forward / forward.norm()

    d = positions - eye
    t = d @ forward
    x = (d @ right) / (t * right.norm() ** 2)
    y = (d @ up) / (t * up.norm() ** 2)

    W, H = camera.width, camera.height
    col = (x + 1.0) * (W - 1) / 2.0
    row = (1.0 - y) * (H - 1) / 2.0
    in_front = t > 0
    return row.cpu().numpy(), col.cpu().numpy(), in_front.cpu().numpy()


_PANEL_SPECS = [
    ("E", "log10(E) [Pa]", "inferno", True),
    ("nu", "Poisson's ratio", "viridis", False),
    # "cividis" was here before: its low end is a dark blue-purple that blends into a blue backdrop
    ("density", "log10(density) [kg/m^3]", "plasma", True),
]


def render_material_field_triptych(
    backdrop_img: np.ndarray,
    camera,
    positions: torch.Tensor,
    E: torch.Tensor,
    nu: torch.Tensor,
    density: torch.Tensor,
    output_path: str,
    covered: torch.Tensor = None,
    point_size: float = 2.5,
    title_prefix: str = "",
):
    """Save a horizontally-concatenated E / nu / density spatial visualization."""
    row, col, in_front = project_to_pixels(camera, positions)
    H, W = backdrop_img.shape[:2]
    valid = in_front & (row >= 0) & (row < H) & (col >= 0) & (col < W)

    covered_np = covered.cpu().numpy() if covered is not None else np.ones(positions.shape[0], dtype=bool)
    values = {"E": E, "nu": nu, "density": density}

    fig, axes = plt.subplots(1, 3, figsize=(3 * W / 100, H / 100), dpi=100)
    for ax, (key, label, cmap, log_scale) in zip(axes, _PANEL_SPECS):
        ax.imshow(backdrop_img)
        v = values[key].detach().cpu().numpy()
        v = np.log10(np.clip(v, 1e-12, None)) if log_scale else v

        uncovered_mask = valid & (~covered_np)
        ax.scatter(col[uncovered_mask], row[uncovered_mask], s=point_size * 0.6, c="0.5", alpha=0.4, linewidths=0)

        covered_mask = valid & covered_np
        sc = ax.scatter(col[covered_mask], row[covered_mask], s=point_size, c=v[covered_mask], cmap=cmap, alpha=0.85, linewidths=0)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label(label)
        ax.set_title(f"{title_prefix}{key}")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
