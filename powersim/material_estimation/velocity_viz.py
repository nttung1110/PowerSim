"""Plot the recovered initial-velocity magnitude over a rendered frame."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from powersim.foam_evolve.viz.material_field_viz import project_to_pixels


def plot_velocity_field(backdrop, camera, positions, v0, output_path, title_prefix=""):
    """|v0| spatial plot in the same backdrop-overlay style as render_material_field_triptych's own
    panels."""
    row, col, in_front = project_to_pixels(camera, positions)
    H, W = backdrop.shape[:2]
    valid = in_front & (row >= 0) & (row < H) & (col >= 0) & (col < W)
    speed = v0.norm(dim=-1).detach().cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(W / 100, H / 100), dpi=100)
    ax.imshow(backdrop)
    sc = ax.scatter(col[valid], row[valid], s=2.5, c=speed[valid], cmap="inferno", alpha=0.85, linewidths=0)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("|v0| [m/s]")
    ax.set_title(f"{title_prefix}|v0|")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
