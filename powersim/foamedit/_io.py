"""Shared helpers for the foamedit CLIs: writing an edited checkpoint and before/after renders."""

import shutil
from pathlib import Path

from powersim.core.frame_renderer import render_rgb, rgb_to_uint8_image, save_uint8_rgb_png


def save_edited_checkpoint(scene, source_checkpoint_dir, output_dir) -> Path:
    """Write model.pt next to copies of the source checkpoint's config.yaml and cameras_*.pt, so the
    result loads like any other checkpoint (same cameras: the scene frame is unchanged)"""
    source_checkpoint_dir = Path(source_checkpoint_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene.save_pt(str(output_dir / "model.pt"))
    shutil.copy2(source_checkpoint_dir / "config.yaml", output_dir / "config.yaml")
    for cam in source_checkpoint_dir.glob("cameras_*.pt"):
        shutil.copy2(cam, output_dir / cam.name)
    return output_dir


def render_view(scene, camera, path) -> None:
    save_uint8_rgb_png(rgb_to_uint8_image(render_rgb(scene, camera)), str(path))
