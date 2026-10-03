"""Render a PowerfoamScene from a camera and save the image."""

import numpy as np
import torch


@torch.no_grad()
def render_rgb(scene, camera, background=None) -> torch.Tensor:
    """Returns an (H, W, 3) float tensor in [0, 1]."""
    rgb, alpha, *_ = scene.forward(camera)
    if background is None:
        return rgb
    bg = torch.as_tensor(background, dtype=rgb.dtype, device=rgb.device)
    return rgb + (1.0 - alpha[..., None]) * bg


def rgb_to_uint8_image(rgb: torch.Tensor) -> np.ndarray:
    """(H, W, 3) float [0, 1] tensor -> (H, W, 3) uint8 numpy array, RGB order."""
    img = rgb.detach().clamp(0.0, 1.0).cpu().numpy()
    return (img * 255.0).round().astype(np.uint8)


def save_uint8_rgb_png(img: np.ndarray, path: str) -> None:
    """Write an already-uint8 (H,W,3) RGB array as PNG."""
    import cv2

    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
