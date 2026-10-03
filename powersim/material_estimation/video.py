"""ffmpeg helpers: compile frames, stack two videos side by side, extract frames."""

import subprocess

from powersim.core.video import ffmpeg_exe
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def compile_video(frame_dir, fps, video_path) -> None:
    """Compiles frame_dir/frame_%04d.png into video_path via ffmpeg."""
    frame_dir = Path(frame_dir)
    video_path = Path(video_path)
    probe = frame_dir / "frame_0000.png"
    if not probe.exists():
        print(f"no frames found in {frame_dir}, skipping video compile")
        return
    subprocess.run(
        [
            ffmpeg_exe(), "-framerate", str(fps),
            "-i", str(frame_dir / "frame_%04d.png"),
            "-c:v", "libx264", "-y", "-pix_fmt", "yuv420p",
            str(video_path),
        ],
        check=True,
    )
    print(f"compiled {video_path}")


def hstack_videos(left_video, right_video, output_video) -> None:
    """Concatenates two videos side by side (left | right) via ffmpeg's hstack filter."""
    left_video, right_video, output_video = Path(left_video), Path(right_video), Path(output_video)
    if not left_video.exists() or not right_video.exists():
        print(f"missing input video ({left_video} or {right_video}), skipping hstack")
        return
    subprocess.run(
        [
            ffmpeg_exe(), "-i", str(left_video), "-i", str(right_video),
            "-filter_complex", "hstack=inputs=2",
            "-y", str(output_video),
        ],
        check=True,
    )
    print(f"compiled {output_video}")


def load_video_frames(video_path, target_hw, fps, device="cuda:0", max_frames=None):
    """Extracts frames from a real captured video."""
    H, W = target_hw
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"GT video not found: {video_path}")

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [
                ffmpeg_exe(), "-i", str(video_path),
                "-vf", f"fps={fps},scale={W}:{H}",
                "-y", str(Path(tmp) / "frame_%04d.png"),
            ],
            check=True,
        )
        frame_files = sorted(Path(tmp).glob("frame_*.png"))
        if not frame_files:
            raise RuntimeError(f"ffmpeg extracted 0 frames from {video_path}")
        if max_frames is not None:
            frame_files = frame_files[:max_frames]

        frames = []
        for f in frame_files:
            arr = np.asarray(Image.open(f).convert("RGB"), dtype=np.float32) / 255.0
            frames.append(torch.from_numpy(arr).to(device))

    print(f"loaded {len(frames)} frames from {video_path} at {fps:.3f}fps, resized to {W}x{H}")
    return frames
