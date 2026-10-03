"""Locate ffmpeg for --compile-video."""

import shutil


def ffmpeg_exe() -> str:
    """Path to an ffmpeg binary: the one on PATH, else the one bundled by imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
    except ImportError as e:
        raise RuntimeError(
            "ffmpeg not found on PATH and imageio-ffmpeg is not installed; "
            "`python -m pip install imageio-ffmpeg` (see README, Setting up Environments)"
        ) from e
    return imageio_ffmpeg.get_ffmpeg_exe()
