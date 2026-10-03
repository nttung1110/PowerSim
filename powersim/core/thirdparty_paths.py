"""Put third_party/powerfoam and third_party/PhysGaussian on sys.path without letting their two
`utils` packages collide."""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PHYSGAUSSIAN_ROOT = REPO_ROOT / "third_party" / "PhysGaussian"
POWERFOAM_ROOT = REPO_ROOT / "third_party" / "powerfoam"


def _ensure_on_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def ensure_physgaussian_on_path() -> None:
    """Put PhysGaussian's repo root on sys.path."""
    _ensure_on_path(PHYSGAUSSIAN_ROOT)


def ensure_powerfoam_on_path() -> None:
    """Put PowerFoam's repo root on sys.path (``powerfoam``, ``configs``, ``data_loader``)"""
    _ensure_on_path(POWERFOAM_ROOT)


def load_module_from_path(module_name: str, file_path: Path):
    """Load a single .py file as a module without touching sys.path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_physgaussian_decode_param():
    """Load PhysGaussian's utils/decode_param.py by path."""
    return load_module_from_path(
        "physgaussian_decode_param",
        PHYSGAUSSIAN_ROOT / "utils" / "decode_param.py",
    )
