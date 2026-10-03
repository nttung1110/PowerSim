"""Scene-wide reflectivity and index of refraction for the ray tracer."""

import json
from dataclasses import dataclass


@dataclass
class RayTraceMaterial:
    reflectivity: float = 0.0
    ior: float = 1.10
    refraction_enabled: bool = False
    background_color: tuple = (0.0, 0.0, 0.0)
    max_bounces: int = 1  # single-bounce only, see the plan; not yet wired past 1

    @staticmethod
    def load(path: str) -> "RayTraceMaterial":
        with open(path) as f:
            data = json.load(f)
        return RayTraceMaterial(
            reflectivity=float(data.get("reflectivity", 0.0)),
            ior=float(data.get("ior", 1.10)),
            refraction_enabled=bool(data.get("refraction_enabled", False)),
            background_color=tuple(data.get("background_color", [0.0, 0.0, 0.0])),
            max_bounces=int(data.get("max_bounces", 1)),
        )
