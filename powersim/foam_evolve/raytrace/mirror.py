"""Inserted reflective objects: a rectangular mirror panel and a chrome ball."""

import math
from dataclasses import dataclass

import torch


@dataclass
class MirrorPlane:
    center: torch.Tensor  # (3,)
    normal: torch.Tensor  # (3,), unit
    tangent: torch.Tensor  # (3,), unit, in-plane "sideways"
    bitangent: torch.Tensor  # (3,), unit, in-plane "up"
    half_width: float
    half_height: float
    reflectivity: float = 1.0
    base_color: tuple = (0.05, 0.05, 0.05)
    ior: float = 1.10
    refraction_enabled: bool = False
    thickness: float = 0.05  # slab depth for the second (exit-surface) refraction
    border_width: float = 0.0  # world-unit rim inside the panel's edge shaded flat `border_color`
    border_color: tuple = (0.12, 0.08, 0.05)

    @staticmethod
    def from_transform(
        translate,
        rotate_deg,
        scale,
        device,
        reflectivity: float = 1.0,
        ior: float = 1.10,
        refraction_enabled: bool = False,
        thickness: float = 0.05,
        border_width: float = 0.0,
        border_color: tuple = (0.12, 0.08, 0.05),
    ) -> "MirrorPlane":
        """Builds a MirrorPlane from the Translate/Rotate/Scale values read off
        render/tools/mirror_placement_viewer's interactive 3D placement tool (a THREE.js artifact)"""
        x, y, z = (math.radians(a) for a in rotate_deg)
        c1, s1 = math.cos(x), math.sin(x)
        c2, s2 = math.cos(y), math.sin(y)
        c3, s3 = math.cos(z), math.sin(z)
        R = torch.tensor(
            [
                [c2 * c3, -c2 * s3, s2],
                [c1 * s3 + s1 * s2 * c3, c1 * c3 - s1 * s2 * s3, -s1 * c2],
                [s1 * s3 - c1 * s2 * c3, s1 * c3 + c1 * s2 * s3, c1 * c2],
            ],
            dtype=torch.float32,
            device=device,
        )

        local_normal = torch.tensor([0.0, 0.0, 1.0], device=device)
        local_tangent = torch.tensor([1.0, 0.0, 0.0], device=device)
        local_bitangent = torch.tensor([0.0, 1.0, 0.0], device=device)

        normal = R @ local_normal
        tangent = R @ local_tangent
        bitangent = R @ local_bitangent

        return MirrorPlane(
            center=torch.tensor(translate, dtype=torch.float32, device=device),
            normal=normal,
            tangent=tangent,
            bitangent=bitangent,
            half_width=float(scale[0]) / 2.0,
            half_height=float(scale[1]) / 2.0,
            reflectivity=reflectivity,
            ior=ior,
            refraction_enabled=refraction_enabled,
            thickness=thickness,
            border_width=float(border_width),
            border_color=tuple(border_color),
        )


@dataclass
class ChromeBall:
    center: torch.Tensor  # (3,)
    radius: float
    reflectivity: float = 1.0
    base_color: tuple = (0.05, 0.05, 0.05)
    ior: float = 1.10
    refraction_enabled: bool = False

    @staticmethod
    def from_transform(
        translate,
        radius,
        device,
        reflectivity: float = 1.0,
        ior: float = 1.10,
        refraction_enabled: bool = False,
    ) -> "ChromeBall":
        """Builds a ChromeBall from the Translate/Radius values read off the interactive 3D placement
        tool in its "chrome ball" mode."""
        return ChromeBall(
            center=torch.tensor(translate, dtype=torch.float32, device=device),
            radius=float(radius),
            reflectivity=reflectivity,
            ior=ior,
            refraction_enabled=refraction_enabled,
        )
