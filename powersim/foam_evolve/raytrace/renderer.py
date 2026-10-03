"""Feeds a PowerfoamScene to the ray-tracing kernel."""

import torch

from .kernels.bvh_ray_query import build_scene_bvh
from .kernels.raytrace_kernel import RayTraceKernels
from .material import RayTraceMaterial
from .mirror import ChromeBall, MirrorPlane
from .scene_prep import (
    compute_adjacency_diff,
    compute_start_point_idx,
    compute_texel_rgb,
    prepare_raytrace_buffers,
)


class RayTraceRenderer:
    def __init__(self, scene, secondary_min_brightness=0.03, secondary_max_brightness=1.02,
                 secondary_radius_scale=1.0):
        self.secondary_radius_scale = float(secondary_radius_scale)
        self.kernels = RayTraceKernels(
            scene.args, scene.device, scene.attr_dtype,
            secondary_min_brightness=secondary_min_brightness,
            secondary_max_brightness=secondary_max_brightness,
            secondary_radius_scale=secondary_radius_scale,
        )

    def render(
        self,
        scene,
        camera,
        material: RayTraceMaterial = None,
        mirror: MirrorPlane = None,
        ball: ChromeBall = None,
        adjacency: torch.Tensor = None,
        adjacency_offsets: torch.Tensor = None,
        transmittance_threshold: float = 1e-3,
    ) -> torch.Tensor:
        """Returns an (H, W, 3) float tensor in [0, 1] (same convention as
        powersim.core.frame_renderer.render_rgb)"""
        if mirror is not None and ball is not None:
            raise ValueError("pass at most one of mirror and ball")
        if material is None:
            material = RayTraceMaterial()
        if adjacency is None:
            adjacency = scene.adjacency
        if adjacency_offsets is None:
            adjacency_offsets = scene.adjacency_offsets

        with torch.no_grad():
            buffers = prepare_raytrace_buffers(scene)
            texel_rgb = compute_texel_rgb(scene, buffers, camera)
            adjacency_diff = compute_adjacency_diff(
                buffers.points, buffers.radii, adjacency, adjacency_offsets
            )
            start_point_idx = compute_start_point_idx(buffers.points, buffers.radii, camera)
            # BVH bounds inflated to match nearest_ray_hit's secondary_radius_scale
            bvh = build_scene_bvh(buffers.points, buffers.radii * self.secondary_radius_scale, scene.device)

            obj_kwargs = {}
            if mirror is not None:
                obj_kwargs = dict(
                    mirror_enabled=True,
                    mirror_center=mirror.center,
                    mirror_normal=mirror.normal,
                    mirror_tangent=mirror.tangent,
                    mirror_bitangent=mirror.bitangent,
                    mirror_half_width=mirror.half_width,
                    mirror_half_height=mirror.half_height,
                    mirror_reflectivity=mirror.reflectivity,
                    mirror_base_color=mirror.base_color,
                    mirror_ior=mirror.ior,
                    mirror_refraction_enabled=mirror.refraction_enabled,
                    mirror_thickness=mirror.thickness,
                    mirror_border_width=mirror.border_width,
                    mirror_border_color=mirror.border_color,
                )
            elif ball is not None:
                obj_kwargs = dict(
                    ball_enabled=True,
                    ball_center=ball.center,
                    ball_radius=ball.radius,
                    ball_reflectivity=ball.reflectivity,
                    ball_base_color=ball.base_color,
                    ball_ior=ball.ior,
                    ball_refraction_enabled=ball.refraction_enabled,
                )

            return self.kernels.render(
                camera,
                start_point_idx,
                bvh.id,
                buffers.points,
                buffers.radii,
                buffers.density,
                buffers.normals,
                buffers.texel_sites,
                texel_rgb,
                buffers.att_sites,
                buffers.att_values,
                buffers.att_temps,
                buffers.texel_height,
                adjacency,
                adjacency_offsets,
                adjacency_diff,
                transmittance_threshold=transmittance_threshold,
                reflectivity=material.reflectivity,
                ior=material.ior,
                refraction_enabled=material.refraction_enabled,
                bkgd_color=material.background_color,
                **obj_kwargs,
            )
