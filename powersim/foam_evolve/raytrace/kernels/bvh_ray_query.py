"""BVH over the primitives' bounding spheres, for nearest-hit queries of secondary rays."""

import torch
import warp as wp

from .rendering_math_ext import ray_sphere_intersect


def build_scene_bvh(points: torch.Tensor, radii: torch.Tensor, device) -> wp.Bvh:
    lowers = wp.from_torch(points - radii[:, None], dtype=wp.vec3f, requires_grad=False)
    uppers = wp.from_torch(points + radii[:, None], dtype=wp.vec3f, requires_grad=False)

    torch_stream = torch.cuda.current_stream()
    wp_stream = wp.stream_from_torch(torch_stream)

    with wp.ScopedDevice(str(device)):
        with wp.ScopedStream(wp_stream):
            return wp.Bvh(lowers=lowers, uppers=uppers)


@wp.func
def nearest_ray_hit(
    bvh_id: wp.uint64,
    ray_o: wp.vec3f,
    ray_d: wp.vec3f,
    exclude_idx: int,
    spheres: wp.array(dtype=wp.vec4f),
    nsigmas: wp.array(dtype=wp.vec4f),
    min_sigma: float,
    radius_scale: float,
):
    """Nearest primitive (sphere: center=spheres[i].xyz, radius=spheres[i].w) hit by the ray (ray_o,
    ray_d), excluding `exclude_idx`."""
    query = wp.bvh_query_ray(bvh_id, ray_o, ray_d)
    j = wp.int32(0)

    best_hit = bool(False)
    best_idx = int(0)
    best_t = float(1.0e10)

    while wp.bvh_query_next(query, j):
        if j == exclude_idx:
            continue
        if nsigmas[j][3] < min_sigma:
            continue

        sphere = spheres[j]
        center = wp.vec3f(sphere[0], sphere[1], sphere[2])
        radius = sphere[3] * radius_scale

        hit, t_near, t_far = ray_sphere_intersect(ray_o, ray_d, center, radius)
        if hit and t_near < best_t:
            best_hit = True
            best_idx = j
            best_t = t_near

    return best_hit, best_idx, best_t
