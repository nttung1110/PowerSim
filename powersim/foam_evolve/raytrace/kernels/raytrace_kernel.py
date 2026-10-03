"""Ray-tracing kernel, forked from PowerFoam's raytrace.py: primary rays walk the power diagram;
secondary rays reflect or refract off the scene or an inserted mirror / ball."""

import torch
import warp as wp

from powersim.core.thirdparty_paths import ensure_powerfoam_on_path

# Puts third_party/powerfoam on sys.path so `powerfoam.camera` resolves regardless of whether the
# caller already loaded a checkpoint (which does this too, as a side effect)
ensure_powerfoam_on_path()
from powerfoam.camera import WarpCamera, get_ray_dir  # noqa: E402

from .bvh_ray_query import nearest_ray_hit
from .rendering_math_ext import (
    fresnel_reflectance,
    ray_finite_plane_intersect,
    ray_plane_intersect,
    ray_pface_intersect_diff,
    ray_sphere_intersect,
    reflect_ray,
    refract_ray,
)

TILE_WIDTH = 8
TILE_SIZE = TILE_WIDTH * TILE_WIDTH


def _as_vec3f(v) -> wp.vec3f:
    """mirror.py's MirrorPlane fields are torch tensors (device-resident); wp.vec3f wants plain Python
    floats."""
    if hasattr(v, "tolist"):
        v = v.tolist()
    return wp.vec3f(float(v[0]), float(v[1]), float(v[2]))


class RayTraceKernels:
    def __init__(self, args, device, attr_dtype="float",
                 secondary_min_brightness=0.03, secondary_max_brightness=1.02,
                 secondary_radius_scale=1.0):
        """secondary_min/max_brightness: the reflection/refraction ray's hit-validity window."""
        self.device = device
        self.args = args

        if attr_dtype == "float":
            scalar = wp.float32
            vec3s = wp.vec3f
            self.tscalar = torch.float32
        elif attr_dtype == "half":
            scalar = wp.float16
            vec3s = wp.vec3h
            self.tscalar = torch.float16
        else:
            raise ValueError(f"Unsupported attribute dtype: {attr_dtype}")
        self.attr_dtype = attr_dtype

        num_texel_sites = args.num_texel_sites
        sv_dof = args.sv_dof
        temp = wp.constant(10.0)

        @wp.func
        def evaluate_sv_color(
            view_dir: wp.vec3f,
            att_sites: wp.array2d(dtype=vec3s),
            att_values: wp.array2d(dtype=vec3s),
            att_temps: wp.array2d(dtype=scalar),
            texel_flat_idx: int,
        ):
            """Forked from third_party/powerfoam/powerfoam/color_fn.py's `spherical_voronoi_fwd_kernel`
            per-thread body (its per-point weighted directional blend over `sv_dof` learned (axis,
            value, temp) triples)"""
            weights_sum = float(0.0)
            value_sum = wp.vec3f(0.0, 0.0, 0.0)
            for i in range(sv_dof):
                _axis = att_sites[i, texel_flat_idx]
                _val = att_values[i, texel_flat_idx]
                _temp = att_temps[i, texel_flat_idx]

                axis = wp.vec3f(float(_axis[0]), float(_axis[1]), float(_axis[2]))
                val = wp.vec3f(float(_val[0]), float(_val[1]), float(_val[2]))
                temp_i = float(_temp)

                dist = wp.length(view_dir - axis)
                weight = wp.exp(-temp_i * dist)
                weights_sum += weight
                value_sum += weight * val

            weights_sum = wp.max(weights_sum, 1e-20)
            value = value_sum / weights_sum + wp.vec3f(0.5, 0.5, 0.5)
            # Deliberately NOT clamped from above here (only the pre-existing lower clamp)
            return wp.vec3f(
                wp.max(value[0], 0.0), wp.max(value[1], 0.0), wp.max(value[2], 0.0)
            )

        @wp.func
        def plane_intersection_fwd_local(
            ray_origin: wp.vec3f,
            ray_direction: wp.vec3f,
            t_near: float,
            plane_origin: wp.vec3f,
            plane_normal: wp.vec3f,
            radius: float,
            sites: wp.array(dtype=vec3s),
            rgbs: wp.array(dtype=vec3s),
            heights: wp.array(dtype=scalar),
            num_sites: int,
        ):
            _t_surf, _dp = ray_plane_intersect(
                ray_origin, ray_direction, plane_origin, plane_normal
            )
            _t_query = t_near if _dp >= 0.0 else wp.max(t_near, _t_surf)
            _intersection_pt = ray_origin + _t_query * ray_direction

            inv_radius_sq = 1.0 / (radius * radius)

            height_sum = float(0.0)
            _weight_sum = float(0.0)
            for i in range(num_sites):
                site = sites[i]
                site_f = wp.vec3f(float(site[0]), float(site[1]), float(site[2]))
                dist_sq = wp.length_sq(_intersection_pt - site_f) * inv_radius_sq
                weight = wp.exp(-temp * dist_sq)

                height = float(heights[i])
                height_sum += weight * height
                _weight_sum += weight

            _weight_sum = wp.max(_weight_sum, 1e-20)
            height_out = height_sum / _weight_sum

            t_surf, dp = ray_plane_intersect(
                ray_origin,
                ray_direction,
                plane_origin,
                plane_normal,
                wp.float32(height_out),
            )
            t_query = t_near if dp >= 0.0 else wp.max(t_near, t_surf)
            intersection_pt = ray_origin + t_query * ray_direction

            rgb_sum = wp.vec3f(0.0, 0.0, 0.0)
            weight_sum = float(0.0)
            for i in range(num_sites):
                site = sites[i]
                site_f = wp.vec3f(float(site[0]), float(site[1]), float(site[2]))
                dist_sq = wp.length_sq(intersection_pt - site_f) * inv_radius_sq
                weight = wp.exp(-temp * dist_sq)

                _rgb = rgbs[i]
                rgb = wp.vec3f(float(_rgb[0]), float(_rgb[1]), float(_rgb[2]))
                rgb_sum += weight * rgb
                weight_sum += weight

            weight_sum = wp.max(weight_sum, 1e-20)
            rgb_out = rgb_sum / weight_sum

            return (
                _t_surf,
                _dp,
                height_out,
                _weight_sum,
                t_surf,
                dp,
                rgb_out,
                weight_sum,
            )

        @wp.func
        def plane_intersection_fwd_dynamic_sv(
            ray_origin: wp.vec3f,
            ray_direction: wp.vec3f,
            t_near: float,
            plane_origin: wp.vec3f,
            plane_normal: wp.vec3f,
            radius: float,
            sites: wp.array(dtype=vec3s),
            att_sites: wp.array2d(dtype=vec3s),
            att_values: wp.array2d(dtype=vec3s),
            att_temps: wp.array2d(dtype=scalar),
            texel_base: int,
            view_dir: wp.vec3f,
            heights: wp.array(dtype=scalar),
            num_sites: int,
        ):
            """`plane_intersection_fwd_local`'s exact twin, except the per-texel color comes from a
            live `evaluate_sv_color(view_dir, ...)` call instead of a precomputed `rgbs` array."""
            _t_surf, _dp = ray_plane_intersect(
                ray_origin, ray_direction, plane_origin, plane_normal
            )
            _t_query = t_near if _dp >= 0.0 else wp.max(t_near, _t_surf)
            _intersection_pt = ray_origin + _t_query * ray_direction

            inv_radius_sq = 1.0 / (radius * radius)

            height_sum = float(0.0)
            _weight_sum = float(0.0)
            for i in range(num_sites):
                site = sites[i]
                site_f = wp.vec3f(float(site[0]), float(site[1]), float(site[2]))
                dist_sq = wp.length_sq(_intersection_pt - site_f) * inv_radius_sq
                weight = wp.exp(-temp * dist_sq)

                height = float(heights[i])
                height_sum += weight * height
                _weight_sum += weight

            _weight_sum = wp.max(_weight_sum, 1e-20)
            height_out = height_sum / _weight_sum

            t_surf, dp = ray_plane_intersect(
                ray_origin,
                ray_direction,
                plane_origin,
                plane_normal,
                wp.float32(height_out),
            )
            t_query = t_near if dp >= 0.0 else wp.max(t_near, t_surf)
            intersection_pt = ray_origin + t_query * ray_direction

            rgb_sum = wp.vec3f(0.0, 0.0, 0.0)
            weight_sum = float(0.0)
            for i in range(num_sites):
                site = sites[i]
                site_f = wp.vec3f(float(site[0]), float(site[1]), float(site[2]))
                dist_sq = wp.length_sq(intersection_pt - site_f) * inv_radius_sq
                weight = wp.exp(-temp * dist_sq)

                rgb = evaluate_sv_color(
                    view_dir, att_sites, att_values, att_temps, texel_base + i
                )
                rgb_sum += weight * rgb
                weight_sum += weight

            weight_sum = wp.max(weight_sum, 1e-20)
            rgb_out = rgb_sum / weight_sum

            return (
                _t_surf,
                _dp,
                height_out,
                _weight_sum,
                t_surf,
                dp,
                rgb_out,
                weight_sum,
            )

        MAX_SECONDARY_RETRIES = wp.constant(8)
        MIN_SECONDARY_BRIGHTNESS = wp.constant(float(secondary_min_brightness))
        # Sphere inflation for the reflection ray's nearest-hit test only : 1.0 = exact spheres; >1
        # closes the gaps of thin single-layer objects at the cost of slightly puffed reflected
        SECONDARY_RADIUS_SCALE = wp.constant(float(secondary_radius_scale))
        # Symmetric counterpart to MIN_SECONDARY_BRIGHTNESS: a hit reading suspiciously near-
        # saturated white is the same failure mode as near-black
        MAX_SECONDARY_BRIGHTNESS = wp.constant(float(secondary_max_brightness))
        # A radius-scaled march step (~1e-3 * radius) turned out to be microscopic relative to the
        # spacing between primitives
        SECONDARY_RETRY_STEP = wp.constant(0.1)
        # Marching the ORIGIN forward (above) while keeping the same ray DIRECTION changes which
        # primitive gets hit, but not the view-direction query into evaluate_sv_color
        JITTER_BASE_ANGLE = wp.constant(2.399963)
        JITTER_MAX_SPREAD = wp.constant(0.35)

        @wp.func
        def jitter_secondary_dir(ray_d: wp.vec3f, attempt: int, max_retries: int):
            if attempt == 0:
                return ray_d
            up = wp.vec3f(0.0, 1.0, 0.0)
            if wp.abs(wp.dot(ray_d, up)) > 0.95:
                up = wp.vec3f(1.0, 0.0, 0.0)
            perp1 = wp.normalize(wp.cross(ray_d, up))
            perp2 = wp.cross(ray_d, perp1)
            angle = JITTER_BASE_ANGLE * float(attempt)
            spread = JITTER_MAX_SPREAD * (float(attempt) / float(max_retries))
            offset = spread * (wp.cos(angle) * perp1 + wp.sin(angle) * perp2)
            return wp.normalize(ray_d + offset)

        @wp.func
        def find_valid_secondary_hit(
            bvh_id: wp.uint64,
            ray_o: wp.vec3f,
            ray_d: wp.vec3f,
            exclude_idx: int,
            all_spheres: wp.array(dtype=wp.vec4f),
            all_nsigmas: wp.array(dtype=wp.vec4f),
            all_texel_sites: wp.array2d(dtype=vec3s),
            all_att_sites: wp.array2d(dtype=vec3s),
            all_att_values: wp.array2d(dtype=vec3s),
            all_att_temps: wp.array2d(dtype=scalar),
            all_texel_height: wp.array2d(dtype=scalar),
        ):
            """nearest_ray_hit's density filter (bvh_ray_query.py) only screens out geometrically
            invalid/near-zero-opacity primitives."""
            cur_o = ray_o
            cur_exclude = exclude_idx
            found = bool(False)
            color = wp.vec3f(0.0, 0.0, 0.0)

            # A genuine miss (nearest_ray_hit finds no primitive at all) used to `break`
            # immediately, with zero retries
            for _attempt in range(MAX_SECONDARY_RETRIES):
                try_d = jitter_secondary_dir(ray_d, _attempt, MAX_SECONDARY_RETRIES)
                hit, idx, t = nearest_ray_hit(
                    bvh_id, cur_o, try_d, cur_exclude, all_spheres, all_nsigmas, 1.0e-3,
                    SECONDARY_RADIUS_SCALE,
                )
                if hit:
                    sphere = all_spheres[idx]
                    center = wp.vec3f(sphere[0], sphere[1], sphere[2])
                    radius = sphere[3]
                    nsigma = all_nsigmas[idx]
                    normal = wp.vec3f(float(nsigma[0]), float(nsigma[1]), float(nsigma[2]))

                    _, _, _, _, _, _, c, _ = plane_intersection_fwd_dynamic_sv(
                        cur_o,
                        try_d,
                        t,
                        center,
                        normal,
                        radius,
                        all_texel_sites[idx],
                        all_att_sites,
                        all_att_values,
                        all_att_temps,
                        idx * num_texel_sites,
                        try_d,
                        all_texel_height[idx],
                        num_texel_sites,
                    )
                    found = True
                    color = c

                    brightness = wp.max(wp.max(c[0], c[1]), c[2])
                    if brightness >= MIN_SECONDARY_BRIGHTNESS and brightness <= MAX_SECONDARY_BRIGHTNESS:
                        break

                    hit_pt = cur_o + t * try_d
                    cur_o = hit_pt + SECONDARY_RETRY_STEP * try_d
                    cur_exclude = int(-1)

            # Safety-net clamp: whatever was finally settled on (a genuinely valid hit, or the last-
            # seen color after exhausting all retries) is clamped to display range here, once
            color = wp.vec3f(
                wp.clamp(color[0], 0.0, 1.0),
                wp.clamp(color[1], 0.0, 1.0),
                wp.clamp(color[2], 0.0, 1.0),
            )
            return found, color

        BOUNCE_WALK_MAX_ITERS = wp.constant(2000)
        GUARD_MIN_T = wp.constant(0.05)        # ignore spheres straddling the bounce origin
        GUARD_MIN_TAU = wp.constant(1.5)       # optical depth across the diameter: 'opaque'
        GUARD_MAX_ACCUM = wp.constant(0.7)     # walk transmittance still above this = it saw nothing

        @wp.func
        def continue_walk_from_bounce(
            bvh_id: wp.uint64,
            start_prim_idx: int,
            start_pt_near: float,
            ray_o: wp.vec3f,
            ray_d: wp.vec3f,
            check_ball_exit: bool,
            ball_center: wp.vec3f,
            ball_radius: float,
            obj_ior: float,
            all_spheres: wp.array(dtype=wp.vec4f),
            all_nsigmas: wp.array(dtype=wp.vec4f),
            all_texel_sites: wp.array2d(dtype=vec3s),
            all_att_sites: wp.array2d(dtype=vec3s),
            all_att_values: wp.array2d(dtype=vec3s),
            all_att_temps: wp.array2d(dtype=scalar),
            all_texel_height: wp.array2d(dtype=scalar),
            adjacency: wp.array(dtype=wp.int32),
            adjacency_offsets: wp.array(dtype=wp.int32),
            adjacency_diff: wp.array(dtype=wp.vec4h),
            transmittance_threshold: float,
            bkgd_color: wp.vec3f,
            guard_enabled: int,
            guard_bvh_id: wp.uint64,
            guard_spheres: wp.array(dtype=wp.vec4f),
            guard_nsigmas: wp.array(dtype=wp.vec4f),
            guard_map: wp.array(dtype=wp.int32),
        ):
            """Author-inspired walk-continuation for a reflected/refracted ray leaving the chrome ball."""
            rgb = wp.vec3f(0.0, 0.0, 0.0)
            log_t = float(0.0)
            prim_idx = start_prim_idx
            pt_near = start_pt_near
            pending_exit = check_ball_exit

            # Skip guard : the walk resumes from the primary ray's cell at the bounce and can hop
            # straight past a THIN inserted object
            g_active = bool(False)
            g_t = float(0.0)
            g_idx = int(0)
            g_radius = float(0.0)
            g_sigma = float(0.0)
            if guard_enabled != 0:
                # guard BVH = ONLY the primitives the caller flagged (the inserted thin object);
                # querying the whole scene here misfires on dense captured content, where the
                # nearest sphere is
                g_hit, g_i, g_tt = nearest_ray_hit(
                    guard_bvh_id, ray_o, ray_d, int(-1), guard_spheres, guard_nsigmas, 1.0e-3, 1.0
                )
                if g_hit:
                    g_sphere = guard_spheres[g_i]
                    g_radius = g_sphere[3]
                    g_sigma = float(guard_nsigmas[g_i][3])
                    if g_tt > GUARD_MIN_T and g_sigma * 2.0 * g_radius > GUARD_MIN_TAU:
                        g_active = True
                        g_t = g_tt
                        g_idx = guard_map[g_i]

            iters = int(0)
            while True:
                iters += 1
                if iters > BOUNCE_WALK_MAX_ITERS:
                    break
                trans = wp.exp(log_t)
                if trans < transmittance_threshold:
                    break
                if prim_idx == int(0x7FFFFFFF):
                    break

                if g_active and pt_near > g_t + 2.0 * g_radius and trans > GUARD_MAX_ACCUM:
                    # The walk went past the nearest guarded sphere without seeing it: rewind and
                    # RESUME THE WALK from that primitive's own cell
                    prim_idx = g_idx
                    pt_near = g_t
                    g_active = False
                    continue

                sphere = all_spheres[prim_idx]
                center = wp.vec3f(sphere[0], sphere[1], sphere[2])
                radius = sphere[3]

                hit, t_near, t_far = ray_sphere_intersect(ray_o, ray_d, center, radius)
                v = center - ray_o
                if wp.length(v) < 4.0 * radius:
                    hit = False

                adj_offset_start = adjacency_offsets[prim_idx]
                adj_offset_end = adjacency_offsets[prim_idx + 1]
                n_adj = adj_offset_end - adj_offset_start

                next_prim_idx = int(0x7FFFFFFF)
                pt_far = float(1e10)
                for adj_idx in range(n_adj):
                    current_adj_offset = adj_offset_start + adj_idx
                    adj_point_idx = adjacency[current_adj_offset]
                    adj_diff = adjacency_diff[current_adj_offset]
                    diff = wp.vec3f(
                        float(adj_diff[0]), float(adj_diff[1]), float(adj_diff[2])
                    )
                    pm_diff = float(adj_diff[3])
                    t_face, dp = ray_pface_intersect_diff(ray_o, ray_d, diff, pm_diff)
                    if dp >= 0.0 and t_face < pt_far:
                        next_prim_idx = int(adj_point_idx)
                        pt_far = t_face
                    t_far = wp.min(t_face, t_far) if dp >= 0.0 else t_far
                    t_near = wp.max(t_face, t_near) if dp < 0.0 else t_near

                should_exit = bool(False)
                exit_t = float(0.0)
                if pending_exit:
                    exi_hit, exi_tn, exi_tf = ray_sphere_intersect(ray_o, ray_d, ball_center, ball_radius)
                    if exi_hit and exi_tf > 1.0e-5 and exi_tf < pt_far:
                        should_exit = True
                        exit_t = exi_tf
                    if should_exit and exit_t < t_far:
                        t_far = exit_t

                if next_prim_idx == int(0x7FFFFFFF):
                    if should_exit:
                        exit_pt = ray_o + exit_t * ray_d
                        n_out = wp.normalize(exit_pt - ball_center)
                        valid, r_out = refract_ray(ray_d, n_out, obj_ior)
                        if not valid:
                            r_out = reflect_ray(ray_d, n_out)
                        else:
                            pending_exit = False
                        r_out = wp.normalize(r_out)
                        ray_o = exit_pt + 1.0e-4 * r_out
                        ray_d = r_out
                        pt_near = float(0.0)
                        continue
                    rescue_origin = ray_o + (t_far + 1.0e-4) * ray_d
                    r_hit, r_idx, r_t = nearest_ray_hit(
                        bvh_id, rescue_origin, ray_d, prim_idx, all_spheres, all_nsigmas, 1.0e-3, 1.0
                    )
                    if r_hit:
                        r_sphere = all_spheres[r_idx]
                        r_center = wp.vec3f(r_sphere[0], r_sphere[1], r_sphere[2])
                        r_radius = r_sphere[3]
                        r_nsigma = all_nsigmas[r_idx]
                        r_normal = wp.vec3f(
                            float(r_nsigma[0]), float(r_nsigma[1]), float(r_nsigma[2])
                        )
                        _, _, _, _, _, _, r_color, _ = plane_intersection_fwd_dynamic_sv(
                            rescue_origin, ray_d, r_t, r_center, r_normal, r_radius,
                            all_texel_sites[r_idx], all_att_sites, all_att_values, all_att_temps,
                            r_idx * num_texel_sites, ray_d, all_texel_height[r_idx], num_texel_sites,
                        )
                        rgb += r_color * trans
                        log_t += -30.0
                    break

                nsigma = all_nsigmas[prim_idx]
                prim_normal = wp.vec3f(
                    float(nsigma[0]), float(nsigma[1]), float(nsigma[2])
                )
                sigma = float(nsigma[3])

                if not hit or t_near > t_far or sigma < 1e-3:
                    if should_exit:
                        exit_pt = ray_o + exit_t * ray_d
                        n_out = wp.normalize(exit_pt - ball_center)
                        valid, r_out = refract_ray(ray_d, n_out, obj_ior)
                        if not valid:
                            r_out = reflect_ray(ray_d, n_out)
                        else:
                            pending_exit = False
                        r_out = wp.normalize(r_out)
                        ray_o = exit_pt + 1.0e-4 * r_out
                        ray_d = r_out
                        pt_near = float(0.0)
                        continue
                    prim_idx = next_prim_idx
                    pt_near = wp.max(pt_near, pt_far)
                    continue

                _, _, height, _, t_surf, dp, color, _ = plane_intersection_fwd_dynamic_sv(
                    ray_o, ray_d, t_near, center, prim_normal, radius,
                    all_texel_sites[prim_idx], all_att_sites, all_att_values, all_att_temps,
                    prim_idx * num_texel_sites, ray_d, all_texel_height[prim_idx], num_texel_sites,
                )
                t_far = wp.min(t_surf, t_far) if dp >= 0.0 else t_far
                t_near = wp.max(t_surf, t_near) if dp < 0.0 else t_near

                dt = t_far - t_near
                if hit and dt > 0.0:
                    delta_log_t = -sigma * dt
                    alpha = 1.0 - wp.exp(delta_log_t)
                    rgb += color * alpha * trans
                    log_t += delta_log_t

                if should_exit:
                    exit_pt = ray_o + exit_t * ray_d
                    n_out = wp.normalize(exit_pt - ball_center)
                    valid, r_out = refract_ray(ray_d, n_out, obj_ior)
                    if not valid:
                        r_out = reflect_ray(ray_d, n_out)
                    else:
                        pending_exit = False
                    r_out = wp.normalize(r_out)
                    ray_o = exit_pt + 1.0e-4 * r_out
                    ray_d = r_out
                    pt_near = float(0.0)
                    continue

                prim_idx = next_prim_idx
                pt_near = wp.max(pt_near, pt_far)

            hit_something = log_t < -1.0e-4
            ray_trans = wp.exp(log_t)
            rgb += bkgd_color * ray_trans
            return rgb, hit_something

        @wp.kernel
        def raytrace_kernel(
            camera: WarpCamera,
            start_point_idx: int,
            frame_seed: int,
            bvh_id: wp.uint64,
            all_spheres: wp.array(dtype=wp.vec4f),
            all_nsigmas: wp.array(dtype=wp.vec4f),
            all_texel_sites: wp.array2d(dtype=vec3s),
            all_texel_rgb: wp.array2d(dtype=vec3s),
            all_att_sites: wp.array2d(dtype=vec3s),
            all_att_values: wp.array2d(dtype=vec3s),
            all_att_temps: wp.array2d(dtype=scalar),
            all_texel_height: wp.array2d(dtype=scalar),
            adjacency: wp.array(dtype=wp.int32),
            adjacency_offsets: wp.array(dtype=wp.int32),
            adjacency_diff: wp.array(dtype=wp.vec4h),
            transmittance_threshold: float,
            reflectivity: float,
            ior: float,
            refraction_enabled: int,
            bkgd_color: wp.vec3f,
            mirror_enabled: int,
            mirror_center: wp.vec3f,
            mirror_normal: wp.vec3f,
            mirror_tangent: wp.vec3f,
            mirror_bitangent: wp.vec3f,
            mirror_half_width: float,
            mirror_half_height: float,
            mirror_reflectivity: float,
            mirror_base_color: wp.vec3f,
            mirror_ior: float,
            mirror_refraction_enabled: int,
            mirror_thickness: float,
            mirror_border_width: float,
            mirror_border_color: wp.vec3f,
            ball_enabled: int,
            ball_center: wp.vec3f,
            ball_radius: float,
            ball_reflectivity: float,
            ball_base_color: wp.vec3f,
            ball_ior: float,
            ball_refraction_enabled: int,
            obj_debug_hitmask: int,
            obj_secondary_bvh: int,
            guard_enabled: int,
            guard_bvh_id: wp.uint64,
            guard_spheres: wp.array(dtype=wp.vec4f),
            guard_nsigmas: wp.array(dtype=wp.vec4f),
            guard_map: wp.array(dtype=wp.int32),
            primary_debug_walk: int,
            color_out: wp.array2d(dtype=wp.vec3f),
        ):
            thread_idx = wp.tid()
            tile_idx = thread_idx // TILE_SIZE

            tiles_h = 1 + (camera.height - 1) // TILE_WIDTH

            tile_i = tile_idx % tiles_h
            tile_j = tile_idx // tiles_h

            idx_in_tile = thread_idx % TILE_SIZE
            i_in_tile = idx_in_tile % TILE_WIDTH
            j_in_tile = idx_in_tile // TILE_WIDTH

            pix_i = tile_i * TILE_WIDTH + i_in_tile
            pix_j = tile_j * TILE_WIDTH + j_in_tile

            oob = pix_i >= camera.height or pix_j >= camera.width
            if oob:
                return

            pix_i = wp.min(pix_i, camera.height - 1)
            pix_j = wp.min(pix_j, camera.width - 1)

            ray_d = get_ray_dir(camera, float(pix_i), float(pix_j))
            ray_d = ray_d / wp.length(ray_d)
            ray_o = camera.eye

            obj_hit = bool(False)
            obj_t = float(0.0)
            obj_normal = wp.vec3f(0.0, 0.0, 1.0)
            obj_reflectivity = float(0.0)
            obj_base_color = wp.vec3f(0.0, 0.0, 0.0)
            obj_eps_scale = float(0.0)
            obj_ior = float(1.10)
            obj_refraction_enabled = int(0)
            if mirror_enabled != 0:
                obj_hit, obj_t = ray_finite_plane_intersect(
                    ray_o, ray_d, mirror_center, mirror_normal, mirror_tangent,
                    mirror_bitangent, mirror_half_width, mirror_half_height,
                )
                obj_normal = mirror_normal
                obj_reflectivity = mirror_reflectivity
                obj_base_color = mirror_base_color
                obj_eps_scale = mirror_half_width
                obj_ior = mirror_ior
                obj_refraction_enabled = mirror_refraction_enabled
                if obj_hit and mirror_border_width > 0.0:
                    # flat opaque rim inside the panel edge: a visible frame around the glass
                    m_rel = ray_o + obj_t * ray_d - mirror_center
                    m_u = wp.abs(wp.dot(m_rel, mirror_tangent))
                    m_v = wp.abs(wp.dot(m_rel, mirror_bitangent))
                    if (
                        m_u > mirror_half_width - mirror_border_width
                        or m_v > mirror_half_height - mirror_border_width
                    ):
                        obj_reflectivity = 0.0
                        obj_base_color = mirror_border_color
                        obj_refraction_enabled = 0
            elif ball_enabled != 0:
                b_hit, b_t_near, b_t_far = ray_sphere_intersect(ray_o, ray_d, ball_center, ball_radius)
                if b_hit and b_t_near > 0.0:
                    obj_hit = True
                    obj_t = b_t_near
                    obj_point = ray_o + b_t_near * ray_d
                    obj_normal = wp.normalize(obj_point - ball_center)
                    obj_reflectivity = ball_reflectivity
                    obj_base_color = ball_base_color
                    obj_eps_scale = ball_radius
                    obj_ior = ball_ior
                    obj_refraction_enabled = ball_refraction_enabled

            rgb = wp.vec3f(0.0, 0.0, 0.0)
            log_t = float(0.0)

            prim_idx = start_point_idx
            pt_near = float(0.0)

            hit_captured = bool(False)
            hit_point = wp.vec3f(0.0, 0.0, 0.0)
            hit_normal = wp.vec3f(0.0, 0.0, 1.0)
            hit_radius = float(0.0)
            hit_prim_idx = int(0)

            # Bounded, not `while True` (forked from PowerFoam's own raytrace.py, which uses the
            # same unbounded form): a real hang was observed and reproduced
            MAX_WALK_ITERS = int(20000)
            walk_iters = int(0)
            # walk_break_reason, set only when primary_debug_walk is on: 0 = fully attenuated
            # (normal exit), 1 = reached the inserted object's own depth, 2 = hit MAX_WALK_ITERS, 3
            # = adjacency
            walk_break_reason = int(0)
            # How many primitives actually contributed colour along this ray
            contrib_count = int(0)
            while True:
                walk_iters += 1
                if walk_iters > MAX_WALK_ITERS:
                    walk_break_reason = int(2)
                    break
                trans = wp.exp(log_t)
                if trans < transmittance_threshold:
                    break
                if prim_idx == int(0x7FFFFFFF):
                    walk_break_reason = int(3)
                    break
                if obj_hit and pt_near >= obj_t:
                    walk_break_reason = int(1)
                    break

                sphere = all_spheres[prim_idx]
                center = wp.vec3f(sphere[0], sphere[1], sphere[2])
                radius = sphere[3]

                hit, t_near, t_far = ray_sphere_intersect(ray_o, ray_d, center, radius)
                v = center - ray_o
                if wp.length(v) < 4.0 * radius:
                    hit = False

                adj_offset_start = adjacency_offsets[prim_idx]
                adj_offset_end = adjacency_offsets[prim_idx + 1]
                n_adj = adj_offset_end - adj_offset_start

                next_prim_idx = int(0x7FFFFFFF)
                pt_far = float(1e10)
                for adj_idx in range(n_adj):
                    current_adj_offset = adj_offset_start + adj_idx
                    adj_point_idx = adjacency[current_adj_offset]
                    adj_diff = adjacency_diff[current_adj_offset]

                    diff = wp.vec3f(
                        float(adj_diff[0]), float(adj_diff[1]), float(adj_diff[2])
                    )
                    pm_diff = float(adj_diff[3])

                    t_face, dp = ray_pface_intersect_diff(ray_o, ray_d, diff, pm_diff)

                    if dp >= 0.0 and t_face < pt_far:
                        next_prim_idx = int(adj_point_idx)
                        pt_far = t_face

                    t_far = wp.min(t_face, t_far) if dp >= 0.0 else t_far
                    t_near = wp.max(t_face, t_near) if dp < 0.0 else t_near

                if next_prim_idx == int(0x7FFFFFFF):
                    # Stale-adjacency safety net
                    rescue_origin = ray_o + (t_far + 1.0e-4) * ray_d
                    r_hit, r_idx, r_t = nearest_ray_hit(
                        bvh_id, rescue_origin, ray_d, prim_idx, all_spheres, all_nsigmas, 1.0e-3, 1.0
                    )
                    if r_hit:
                        r_sphere = all_spheres[r_idx]
                        r_center = wp.vec3f(r_sphere[0], r_sphere[1], r_sphere[2])
                        r_radius = r_sphere[3]
                        r_nsigma = all_nsigmas[r_idx]
                        r_normal = wp.vec3f(
                            float(r_nsigma[0]), float(r_nsigma[1]), float(r_nsigma[2])
                        )
                        _, _, _, _, _, _, r_color, _ = plane_intersection_fwd_local(
                            rescue_origin, ray_d, r_t, r_center, r_normal, r_radius,
                            all_texel_sites[r_idx], all_texel_rgb[r_idx],
                            all_texel_height[r_idx], num_texel_sites,
                        )
                        rgb += r_color * trans
                        log_t += -30.0
                    else:
                        walk_break_reason = int(3)
                    break

                nsigma = all_nsigmas[prim_idx]
                prim_normal = wp.vec3f(
                    float(nsigma[0]), float(nsigma[1]), float(nsigma[2])
                )
                sigma = float(nsigma[3])

                if not hit or t_near > t_far or sigma < 1e-3:
                    prim_idx = next_prim_idx
                    pt_near = wp.max(pt_near, pt_far)
                    continue

                _, _, height, _, t_surf, dp, color, _ = plane_intersection_fwd_local(
                    ray_o,
                    ray_d,
                    t_near,
                    center,
                    prim_normal,
                    radius,
                    all_texel_sites[prim_idx],
                    all_texel_rgb[prim_idx],
                    all_texel_height[prim_idx],
                    num_texel_sites,
                )
                t_far = wp.min(t_surf, t_far) if dp >= 0.0 else t_far
                t_near = wp.max(t_surf, t_near) if dp < 0.0 else t_near

                dt = t_far - t_near
                if not hit_captured and hit and dt > 0.0:
                    hit_captured = True
                    hit_point = ray_o + t_surf * ray_d
                    hit_normal = prim_normal
                    hit_radius = radius
                    hit_prim_idx = prim_idx

                prim_idx = next_prim_idx
                pt_near = wp.max(pt_near, pt_far)
                if hit and dt > 0.0:
                    delta_log_t = -sigma * dt
                    alpha = 1.0 - wp.exp(delta_log_t)

                    rgb += color * alpha * trans
                    log_t += delta_log_t
                    contrib_count += 1

            ray_trans = wp.exp(log_t)
            if obj_hit:
                obj_point = ray_o + obj_t * ray_d
                o_facing = wp.dot(ray_d, obj_normal)
                o_outward_n = obj_normal
                if o_facing > 0.0:
                    o_outward_n = -obj_normal

                o_reflect_dir = reflect_ray(ray_d, obj_normal)
                o_reflect_origin = obj_point + 1.0e-3 * obj_eps_scale * o_outward_n

                # Walk-continuation
                o_reflected_color = wp.vec3f(0.0, 0.0, 0.0)
                o_hit = bool(False)
                if obj_secondary_bvh == 1:
                    # BVH nearest-sphere reflection (find_valid_secondary_hit) instead of the walk
                    # continuation: the walk resumes from the primary ray's cell at the mirror and
                    # can hop past a THIN
                    o_hit, o_reflected_color = find_valid_secondary_hit(
                        bvh_id, o_reflect_origin, o_reflect_dir, int(-1),
                        all_spheres, all_nsigmas, all_texel_sites,
                        all_att_sites, all_att_values, all_att_temps, all_texel_height,
                    )
                    if not o_hit:
                        o_reflected_color = bkgd_color
                else:
                    o_reflected_color, o_hit = continue_walk_from_bounce(
                        bvh_id, prim_idx, pt_near, o_reflect_origin, o_reflect_dir,
                        False, ball_center, ball_radius,
                        obj_ior,
                        all_spheres, all_nsigmas, all_texel_sites,
                        all_att_sites, all_att_values, all_att_temps, all_texel_height,
                        adjacency, adjacency_offsets, adjacency_diff,
                        transmittance_threshold, bkgd_color,
                        guard_enabled, guard_bvh_id, guard_spheres, guard_nsigmas, guard_map,
                    )

                if obj_debug_hitmask != 0:
                    inserted_obj_color = wp.vec3f(1.0, 1.0, 1.0) if o_hit else wp.vec3f(0.0, 0.0, 0.0)
                elif obj_refraction_enabled != 0:
                    o_valid, o_refract_dir = refract_ray(ray_d, obj_normal, obj_ior)
                    o_refracted_color = o_reflected_color
                    if o_valid:
                        o_entry_origin = obj_point - 1.0e-3 * obj_eps_scale * o_outward_n
                        if ball_enabled != 0:
                            # Walk-continuation (author-adopted )
                            o_refracted_color, o_refract_walk_hit = continue_walk_from_bounce(
                                bvh_id, prim_idx, pt_near, o_entry_origin, o_refract_dir,
                                True, ball_center, ball_radius,
                                obj_ior,
                                all_spheres, all_nsigmas, all_texel_sites,
                                all_att_sites, all_att_values, all_att_temps, all_texel_height,
                                adjacency, adjacency_offsets, adjacency_diff,
                                transmittance_threshold, bkgd_color,
                                guard_enabled, guard_bvh_id, guard_spheres, guard_nsigmas, guard_map,
                            )
                        elif mirror_thickness > 0.0:
                            # Solid glass SLAB: entry->exit resolved analytically
                            o_exit_plane_point = obj_point - mirror_thickness * o_outward_n
                            o_exit_plane_normal = -o_outward_n
                            e_t, e_dp = ray_plane_intersect(
                                o_entry_origin, o_refract_dir, o_exit_plane_point, o_exit_plane_normal
                            )
                            if e_t > 0.0:
                                o_exit_point = o_entry_origin + e_t * o_refract_dir
                                o_exit_valid, o_exit_dir = refract_ray(
                                    o_refract_dir, o_exit_plane_normal, obj_ior
                                )
                                if o_exit_valid:
                                    o_refract_origin = o_exit_point + 1.0e-3 * obj_eps_scale * o_exit_plane_normal
                                    o_refracted_color, o_refract_walk_hit = continue_walk_from_bounce(
                                        bvh_id, prim_idx, pt_near, o_refract_origin, o_exit_dir,
                                        False, ball_center, ball_radius,
                                        obj_ior,
                                        all_spheres, all_nsigmas, all_texel_sites,
                                        all_att_sites, all_att_values, all_att_temps, all_texel_height,
                                        adjacency, adjacency_offsets, adjacency_diff,
                                        transmittance_threshold, bkgd_color,
                                        guard_enabled, guard_bvh_id, guard_spheres, guard_nsigmas, guard_map,
                                    )
                                else:
                                    # TIR off the slab's own back face
                                    o_refracted_color = o_reflected_color
                            # else: entry-refracted ray doesn't reach the back plane going forward
                        else:
                            # mirror_thickness == 0 (default): infinitely thin panel, single bend,
                            # no second surface
                            o_refracted_color, o_refract_walk_hit = continue_walk_from_bounce(
                                bvh_id, prim_idx, pt_near, o_entry_origin, o_refract_dir,
                                False, ball_center, ball_radius,
                                obj_ior,
                                all_spheres, all_nsigmas, all_texel_sites,
                                all_att_sites, all_att_values, all_att_temps, all_texel_height,
                                adjacency, adjacency_offsets, adjacency_diff,
                                transmittance_threshold, bkgd_color,
                                guard_enabled, guard_bvh_id, guard_spheres, guard_nsigmas, guard_map,
                            )
                    o_fresnel_r = fresnel_reflectance(ray_d, obj_normal, obj_ior)
                    inserted_obj_color = (
                        o_fresnel_r * o_reflected_color + (1.0 - o_fresnel_r) * o_refracted_color
                    )
                else:
                    inserted_obj_color = (
                        (1.0 - obj_reflectivity) * obj_base_color
                        + obj_reflectivity * o_reflected_color
                    )
                rgb += inserted_obj_color * ray_trans
            else:
                if obj_debug_hitmask != 0:
                    # Distinct from the white/black secondary-ray-hit coding above
                    rgb += wp.vec3f(1.0, 0.0, 1.0) * ray_trans
                else:
                    rgb += bkgd_color * ray_trans

            final_rgb = rgb

            if primary_debug_walk == 2:
                # Contribution-count heat map: how many primitives were actually composited on this
                # ray
                cc = float(contrib_count)
                t_norm = wp.min(cc / 40.0, 1.0)
                color_out[pix_i, pix_j] = wp.vec3f(t_norm, 1.0 - wp.abs(2.0 * t_norm - 1.0), 1.0 - t_norm)
                return

            if primary_debug_walk != 0:
                if walk_break_reason == 2:
                    color_out[pix_i, pix_j] = wp.vec3f(1.0, 0.5, 0.0)  # orange: hit MAX_WALK_ITERS
                elif walk_break_reason == 3:
                    color_out[pix_i, pix_j] = wp.vec3f(0.0, 1.0, 1.0)  # cyan: adjacency exhausted early
                else:
                    color_out[pix_i, pix_j] = final_rgb
                return

            if hit_captured and (reflectivity > 0.0 or refraction_enabled != 0):
                eps = 1.0e-3 * hit_radius
                facing = wp.dot(ray_d, hit_normal)
                outward_n = hit_normal
                if facing > 0.0:
                    outward_n = -hit_normal

                reflect_dir = reflect_ray(ray_d, hit_normal)
                reflect_origin = hit_point + eps * outward_n

                r_hit, r_color = find_valid_secondary_hit(
                    bvh_id, reflect_origin, reflect_dir, hit_prim_idx,
                    all_spheres, all_nsigmas, all_texel_sites,
                    all_att_sites, all_att_values, all_att_temps, all_texel_height,
                )
                reflected_color = r_color if r_hit else bkgd_color

                if refraction_enabled != 0:
                    valid, refract_dir = refract_ray(ray_d, hit_normal, ior)
                    refracted_color = reflected_color
                    if valid:
                        refract_origin = hit_point - eps * outward_n
                        f_hit, f_color = find_valid_secondary_hit(
                            bvh_id, refract_origin, refract_dir, hit_prim_idx,
                            all_spheres, all_nsigmas, all_texel_sites,
                            all_att_sites, all_att_values, all_att_temps, all_texel_height,
                        )
                        refracted_color = f_color if f_hit else bkgd_color
                    fresnel_r = fresnel_reflectance(ray_d, hit_normal, ior)
                    final_rgb = fresnel_r * reflected_color + (1.0 - fresnel_r) * refracted_color
                else:
                    final_rgb = (1.0 - reflectivity) * rgb + reflectivity * reflected_color

            color_out[pix_i, pix_j] = final_rgb

        self.raytrace_kernel = raytrace_kernel

    def render(
        self,
        camera,
        start_point_idx,
        bvh_id,
        points,
        radii,
        density,
        normals,
        texel_sites,
        texel_rgb,
        att_sites,
        att_values,
        att_temps,
        texel_height,
        adjacency,
        adjacency_offsets,
        adjacency_diff,
        transmittance_threshold=1e-3,
        reflectivity=0.0,
        ior=1.0,
        refraction_enabled=False,
        bkgd_color=(0.0, 0.0, 0.0),
        mirror_enabled=False,
        mirror_center=(0.0, 0.0, 0.0),
        mirror_normal=(0.0, 0.0, 1.0),
        mirror_tangent=(1.0, 0.0, 0.0),
        mirror_bitangent=(0.0, 1.0, 0.0),
        mirror_half_width=1.0,
        mirror_half_height=1.0,
        mirror_reflectivity=1.0,
        mirror_base_color=(0.05, 0.05, 0.05),
        mirror_ior=1.10,
        mirror_refraction_enabled=False,
        mirror_thickness=0.0,
        mirror_border_width=0.0,
        mirror_border_color=(0.12, 0.08, 0.05),
        ball_enabled=False,
        ball_center=(0.0, 0.0, 0.0),
        ball_radius=1.0,
        ball_reflectivity=1.0,
        ball_base_color=(0.05, 0.05, 0.05),
        ball_ior=1.10,
        ball_refraction_enabled=False,
        obj_debug_hitmask=False,
        obj_secondary_bvh=False,
        guard_indices=None,
        primary_debug_walk=False,
        frame_seed=0,
    ):
        with wp.ScopedDevice(str(self.device)):
            torch_stream = torch.cuda.current_stream()
            wp_stream = wp.stream_from_torch(torch_stream)
            wp.set_stream(wp_stream)

            tiles_h = 1 + (camera.height - 1) // TILE_WIDTH
            tiles_w = 1 + (camera.width - 1) // TILE_WIDTH
            total_tiles = tiles_h * tiles_w

            all_spheres = torch.cat([points, radii[:, None]], dim=-1).to(torch.float32)
            all_nsigmas = torch.cat([normals, density[:, None]], dim=-1).to(torch.float32)

            # skip-guard sub-scene : a BVH over just the flagged primitives, with their global
            # indices for colour lookup
            guard_enabled = guard_indices is not None and int(guard_indices.numel()) > 0
            if guard_enabled:
                g_idx = guard_indices.to(points.device).long()
                guard_spheres_t = all_spheres[g_idx].contiguous()
                guard_nsigmas_t = all_nsigmas[g_idx].contiguous()
                guard_map_t = g_idx.to(torch.int32).contiguous()
                from .bvh_ray_query import build_scene_bvh as _build_guard_bvh
                self._guard_bvh = _build_guard_bvh(points[g_idx].contiguous(), radii[g_idx].contiguous(), self.device)
                guard_bvh_id = self._guard_bvh.id
            else:
                guard_spheres_t = all_spheres[:1].contiguous()
                guard_nsigmas_t = all_nsigmas[:1].contiguous()
                guard_map_t = torch.zeros(1, dtype=torch.int32, device=points.device)
                guard_bvh_id = bvh_id
            guard_spheres_wp = wp.from_torch(guard_spheres_t, dtype=wp.vec4f, requires_grad=False)
            guard_nsigmas_wp = wp.from_torch(guard_nsigmas_t, dtype=wp.vec4f, requires_grad=False)
            guard_map_wp = wp.from_torch(guard_map_t, dtype=wp.int32, requires_grad=False)
            all_texel_sites = texel_sites.to(self.tscalar)
            all_texel_rgb = texel_rgb.to(self.tscalar)
            all_att_sites = att_sites.to(self.tscalar)
            all_att_values = att_values.to(self.tscalar)
            all_att_temps = att_temps.to(self.tscalar)
            all_texel_height = texel_height.to(self.tscalar)

            color_out = torch.zeros(
                (camera.height, camera.width, 3),
                dtype=torch.float32,
                device=self.device,
            )

            ray_trace_threads = total_tiles * TILE_SIZE
            wp.launch(
                self.raytrace_kernel,
                dim=ray_trace_threads,
                inputs=[
                    camera.to_warp(),
                    start_point_idx,
                    int(frame_seed),
                    bvh_id,
                    all_spheres.detach(),
                    all_nsigmas.detach(),
                    all_texel_sites.detach(),
                    all_texel_rgb.detach(),
                    all_att_sites.detach(),
                    all_att_values.detach(),
                    all_att_temps.detach(),
                    all_texel_height.detach(),
                    adjacency,
                    adjacency_offsets,
                    adjacency_diff,
                    transmittance_threshold,
                    float(reflectivity),
                    float(ior),
                    int(refraction_enabled),
                    wp.vec3f(*bkgd_color),
                    int(mirror_enabled),
                    _as_vec3f(mirror_center),
                    _as_vec3f(mirror_normal),
                    _as_vec3f(mirror_tangent),
                    _as_vec3f(mirror_bitangent),
                    float(mirror_half_width),
                    float(mirror_half_height),
                    float(mirror_reflectivity),
                    _as_vec3f(mirror_base_color),
                    float(mirror_ior),
                    int(mirror_refraction_enabled),
                    float(mirror_thickness),
                    float(mirror_border_width),
                    _as_vec3f(mirror_border_color),
                    int(ball_enabled),
                    _as_vec3f(ball_center),
                    float(ball_radius),
                    float(ball_reflectivity),
                    _as_vec3f(ball_base_color),
                    float(ball_ior),
                    int(ball_refraction_enabled),
                    int(obj_debug_hitmask),
                    int(obj_secondary_bvh),
                    int(guard_enabled),
                    guard_bvh_id,
                    guard_spheres_wp,
                    guard_nsigmas_wp,
                    guard_map_wp,
                    int(primary_debug_walk),
                    color_out,
                ],
                block_dim=TILE_SIZE,
            )

            return color_out
