"""Ray intersection, reflection, refraction and Fresnel helpers (forward only)."""

import warp as wp


@wp.func
def ray_sphere_intersect(eye: wp.vec3f, dir: wp.vec3f, c: wp.vec3f, r: float):
    oc = eye - c
    qb = 2.0 * wp.dot(oc, dir)
    qc = wp.dot(oc, oc) - r * r
    discriminant = qb * qb - 4.0 * qc

    if discriminant < 0:
        return False, 0.0, 0.0

    t_far = (-qb + wp.sqrt(discriminant)) / 2.0
    t_near = (-qb - wp.sqrt(discriminant)) / 2.0

    if t_near < 0.0 and t_far < 0.0:
        return False, 0.0, 0.0
    elif t_near < 0.0:
        return True, 0.0, t_far
    else:
        return True, t_near, t_far


@wp.func
def ray_pface_intersect_diff(
    eye: wp.vec3f,
    dir: wp.vec3f,
    diff: wp.vec3f,
    pm_diff: float,
):
    # Power face equation with precomputed difference: face_n = diff, face_offset = pm_diff
    dp = wp.dot(dir, diff)
    t = (pm_diff - wp.dot(eye, diff)) / dp
    return t, dp


@wp.func
def ray_plane_intersect(
    eye: wp.vec3f, dir: wp.vec3f, p: wp.vec3f, n: wp.vec3f, h: wp.float32 = 0.0
):
    # Plane equation: n . x = n . p + h
    dp = wp.dot(n, dir)
    t = (wp.dot(p - eye, n) + h) / dp
    return t, dp


@wp.func
def ray_finite_plane_intersect(
    ray_o: wp.vec3f,
    ray_d: wp.vec3f,
    center: wp.vec3f,
    normal: wp.vec3f,
    tangent: wp.vec3f,
    bitangent: wp.vec3f,
    half_width: float,
    half_height: float,
):
    """`ray_plane_intersect` bounded to a finite rectangle."""
    t, dp = ray_plane_intersect(ray_o, ray_d, center, normal)
    if wp.abs(dp) < 1.0e-8 or t <= 0.0:
        return False, float(0.0)

    hit_pt = ray_o + t * ray_d
    local = hit_pt - center
    u = wp.dot(local, tangent)
    v = wp.dot(local, bitangent)
    if wp.abs(u) > half_width or wp.abs(v) > half_height:
        return False, float(0.0)

    return True, t


@wp.func
def reflect_ray(d: wp.vec3f, n: wp.vec3f) -> wp.vec3f:
    """Mirror-reflect unit direction `d` off a surface with unit normal `n`."""
    return d - 2.0 * wp.dot(d, n) * n


@wp.func
def refract_ray(d: wp.vec3f, n: wp.vec3f, ior: float):
    """Snell's-law refraction of unit direction `d` through a surface with unit normal `n` and material
    refractive index `ior` (the medium on the other side of `n` from `d`'s origin is assumed
    vacuum/air, IOR 1.0)"""
    nn = n
    eta = 1.0 / ior
    cos_i = wp.dot(d, n)
    if cos_i > 0.0:
        # Exiting the material into air: flip the normal to face against d, invert eta.
        nn = -n
        cos_i = -cos_i
        eta = ior

    cos_theta = wp.min(-cos_i, 1.0)
    r_out_perp = eta * (d + cos_theta * nn)
    k = 1.0 - wp.length_sq(r_out_perp)
    if k < 0.0:
        return False, wp.vec3f(0.0, 0.0, 0.0)

    r_out_parallel = -wp.sqrt(k) * nn
    return True, r_out_perp + r_out_parallel


@wp.func
def fresnel_reflectance(d: wp.vec3f, n: wp.vec3f, ior: float) -> float:
    """Schlick's approximation to the Fresnel reflectance at incidence angle
    acos(|dot(d,n)|), for a surface separating vacuum/air (IOR 1.0) from `ior`."""
    r0 = (1.0 - ior) / (1.0 + ior)
    r0 = r0 * r0
    cos_theta = wp.abs(wp.dot(d, n))
    return r0 + (1.0 - r0) * wp.pow(1.0 - cos_theta, 5.0)
