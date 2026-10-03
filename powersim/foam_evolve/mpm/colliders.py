"""Surface collider with friction (BC type `powersim_surface_collider`); upstream's is sticky only."""

import warp as wp

from powersim.core.thirdparty_paths import ensure_physgaussian_on_path

POWERSIM_BC_TYPES = ("powersim_surface_collider",)


def split_powersim_bcs(bc_params):
    """-> (upstream_bcs, powersim_bcs): the BC dicts PhysGaussian understands vs. ours."""
    ours = [bc for bc in bc_params if bc["type"] in POWERSIM_BC_TYPES]
    upstream = [bc for bc in bc_params if bc["type"] not in POWERSIM_BC_TYPES]
    return upstream, ours


_kernel_cache = {}


def _friction_collide_kernel():
    """Built lazily: MPMStateStruct/MPMModelStruct/Dirichlet_collider only import once
    PhysGaussian's mpm_solver_warp package is on sys.path."""
    if "k" in _kernel_cache:
        return _kernel_cache["k"]
    ensure_physgaussian_on_path()
    from mpm_solver_warp.mpm_utils import MPMModelStruct, MPMStateStruct
    from mpm_solver_warp.warp_utils import Dirichlet_collider

    @wp.kernel
    def friction_collide(
        time: float,
        dt: float,
        state: MPMStateStruct,
        model: MPMModelStruct,
        param: Dirichlet_collider,
    ):
        grid_x, grid_y, grid_z = wp.tid()
        if time >= param.start_time and time < param.end_time:
            offset = wp.vec3(
                float(grid_x) * model.dx - param.point[0],
                float(grid_y) * model.dx - param.point[1],
                float(grid_z) * model.dx - param.point[2],
            )
            n = wp.vec3(param.normal[0], param.normal[1], param.normal[2])
            if wp.dot(offset, n) < 0.0:
                v = state.grid_v_out[grid_x, grid_y, grid_z]
                vn = wp.dot(v, n)
                if param.surface_type == 1:
                    v = v - vn * n
                else:
                    v = v - wp.min(vn, 0.0) * n
                if vn < 0.0 and wp.length(v) > 1e-20:
                    v = wp.max(0.0, wp.length(v) + vn * param.friction) * wp.normalize(v)
                state.grid_v_out[grid_x, grid_y, grid_z] = v

    _kernel_cache["k"] = friction_collide
    return friction_collide


def add_friction_surface_collider(
    mpm_solver, point, normal, surface="separate", friction=0.0, start_time=0.0, end_time=999.0
):
    ensure_physgaussian_on_path()
    from mpm_solver_warp.warp_utils import Dirichlet_collider

    nrm = 1.0 / float(sum(float(x) ** 2 for x in normal)) ** 0.5
    normal = [float(x) * nrm for x in normal]
    param = Dirichlet_collider()
    param.start_time = float(start_time)
    param.end_time = float(end_time)
    param.point = wp.vec3(float(point[0]), float(point[1]), float(point[2]))
    param.normal = wp.vec3(normal[0], normal[1], normal[2])
    param.surface_type = 1 if surface == "slip" else 2
    param.friction = float(friction)

    mpm_solver.collider_params.append(param)
    mpm_solver.grid_postprocess.append(_friction_collide_kernel())
    mpm_solver.modify_bc.append(None)


def apply_powersim_bcs(mpm_solver, powersim_bcs):
    for bc in powersim_bcs:
        if bc["type"] == "powersim_surface_collider":
            add_friction_surface_collider(
                mpm_solver,
                point=bc["point"],
                normal=bc["normal"],
                surface=bc.get("surface", "separate"),
                friction=bc.get("friction", 0.0),
                start_time=bc.get("start_time", 0.0),
                end_time=bc.get("end_time", 999.0),
            )
        else:
            raise TypeError(f"unknown powersim BC type {bc['type']!r}")
