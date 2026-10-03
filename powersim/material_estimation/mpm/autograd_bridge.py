"""torch.autograd bridge over the Warp MPM rollout, with a checkpointed variant for long rollouts."""

import torch
import warp as wp

from powersim.material_estimation.mpm.differentiable_mpm import (
    MPMModelStruct,
    MPMStateStruct,
    clone_model_with_grad,
    diff_state_from,
    run_rollout,
    copy_vec3_kernel,
    copy_mat33_kernel,
)


@wp.kernel
def _seed_output_grads_kernel(
    x: wp.array(dtype=wp.vec3),
    grad_x: wp.array(dtype=wp.vec3),
    F_trial: wp.array(dtype=wp.mat33),
    grad_F_trial: wp.array(dtype=wp.mat33),
    v: wp.array(dtype=wp.vec3),
    grad_v: wp.array(dtype=wp.vec3),
    C: wp.array(dtype=wp.mat33),
    grad_C: wp.array(dtype=wp.mat33),
    loss: wp.array(dtype=wp.float32),
):
    """loss = sum_p [dot(x_p, grad_x_p) + frobenius_dot(F_trial_p, grad_F_trial_p) + dot(v_p, grad_v_p)
    + frobenius_dot(C_p, grad_C_p)]."""
    p = wp.tid()
    contrib = wp.dot(x[p], grad_x[p]) + wp.dot(v[p], grad_v[p])
    Fm = F_trial[p]
    Gm = grad_F_trial[p]
    contrib += (
        Fm[0, 0] * Gm[0, 0] + Fm[0, 1] * Gm[0, 1] + Fm[0, 2] * Gm[0, 2]
        + Fm[1, 0] * Gm[1, 0] + Fm[1, 1] * Gm[1, 1] + Fm[1, 2] * Gm[1, 2]
        + Fm[2, 0] * Gm[2, 0] + Fm[2, 1] * Gm[2, 1] + Fm[2, 2] * Gm[2, 2]
    )
    Cm = C[p]
    Hm = grad_C[p]
    contrib += (
        Cm[0, 0] * Hm[0, 0] + Cm[0, 1] * Hm[0, 1] + Cm[0, 2] * Hm[0, 2]
        + Cm[1, 0] * Hm[1, 0] + Cm[1, 1] * Hm[1, 1] + Cm[1, 2] * Hm[1, 2]
        + Cm[2, 0] * Hm[2, 0] + Cm[2, 1] * Hm[2, 1] + Cm[2, 2] * Hm[2, 2]
    )
    wp.atomic_add(loss, 0, contrib)


class MPMRolloutFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x0: torch.Tensor,
        v0: torch.Tensor,
        F_trial0: torch.Tensor,
        C0: torch.Tensor,
        E: torch.Tensor,
        nu: torch.Tensor,
        base_state: MPMStateStruct,
        base_model: MPMModelStruct,
        num_substeps: int,
        dt: float,
        device: str,
        floor_point=None,
        floor_normal=None,
        damping_scale: float = 1.0,
        pin_point=None,
        pin_size=None,
        impulse_force: torch.Tensor = None,
        impulse_substeps: int = 0,
        floor_mode: int = 0,
    ):
        """Args: x0: (n, 3) initial particle positions, requires_grad as desired."""
        n = x0.shape[0]

        x0_wp = wp.from_torch(x0.detach().contiguous(), dtype=wp.vec3, requires_grad=True)
        v0_wp = wp.from_torch(v0.detach().contiguous(), dtype=wp.vec3, requires_grad=True)
        F_trial0_wp = wp.from_torch(F_trial0.detach().contiguous(), dtype=wp.mat33, requires_grad=True)
        C0_wp = wp.from_torch(C0.detach().contiguous(), dtype=wp.mat33, requires_grad=True)
        E_wp = wp.from_torch(E.detach().contiguous(), dtype=wp.float32, requires_grad=True)
        nu_wp = wp.from_torch(nu.detach().contiguous(), dtype=wp.float32, requires_grad=True)

        diff_model = clone_model_with_grad(base_model, E_wp, nu_wp, device)

        tape = wp.Tape()
        with tape:
            diff_state = diff_state_from(base_state, n, device)
            # x0/v0/F_trial0/C0 may differ from base_state's own particle_x/v/F_trial/C
            wp.launch(kernel=copy_vec3_kernel, dim=n, inputs=[x0_wp, diff_state.particle_x], device=device)
            wp.launch(kernel=copy_vec3_kernel, dim=n, inputs=[v0_wp, diff_state.particle_v], device=device)
            wp.launch(kernel=copy_mat33_kernel, dim=n, inputs=[F_trial0_wp, diff_state.particle_F_trial], device=device)
            wp.launch(kernel=copy_mat33_kernel, dim=n, inputs=[C0_wp, diff_state.particle_C], device=device)

            floor_point_wp = wp.vec3(*floor_point) if floor_point is not None else None
            floor_normal_wp = wp.vec3(*floor_normal) if floor_normal is not None else None
            pin_point_wp = wp.vec3(*pin_point) if pin_point is not None else None
            pin_size_wp = wp.vec3(*pin_size) if pin_size is not None else None
            impulse_force_wp = (
                wp.from_torch(impulse_force.detach().contiguous(), dtype=wp.vec3, requires_grad=False)
                if impulse_force is not None else None
            )
            final_state = run_rollout(
                diff_state, diff_model, n, num_substeps, dt, device,
                floor_point=floor_point_wp, floor_normal=floor_normal_wp, damping_scale=damping_scale,
                pin_point=pin_point_wp, pin_size=pin_size_wp,
                impulse_force=impulse_force_wp, impulse_substeps=impulse_substeps,
                floor_mode=floor_mode,
            )

        ctx.tape = tape
        ctx.final_state = final_state
        ctx.x0_wp = x0_wp
        ctx.v0_wp = v0_wp
        ctx.F_trial0_wp = F_trial0_wp
        ctx.C0_wp = C0_wp
        ctx.E_wp = E_wp
        ctx.nu_wp = nu_wp
        ctx.n = n
        ctx.device = device
        ctx.needs_x0_grad = x0.requires_grad
        ctx.needs_v0_grad = v0.requires_grad
        ctx.needs_F0_grad = F_trial0.requires_grad
        ctx.needs_C0_grad = C0.requires_grad
        ctx.needs_E_grad = E.requires_grad
        ctx.needs_nu_grad = nu.requires_grad

        x_final = wp.to_torch(final_state.particle_x).detach().clone()
        F_trial_final = wp.to_torch(final_state.particle_F_trial).detach().clone()
        v_final = wp.to_torch(final_state.particle_v).detach().clone()
        C_final = wp.to_torch(final_state.particle_C).detach().clone()
        return x_final, F_trial_final, v_final, C_final

    @staticmethod
    def backward(
        ctx,
        grad_x_final: torch.Tensor,
        grad_F_trial_final: torch.Tensor,
        grad_v_final: torch.Tensor,
        grad_C_final: torch.Tensor,
    ):
        device = ctx.device
        n = ctx.n
        tape = ctx.tape
        final_state = ctx.final_state

        grad_x_wp = wp.from_torch(grad_x_final.contiguous(), dtype=wp.vec3, requires_grad=False)
        grad_F_wp = wp.from_torch(grad_F_trial_final.contiguous(), dtype=wp.mat33, requires_grad=False)
        grad_v_wp = wp.from_torch(grad_v_final.contiguous(), dtype=wp.vec3, requires_grad=False)
        grad_C_wp = wp.from_torch(grad_C_final.contiguous(), dtype=wp.mat33, requires_grad=False)
        loss_wp = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

        with tape:
            wp.launch(
                kernel=_seed_output_grads_kernel,
                dim=n,
                inputs=[
                    final_state.particle_x, grad_x_wp,
                    final_state.particle_F_trial, grad_F_wp,
                    final_state.particle_v, grad_v_wp,
                    final_state.particle_C, grad_C_wp,
                    loss_wp,
                ],
                device=device,
            )

        tape.backward(loss=loss_wp)

        x0_grad = wp.to_torch(ctx.x0_wp.grad).detach().clone() if ctx.needs_x0_grad else None
        v0_grad = wp.to_torch(ctx.v0_wp.grad).detach().clone() if ctx.needs_v0_grad else None
        F0_grad = wp.to_torch(ctx.F_trial0_wp.grad).detach().clone() if ctx.needs_F0_grad else None
        C0_grad = wp.to_torch(ctx.C0_wp.grad).detach().clone() if ctx.needs_C0_grad else None
        E_grad = wp.to_torch(ctx.E_wp.grad).detach().clone() if ctx.needs_E_grad else None
        nu_grad = wp.to_torch(ctx.nu_wp.grad).detach().clone() if ctx.needs_nu_grad else None

        tape.zero()

        # One None per non-tensor forward() arg
        return (
            x0_grad, v0_grad, F0_grad, C0_grad, E_grad, nu_grad,
            None, None, None, None, None, None, None, None, None, None, None, None, None,
        )


class MPMCheckpointedRolloutFunction(torch.autograd.Function):
    """Gradient-checkpointed variant of MPMRolloutFunction."""

    @staticmethod
    def forward(
        ctx,
        x0: torch.Tensor,
        v0: torch.Tensor,
        F_trial0: torch.Tensor,
        C0: torch.Tensor,
        E: torch.Tensor,
        nu: torch.Tensor,
        base_state: MPMStateStruct,
        base_model: MPMModelStruct,
        num_substeps: int,
        dt: float,
        device: str,
        floor_point=None,
        floor_normal=None,
        damping_scale: float = 1.0,
        pin_point=None,
        pin_size=None,
        impulse_force: torch.Tensor = None,
        impulse_substeps: int = 0,
        floor_mode: int = 0,
    ):
        n = x0.shape[0]

        x0_wp = wp.from_torch(x0.detach().contiguous(), dtype=wp.vec3, requires_grad=True)
        v0_wp = wp.from_torch(v0.detach().contiguous(), dtype=wp.vec3, requires_grad=True)
        F_trial0_wp = wp.from_torch(F_trial0.detach().contiguous(), dtype=wp.mat33, requires_grad=True)
        C0_wp = wp.from_torch(C0.detach().contiguous(), dtype=wp.mat33, requires_grad=True)
        E_wp = wp.from_torch(E.detach().contiguous(), dtype=wp.float32, requires_grad=True)
        nu_wp = wp.from_torch(nu.detach().contiguous(), dtype=wp.float32, requires_grad=True)

        diff_model = clone_model_with_grad(base_model, E_wp, nu_wp, device)

        # Deliberately NO `with tape:` here
        diff_state = diff_state_from(base_state, n, device)
        wp.launch(kernel=copy_vec3_kernel, dim=n, inputs=[x0_wp, diff_state.particle_x], device=device)
        wp.launch(kernel=copy_vec3_kernel, dim=n, inputs=[v0_wp, diff_state.particle_v], device=device)
        wp.launch(kernel=copy_mat33_kernel, dim=n, inputs=[F_trial0_wp, diff_state.particle_F_trial], device=device)
        wp.launch(kernel=copy_mat33_kernel, dim=n, inputs=[C0_wp, diff_state.particle_C], device=device)

        floor_point_wp = wp.vec3(*floor_point) if floor_point is not None else None
        floor_normal_wp = wp.vec3(*floor_normal) if floor_normal is not None else None
        pin_point_wp = wp.vec3(*pin_point) if pin_point is not None else None
        pin_size_wp = wp.vec3(*pin_size) if pin_size is not None else None
        impulse_force_wp = (
            wp.from_torch(impulse_force.detach().contiguous(), dtype=wp.vec3, requires_grad=False)
            if impulse_force is not None else None
        )
        final_state = run_rollout(
            diff_state, diff_model, n, num_substeps, dt, device,
            floor_point=floor_point_wp, floor_normal=floor_normal_wp, damping_scale=damping_scale,
            pin_point=pin_point_wp, pin_size=pin_size_wp,
            impulse_force=impulse_force_wp, impulse_substeps=impulse_substeps,
            floor_mode=floor_mode,
        )

        # Save only the ORIGINAL torch tensors (cheap)
        ctx.save_for_backward(x0, v0, F_trial0, C0, E, nu)
        ctx.base_state = base_state
        ctx.base_model = base_model
        ctx.num_substeps = num_substeps
        ctx.dt = dt
        ctx.device = device
        ctx.floor_point = floor_point
        ctx.floor_normal = floor_normal
        ctx.damping_scale = damping_scale
        ctx.pin_point = pin_point
        ctx.pin_size = pin_size
        ctx.impulse_force = impulse_force
        ctx.impulse_substeps = impulse_substeps
        ctx.floor_mode = floor_mode
        ctx.n = n
        ctx.needs_x0_grad = x0.requires_grad
        ctx.needs_v0_grad = v0.requires_grad
        ctx.needs_F0_grad = F_trial0.requires_grad
        ctx.needs_C0_grad = C0.requires_grad
        ctx.needs_E_grad = E.requires_grad
        ctx.needs_nu_grad = nu.requires_grad

        x_final = wp.to_torch(final_state.particle_x).detach().clone()
        F_trial_final = wp.to_torch(final_state.particle_F_trial).detach().clone()
        v_final = wp.to_torch(final_state.particle_v).detach().clone()
        C_final = wp.to_torch(final_state.particle_C).detach().clone()
        return x_final, F_trial_final, v_final, C_final

    @staticmethod
    def backward(
        ctx,
        grad_x_final: torch.Tensor,
        grad_F_trial_final: torch.Tensor,
        grad_v_final: torch.Tensor,
        grad_C_final: torch.Tensor,
    ):
        x0, v0, F_trial0, C0, E, nu = ctx.saved_tensors
        device = ctx.device
        n = ctx.n

        # Rebuild fresh, grad-tracked warp arrays and re-run the IDENTICAL rollout, this time
        # wrapped in a real Tape
        x0_wp = wp.from_torch(x0.detach().contiguous(), dtype=wp.vec3, requires_grad=True)
        v0_wp = wp.from_torch(v0.detach().contiguous(), dtype=wp.vec3, requires_grad=True)
        F_trial0_wp = wp.from_torch(F_trial0.detach().contiguous(), dtype=wp.mat33, requires_grad=True)
        C0_wp = wp.from_torch(C0.detach().contiguous(), dtype=wp.mat33, requires_grad=True)
        E_wp = wp.from_torch(E.detach().contiguous(), dtype=wp.float32, requires_grad=True)
        nu_wp = wp.from_torch(nu.detach().contiguous(), dtype=wp.float32, requires_grad=True)

        diff_model = clone_model_with_grad(ctx.base_model, E_wp, nu_wp, device)

        tape = wp.Tape()
        with tape:
            diff_state = diff_state_from(ctx.base_state, n, device)
            wp.launch(kernel=copy_vec3_kernel, dim=n, inputs=[x0_wp, diff_state.particle_x], device=device)
            wp.launch(kernel=copy_vec3_kernel, dim=n, inputs=[v0_wp, diff_state.particle_v], device=device)
            wp.launch(kernel=copy_mat33_kernel, dim=n, inputs=[F_trial0_wp, diff_state.particle_F_trial], device=device)
            wp.launch(kernel=copy_mat33_kernel, dim=n, inputs=[C0_wp, diff_state.particle_C], device=device)

            floor_point_wp = wp.vec3(*ctx.floor_point) if ctx.floor_point is not None else None
            floor_normal_wp = wp.vec3(*ctx.floor_normal) if ctx.floor_normal is not None else None
            pin_point_wp = wp.vec3(*ctx.pin_point) if ctx.pin_point is not None else None
            pin_size_wp = wp.vec3(*ctx.pin_size) if ctx.pin_size is not None else None
            impulse_force_wp = (
                wp.from_torch(ctx.impulse_force.detach().contiguous(), dtype=wp.vec3, requires_grad=False)
                if ctx.impulse_force is not None else None
            )
            final_state = run_rollout(
                diff_state, diff_model, n, ctx.num_substeps, ctx.dt, device,
                floor_point=floor_point_wp, floor_normal=floor_normal_wp, damping_scale=ctx.damping_scale,
                pin_point=pin_point_wp, pin_size=pin_size_wp,
                impulse_force=impulse_force_wp, impulse_substeps=ctx.impulse_substeps,
                floor_mode=ctx.floor_mode,
            )

            grad_x_wp = wp.from_torch(grad_x_final.contiguous(), dtype=wp.vec3, requires_grad=False)
            grad_F_wp = wp.from_torch(grad_F_trial_final.contiguous(), dtype=wp.mat33, requires_grad=False)
            grad_v_wp = wp.from_torch(grad_v_final.contiguous(), dtype=wp.vec3, requires_grad=False)
            grad_C_wp = wp.from_torch(grad_C_final.contiguous(), dtype=wp.mat33, requires_grad=False)
            loss_wp = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

            wp.launch(
                kernel=_seed_output_grads_kernel,
                dim=n,
                inputs=[
                    final_state.particle_x, grad_x_wp,
                    final_state.particle_F_trial, grad_F_wp,
                    final_state.particle_v, grad_v_wp,
                    final_state.particle_C, grad_C_wp,
                    loss_wp,
                ],
                device=device,
            )

        tape.backward(loss=loss_wp)

        x0_grad = wp.to_torch(x0_wp.grad).detach().clone() if ctx.needs_x0_grad else None
        v0_grad = wp.to_torch(v0_wp.grad).detach().clone() if ctx.needs_v0_grad else None
        F0_grad = wp.to_torch(F_trial0_wp.grad).detach().clone() if ctx.needs_F0_grad else None
        C0_grad = wp.to_torch(C0_wp.grad).detach().clone() if ctx.needs_C0_grad else None
        E_grad = wp.to_torch(E_wp.grad).detach().clone() if ctx.needs_E_grad else None
        nu_grad = wp.to_torch(nu_wp.grad).detach().clone() if ctx.needs_nu_grad else None

        tape.zero()

        return (
            x0_grad, v0_grad, F0_grad, C0_grad, E_grad, nu_grad,
            None, None, None, None, None, None, None, None, None, None, None, None, None,
        )


def run_checkpointed_rollout(
    x0: torch.Tensor,
    v0: torch.Tensor,
    F_trial0: torch.Tensor,
    C0: torch.Tensor,
    E: torch.Tensor,
    nu: torch.Tensor,
    base_state: MPMStateStruct,
    base_model: MPMModelStruct,
    num_substeps_total: int,
    checkpoint_chunk_size: int,
    dt: float,
    device: str,
    floor_point=None,
    floor_normal=None,
    damping_scale: float = 1.0,
    pin_point=None,
    pin_size=None,
    impulse_force: torch.Tensor = None,
    impulse_substeps: int = 0,
    floor_mode: int = 0,
):
    """Chains ceil(num_substeps_total / checkpoint_chunk_size) calls to MPMCheckpointedRolloutFunction,
    each covering at most checkpoint_chunk_size substeps, threading state between them WITHOUT
    detaching."""
    x, v, F_trial, C = x0, v0, F_trial0, C0
    remaining = num_substeps_total
    # The impulse may last longer than one chunk (garden-bonsai poke: num_dt=1000 vs chunk 150);
    # carry the unapplied remainder into subsequent chunks so the total impulse duration
    imp_remaining = impulse_substeps if impulse_force is not None else 0
    while remaining > 0:
        step = min(checkpoint_chunk_size, remaining)
        this_imp = min(imp_remaining, step)
        x, F_trial, v, C = MPMCheckpointedRolloutFunction.apply(
            x, v, F_trial, C, E, nu, base_state, base_model, step, dt, device,
            floor_point, floor_normal, damping_scale, pin_point, pin_size,
            impulse_force if this_imp > 0 else None, this_imp,
            floor_mode,
        )
        remaining -= step
        imp_remaining -= this_imp
    return x, F_trial, v, C
