"""Windowed rollout, loss, and the two optimisation stages (initial velocity, then E field)."""

import torch

from powersim.material_estimation.lr_schedule import get_linear_schedule_with_warmup

from powersim.core import transform

from powersim.material_estimation.mpm.autograd_bridge import MPMRolloutFunction, run_checkpointed_rollout
from powersim.material_estimation.render.differentiable_render import (
    render_rgb_differentiable,
    compute_deformed_quaternions_radii,
)
from powersim.material_estimation.losses import photometric_loss


def simulate_window(
    scene, camera, base_state, base_model,
    x_start, v_start, F_trial_start, C_start,
    n_orig, base_quaternions, base_radii, transform_info,
    E, nu, num_substeps, dt, device,
    floor_point=None, floor_normal=None, checkpoint_chunk_size=None, damping_scale=1.0,
    pin_point=None, pin_size=None, impulse_force=None, impulse_substeps=0,
    render_indices=None, floor_mode=0,
):
    """One differentiable window: advances MPM num_substeps from the given starting state, renders the
    end-of-window frame."""
    if checkpoint_chunk_size is None:
        x_final, F_trial_final, v_final, C_final = MPMRolloutFunction.apply(
            x_start, v_start, F_trial_start, C_start, E, nu,
            base_state, base_model, num_substeps, dt, device,
            floor_point, floor_normal, damping_scale,
            pin_point, pin_size, impulse_force, impulse_substeps, floor_mode,
        )
    else:
        x_final, F_trial_final, v_final, C_final = run_checkpointed_rollout(
            x_start, v_start, F_trial_start, C_start, E, nu,
            base_state, base_model, num_substeps, checkpoint_chunk_size, dt, device,
            floor_point, floor_normal, damping_scale,
            pin_point, pin_size, impulse_force, impulse_substeps, floor_mode,
        )

    scale_factor, original_mean_pos = transform_info
    x_surface_world = transform.undotransform2origin(
        transform.undoshift2center111(x_final[:n_orig]), scale_factor, original_mean_pos
    )
    F_surface = F_trial_final[:n_orig]
    new_quaternions, new_radii = compute_deformed_quaternions_radii(F_surface, base_quaternions, base_radii)
    rgb = render_rgb_differentiable(
        scene, camera, x_surface_world, new_radii, new_quaternions, indices=render_indices
    )

    return rgb, x_final, v_final, F_trial_final, C_final


def windowed_rollout_and_loss(
    scene, camera, base_state, base_model, x0, v0, n_orig, base_quaternions, base_radii,
    transform_info, get_E, nu, gt_frames, substeps_per_window, dt, device, ssim_weight=0.2,
    floor_point=None, floor_normal=None, checkpoint_chunk_size=None, damping_scale=1.0,
    pin_point=None, pin_size=None, impulse_force=None, impulse_substeps=0,
    render_indices=None,
):
    """Runs one full windowed-BPTT pass over len(gt_frames) windows starting from (x0, v0, F_trial=I,
    C=0), calling loss.backward() once per window and returning the per-window losses and rendered
    frames for logging."""
    n = x0.shape[0]
    x = x0
    v = v0
    F_trial = torch.eye(3, device=device, dtype=torch.float32).expand(n, 3, 3).contiguous()
    C = torch.zeros_like(F_trial)

    per_window_loss = []
    rendered_frames = []
    for window_idx, gt_frame in enumerate(gt_frames):
        E = get_E()
        this_impulse_force = impulse_force if window_idx == 0 else None
        this_impulse_substeps = impulse_substeps if window_idx == 0 else 0
        rgb, x_final, v_final, F_trial_final, C_final = simulate_window(
            scene, camera, base_state, base_model, x, v, F_trial, C,
            n_orig, base_quaternions, base_radii, transform_info,
            E, nu, substeps_per_window, dt, device,
            floor_point, floor_normal, checkpoint_chunk_size, damping_scale,
            pin_point, pin_size, this_impulse_force, this_impulse_substeps,
            render_indices,
        )

        loss = photometric_loss(rgb, gt_frame, ssim_weight=ssim_weight) / len(gt_frames)
        loss.backward()

        per_window_loss.append(loss.item() * len(gt_frames))  # undo the averaging, for logging
        rendered_frames.append(rgb.detach())

        x, v, F_trial, C = x_final.detach(), v_final.detach(), F_trial_final.detach(), C_final.detach()

    return per_window_loss, rendered_frames


def optimize_field_E(
    scene, camera, base_state, base_model, x0, v0, n_orig, base_quaternions, base_radii,
    transform_info, gt_frames, nu_fixed, substeps_per_window, dt, device,
    material_field, n_iters=25, lr=0.01, ssim_weight=0.2, smoothness_weight=1e-4, log_fn=print,
    floor_point=None, floor_normal=None, checkpoint_chunk_size=None, damping_scale=1.0,
    pin_point=None, pin_size=None, impulse_force=None, impulse_substeps=0,
    window_schedule=None, on_iteration=None, weight_decay=1e-4, warmup_step=None,
    max_grad_norm=1.0, render_indices=None,
):
    """Optimizes a spatially-varying E via a caller-supplied
    powersim.material_estimation.fields.material_field."""
    optimizer = torch.optim.AdamW(material_field.parameters(), lr=lr, weight_decay=weight_decay)
    if warmup_step is None:
        warmup_step = max(1, int(0.1 * n_iters))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=n_iters)

    def get_E():
        return material_field(x0)

    history = []
    for it in range(n_iters):
        if window_schedule is not None:
            window_count = int(round(window_schedule.compute_state(it)[0]))
            window_count = max(1, min(window_count, len(gt_frames)))
        else:
            window_count = len(gt_frames)
        active_gt_frames = gt_frames[:window_count]

        optimizer.zero_grad()
        per_window_loss, _ = windowed_rollout_and_loss(
            scene, camera, base_state, base_model, x0, v0, n_orig, base_quaternions, base_radii,
            transform_info, get_E, nu_fixed, active_gt_frames, substeps_per_window, dt, device, ssim_weight,
            floor_point, floor_normal, checkpoint_chunk_size, damping_scale,
            pin_point, pin_size, impulse_force, impulse_substeps, render_indices,
        )
        smoothness_loss = material_field.compute_smoothness_loss() * smoothness_weight
        smoothness_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            material_field.parameters(), max_grad_norm, error_if_nonfinite=False
        ).item()
        optimizer.step()
        scheduler.step()

        total_loss = sum(per_window_loss) + smoothness_loss.item()
        history.append((it, total_loss, window_count))
        if log_fn is not None:
            with torch.no_grad():
                E_sample = material_field(x0)
            per_window_str = ", ".join(f"{l:.3e}" for l in per_window_loss)
            log_fn(f"iter {it:3d} (windows={window_count:2d}, lr={scheduler.get_last_lr()[0]:.2e}, "
                   f"grad_norm={grad_norm:.3e}): "
                   f"total_loss={total_loss:.6e} "
                   f"E[min={E_sample.min().item():.4e} max={E_sample.max().item():.4e} "
                   f"mean={E_sample.mean().item():.4e}] smoothness={smoothness_loss.item():.4e} "
                   f"per_window=[{per_window_str}]")

        if on_iteration is not None:
            on_iteration(it, material_field)

    return material_field, history


def optimize_velocity_only(
    scene, camera, base_state, base_model, x0, n_orig, base_quaternions, base_radii,
    transform_info, gt_frames, nu_fixed, E_fixed, substeps_per_window, dt, device,
    velocity_field, n_iters=25, lr=0.01, ssim_weight=0.2,
    velocity_smoothness_weight=1e-4, log_fn=print,
    floor_point=None, floor_normal=None, checkpoint_chunk_size=None, damping_scale=1.0,
    pin_point=None, pin_size=None,
    on_iteration=None, weight_decay=1e-4, warmup_step=None,
    max_grad_norm=1.0, render_indices=None,
):
    """Stage 1 of PhysDreamer's own two-stage optimization, per their paper: "we randomly initialize
    the Young's modulus for each Gaussian particle and freeze it."""
    optimizer = torch.optim.AdamW(velocity_field.parameters(), lr=lr, weight_decay=weight_decay)
    if warmup_step is None:
        warmup_step = max(1, int(0.1 * n_iters))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=n_iters)

    def get_E():
        return E_fixed

    history = []
    for it in range(n_iters):
        optimizer.zero_grad()
        v0 = velocity_field(x0)
        per_window_loss, _ = windowed_rollout_and_loss(
            scene, camera, base_state, base_model, x0, v0, n_orig, base_quaternions, base_radii,
            transform_info, get_E, nu_fixed, gt_frames, substeps_per_window, dt, device, ssim_weight,
            floor_point, floor_normal, checkpoint_chunk_size, damping_scale,
            pin_point, pin_size, None, 0, render_indices,
        )
        smoothness_loss = velocity_field.compute_smoothness_loss() * velocity_smoothness_weight
        smoothness_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            velocity_field.parameters(), max_grad_norm, error_if_nonfinite=False
        ).item()
        optimizer.step()
        scheduler.step()

        total_loss = sum(per_window_loss) + smoothness_loss.item()
        history.append((it, total_loss, len(gt_frames)))
        if log_fn is not None:
            with torch.no_grad():
                v_speed = velocity_field(x0).norm(dim=-1)
            per_window_str = ", ".join(f"{l:.3e}" for l in per_window_loss)
            log_fn(f"[stage1] iter {it:3d} (windows={len(gt_frames)}, lr={scheduler.get_last_lr()[0]:.2e}, "
                   f"grad_norm={grad_norm:.3e}): total_loss={total_loss:.6e} "
                   f"|v|[max={v_speed.max().item():.4e} mean={v_speed.mean().item():.4e}] "
                   f"smoothness={smoothness_loss.item():.4e} per_window=[{per_window_str}]")

        if on_iteration is not None:
            on_iteration(it, velocity_field)

    return velocity_field, history
