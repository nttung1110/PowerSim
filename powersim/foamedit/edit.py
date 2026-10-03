"""Crop, transform and compose PowerFoam primitives across scenes; grow selections; placement frames."""

import json
from dataclasses import dataclass, fields

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation

from powersim.core.primitive_state import inverse_softplus_beta100

# Per-primitive attribute names on PowerfoamScene, in the order crop/compose operate on them
PRIMITIVE_ATTRS = (
    "points",
    "radii",
    "quaternions",
    "density",
    "texel_sites",
    "texel_sv_axis",
    "texel_sv_rgb",
    "texel_height",
)


@dataclass
class CroppedPrimitives:
    points: torch.Tensor
    radii: torch.Tensor
    quaternions: torch.Tensor
    density: torch.Tensor
    texel_sites: torch.Tensor
    texel_sv_axis: torch.Tensor
    texel_sv_rgb: torch.Tensor
    texel_height: torch.Tensor

    def __len__(self):
        return self.points.shape[0]


def crop_primitives(scene, mask: torch.Tensor) -> CroppedPrimitives:
    """Extract a boolean-masked subset of scene's primitives as detached, cloned tensors."""
    mask = mask.to(scene.points.device).bool()
    return CroppedPrimitives(**{
        attr: getattr(scene, attr).data[mask].clone() for attr in PRIMITIVE_ATTRS
    })


def mad_inlier_mask(points: torch.Tensor, k: float = 6.0) -> torch.Tensor:
    """True for points whose distance from the centroid is within median + k * MAD."""
    dist = torch.norm(points - points.mean(dim=0), dim=1)
    median = torch.median(dist)
    mad = torch.median(torch.abs(dist - median)) + 1e-6
    return dist < median + k * mad


def remove_outliers(cropped: CroppedPrimitives, k: float = 6.0, return_mask: bool = False):
    """Drop primitives whose distance from the crop's own centroid is a MAD-based outlier."""
    keep_mask = mad_inlier_mask(cropped.points, k)
    print(f"remove_outliers: keeping {keep_mask.sum().item()}/{keep_mask.shape[0]} primitives")
    result = CroppedPrimitives(**{
        attr: getattr(cropped, attr)[keep_mask] for attr in PRIMITIVE_ATTRS
    })
    if return_mask:
        return result, keep_mask
    return result


def expand_mask_by_distance(scene, mask: torch.Tensor, radius_multiple: float = 2.0) -> torch.Tensor:
    """Grow a selection mask by one physical "hop": include any not-yet-selected primitive within
    `radius_multiple * that seed primitive's own radius` of an already-selected primitive's
    position."""
    from scipy.spatial import cKDTree

    mask = mask.bool()
    points = scene.points.data.detach().cpu().numpy()
    radii = scene.get_radii().detach().cpu().numpy()
    seed_idx = mask.nonzero(as_tuple=True)[0].cpu().numpy()
    if seed_idx.size == 0:
        return mask.clone()

    tree = cKDTree(points)
    hits = tree.query_ball_point(points[seed_idx], r=radius_multiple * radii[seed_idx])

    expanded = mask.clone()
    for hit_list in hits:
        if hit_list:
            expanded[torch.tensor(hit_list, dtype=torch.long, device=mask.device)] = True
    return expanded


def expand_mask_k_hops(scene, mask: torch.Tensor, hops: int, radius_multiple: float = 2.0) -> torch.Tensor:
    """expand_mask_by_distance, applied `hops` times."""
    for _ in range(hops):
        mask = expand_mask_by_distance(scene, mask, radius_multiple=radius_multiple)
    return mask


def remap_adjacency(adjacency: torch.Tensor, adjacency_offsets: torch.Tensor, keep_mask: torch.Tensor):
    """Restrict an existing CSR adjacency graph to a boolean keep_mask over its OWN current index
    space, renumbering kept indices to a compacted 0..keep_mask.sum()-1 range."""
    device = adjacency.device
    if bool(keep_mask.all()):
        # Nothing dropped (append-only compose_scene): the graph is already the answer
        return adjacency.to(torch.int32), adjacency_offsets.to(torch.int32)
    n_kept = int(keep_mask.sum().item())
    old_to_new = torch.full((keep_mask.shape[0],), -1, dtype=torch.long, device=device)
    old_to_new[keep_mask] = torch.arange(n_kept, device=device)

    seed_idx = keep_mask.nonzero(as_tuple=True)[0]  # ascending, as boolean indexing orders them
    offsets_long = adjacency_offsets.to(torch.long)
    adjacency_long = adjacency.to(torch.long)

    if seed_idx.numel() == 0:
        return (
            torch.zeros(0, dtype=torch.int32, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
        )

    starts = offsets_long[seed_idx]
    counts = offsets_long[seed_idx + 1] - starts
    total = int(counts.sum().item())
    if total == 0:
        return (
            torch.zeros(0, dtype=torch.int32, device=device),
            torch.zeros(n_kept + 1, dtype=torch.int32, device=device),
        )

    rel = torch.arange(total, device=device) - starts.repeat_interleave(counts)
    flat_offsets = starts.repeat_interleave(counts) + rel
    raw_neighbors = adjacency_long[flat_offsets]
    remapped_neighbors = old_to_new[raw_neighbors]

    valid = remapped_neighbors >= 0
    # seed_of_entry is non-decreasing (built via repeat_interleave over ascending seed order), and
    # boolean-filtering by `valid` preserves relative order
    seed_of_entry = torch.arange(n_kept, device=device).repeat_interleave(counts)
    kept_neighbors = remapped_neighbors[valid]
    kept_seed = seed_of_entry[valid]

    new_counts = torch.zeros(n_kept, dtype=torch.long, device=device)
    new_counts.scatter_add_(0, kept_seed, torch.ones_like(kept_seed))
    new_offsets = torch.zeros(n_kept + 1, dtype=torch.long, device=device)
    new_offsets[1:] = torch.cumsum(new_counts, dim=0)

    return kept_neighbors.to(torch.int32), new_offsets.to(torch.int32)


def crop_adjacency(scene, mask: torch.Tensor):
    """remap_adjacency against `scene`'s own current adjacency/adjacency_offsets, for the same mask
    crop_primitives(scene, mask) would use."""
    mask = mask.to(scene.points.device).bool()
    return remap_adjacency(scene.adjacency, scene.adjacency_offsets, mask)


def rotation_aligning_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix R (applied as row @ R.T, matching powersim.core.transform's
    convention) that rotates unit vector a onto unit vector b, via Rodrigues' formula."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-8:
        if c > 0:
            return np.eye(3)
        # a and b are anti-parallel: rotate 180 degrees about any axis perpendicular to a.
        perp = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp)
        axis = axis / np.linalg.norm(axis)
        return Rotation.from_rotvec(axis * np.pi).as_matrix()
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))


def rotate_quaternions(quats_wxyz: torch.Tensor, R: np.ndarray) -> torch.Tensor:
    """Compose each (w,x,y,z) quaternion with world-space rotation R, so that the primitive's normal
    (used for spherical-Voronoi shading) rotates correctly under R."""
    q_np = quats_wxyz.detach().cpu().numpy()
    xyzw = q_np[:, [1, 2, 3, 0]]
    rot_old = Rotation.from_quat(xyzw)
    rot_delta = Rotation.from_matrix(R.T)
    rot_new = rot_old * rot_delta
    new_xyzw = rot_new.as_quat()
    new_wxyz = new_xyzw[:, [3, 0, 1, 2]]
    return torch.tensor(new_wxyz, dtype=quats_wxyz.dtype, device=quats_wxyz.device)


def rotate_vectors(v: torch.Tensor, R: np.ndarray) -> torch.Tensor:
    """Rotate the trailing 3-vector of an arbitrarily-shaped tensor by R (world-space rotation)"""
    shape = v.shape
    v_np = v.detach().cpu().numpy().reshape(-1, 3)
    rotated = v_np @ R.T
    return torch.tensor(rotated.reshape(shape), dtype=v.dtype, device=v.device)


def transform_cropped_primitives(
    cropped: CroppedPrimitives,
    R: np.ndarray,
    scale: float,
    translation: torch.Tensor,
    center: torch.Tensor,
) -> CroppedPrimitives:
    """Recenter on `center`, rotate by R, scale isotropically, then translate by `translation`."""
    device = cropped.points.device
    R_t = torch.tensor(R, dtype=cropped.points.dtype, device=device)

    centered = cropped.points - center.to(device)
    rotated = centered @ R_t.T
    new_points = rotated * scale + translation.to(device)

    new_radii = cropped.radii * scale
    new_quaternions = rotate_quaternions(cropped.quaternions, R)
    new_texel_sv_axis = rotate_vectors(cropped.texel_sv_axis, R)

    return CroppedPrimitives(
        points=new_points,
        radii=new_radii,
        quaternions=new_quaternions,
        density=cropped.density,
        texel_sites=cropped.texel_sites,
        texel_sv_axis=new_texel_sv_axis,
        texel_sv_rgb=cropped.texel_sv_rgb,
        texel_height=cropped.texel_height,
    )


def neutralize_insert_radii_near_boundary(
    target_scene,
    insert: CroppedPrimitives,
    max_ratio: float = 3.0,
    search_radius_multiple: float = 4.0,
) -> CroppedPrimitives:
    """Boost the effective radius of `insert` primitives that sit close to a much larger target_scene
    primitive, so a subsequent *real* target_scene.rebuild_adjacency() over the full merged point
    set."""
    from scipy.spatial import cKDTree

    device = insert.points.device
    target_points = target_scene.points.data.detach().cpu().numpy()
    target_radii_eff = target_scene.get_radii().detach().cpu().numpy()

    insert_points = insert.points.detach().cpu().numpy()
    insert_radii_eff = torch.nn.functional.softplus(insert.radii.detach(), beta=100.0).cpu().numpy()

    insert_median_radius = float(np.median(insert_radii_eff))
    danger_threshold = max_ratio * insert_median_radius
    large_idx = np.nonzero(target_radii_eff > danger_threshold)[0]

    boosted_eff = insert_radii_eff.copy()
    n_boosted = 0
    if large_idx.size > 0:
        insert_tree = cKDTree(insert_points)
        large_centers = target_points[large_idx]
        large_radii = target_radii_eff[large_idx]
        search_r = search_radius_multiple * large_radii
        hits = insert_tree.query_ball_point(large_centers, r=search_r)

        for j, hit_list in enumerate(hits):
            if not hit_list:
                continue
            required = large_radii[j] / max_ratio
            for i in hit_list:
                if required > boosted_eff[i]:
                    if boosted_eff[i] == insert_radii_eff[i]:
                        n_boosted += 1
                    boosted_eff[i] = required

    print(f"neutralize_insert_radii_near_boundary: {large_idx.size} oversized target primitives "
          f"found (radius > {danger_threshold:.5f} = max_ratio * insert median radius "
          f"{insert_median_radius:.5f}); boosted {n_boosted}/{len(insert)} insert primitives "
          f"(max_ratio={max_ratio}, search_radius_multiple={search_radius_multiple})")

    boosted_eff_t = torch.tensor(boosted_eff, dtype=insert.radii.dtype, device=device)
    new_raw_radii = inverse_softplus_beta100(boosted_eff_t)

    return CroppedPrimitives(
        points=insert.points,
        radii=new_raw_radii,
        quaternions=insert.quaternions,
        density=insert.density,
        texel_sites=insert.texel_sites,
        texel_sv_axis=insert.texel_sv_axis,
        texel_sv_rgb=insert.texel_sv_rgb,
        texel_height=insert.texel_height,
    )


def compose_scene(
    target_scene,
    insert: CroppedPrimitives,
    replace_mask: torch.Tensor = None,
    preserve_source_adjacency=None,
) -> None:
    """Splice `insert`'s primitives into target_scene in place."""
    if target_scene.args.num_texel_sites != insert.texel_sites.shape[1]:
        raise ValueError(
            f"num_texel_sites mismatch: target has {target_scene.args.num_texel_sites}, "
            f"insert has {insert.texel_sites.shape[1]}"
        )

    device = target_scene.points.device
    if replace_mask is not None:
        replace_mask = replace_mask.to(device).bool()
        keep_mask = ~replace_mask
    else:
        keep_mask = torch.ones(target_scene.points.shape[0], dtype=torch.bool, device=device)

    if preserve_source_adjacency is not None:
        target_adjacency, target_offsets = remap_adjacency(
            target_scene.adjacency, target_scene.adjacency_offsets, keep_mask
        )

    with torch.no_grad():
        for attr in PRIMITIVE_ATTRS:
            kept = getattr(target_scene, attr).data[keep_mask]
            added = getattr(insert, attr).to(device)
            new_tensor = torch.cat([kept, added], dim=0)
            setattr(target_scene, attr, nn.Parameter(new_tensor))

    if preserve_source_adjacency is not None:
        insert_adjacency, insert_offsets = preserve_source_adjacency
        insert_adjacency = insert_adjacency.to(device=device, dtype=torch.int32)
        insert_offsets = insert_offsets.to(device=device, dtype=torch.int32)

        n_target_kept = int(keep_mask.sum().item())
        shifted_insert_adjacency = insert_adjacency + n_target_kept
        final_adjacency = torch.cat([target_adjacency, shifted_insert_adjacency])
        final_offsets = torch.cat([
            target_offsets,
            insert_offsets[1:] + target_offsets[-1],
        ])
        target_scene.adjacency = final_adjacency
        target_scene.adjacency_offsets = final_offsets
    else:
        target_scene.rebuild_adjacency()


def expand_mask_adjacency_hops(scene, mask: torch.Tensor, hops: int) -> torch.Tensor:
    """Grow a selection by `hops` rings of power-diagram neighbours (scene.adjacency, CSR)"""
    adj, off = scene.adjacency.long(), scene.adjacency_offsets.long()
    n = mask.shape[0]
    src = torch.repeat_interleave(torch.arange(n, device=mask.device), off[1:] - off[:-1])
    out = mask.clone()
    for _ in range(hops):
        grown = out.clone()
        grown[adj[out[src]]] = True
        out = grown
    return out


def extend_selection(scene, mask: torch.Tensor, hops: int = 1, max_distance: float = 0.1,
                     floor_distance=None, floor_margin: float = 0.05) -> torch.Tensor:
    """Grow a selection by adjacency hops but keep only extras that (a) lie within `max_distance`
    (world units) of the original selection and."""
    grown = expand_mask_adjacency_hops(scene, mask, hops)
    extras = grown & ~mask
    pts = scene.points.data
    if extras.any():
        from scipy.spatial import cKDTree

        tree = cKDTree(pts[mask].detach().cpu().numpy())
        d, _ = tree.query(pts[extras].detach().cpu().numpy(), k=1)
        keep = torch.from_numpy(d <= max_distance).to(mask.device)
        if floor_distance is not None:
            keep &= floor_distance(pts[extras]) >= floor_margin
        idx = extras.nonzero(as_tuple=True)[0]
        extras[idx[~keep]] = False
    return mask | extras


def floor_distance_fn(frame_or_sim_config, reference_points: torch.Tensor):
    """Signed distance above the floor plane of a frame JSON (world space) or of a sim config's
    surface_collider."""
    cfg = load_sim_config(frame_or_sim_config)
    if "boundary_conditions" in cfg:
        from powersim.core import transform

        point, normal = collider_point_normal(frame_or_sim_config)
        _, scale_factor, mean = transform.transform2origin(reference_points, cfg.get("scale", 1.0))
        n = torch.as_tensor(normal, dtype=reference_points.dtype, device=reference_points.device)
        p = torch.as_tensor(point, dtype=reference_points.dtype, device=reference_points.device)
        return lambda x: (transform.shift2center111((x - mean) * scale_factor) - p) @ n
    up, floor_point = load_frame(frame_or_sim_config)
    n = torch.as_tensor(up, dtype=reference_points.dtype, device=reference_points.device)
    p = torch.as_tensor(floor_point, dtype=reference_points.dtype, device=reference_points.device)
    return lambda x: (x - p) @ n


# ---------------------------------------------------------------------------------------------
# Placement geometry (frames)
# ---------------------------------------------------------------------------------------------
def load_sim_config(sim_config_path) -> dict:
    with open(sim_config_path) as f:
        return json.load(f)


def collider_point_normal(sim_config_path):
    """First surface_collider boundary condition's (point, unit normal), as (np.ndarray, np.ndarray)"""
    cfg = load_sim_config(sim_config_path)
    for bc in cfg["boundary_conditions"]:
        if bc["type"] == "surface_collider":
            point = np.array(bc["point"], dtype=np.float64)
            normal = np.array(bc["normal"], dtype=np.float64)
            return point, normal / np.linalg.norm(normal)
    raise ValueError(f"no surface_collider boundary condition in {sim_config_path}")


def bbox_diag(points: torch.Tensor) -> float:
    mins = points.min(dim=0).values
    maxs = points.max(dim=0).values
    return float((maxs - mins).norm().item())


def load_frame(path):
    """(up, floor_point) of a scene, as unit np.float64 vectors."""
    cfg = load_sim_config(path)
    if "up" in cfg:
        up = np.array(cfg["up"], dtype=np.float64)
        return up / np.linalg.norm(up), np.array(cfg["floor_point"], dtype=np.float64)
    point, normal = collider_point_normal(path)
    return normal, point
