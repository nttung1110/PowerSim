"""Insert a captured object (a whole checkpoint) into another scene, optionally replacing a selection.

The object is aligned by a similarity transform: source up onto target up, a uniform scale, and
a translation onto the target's floor. Frames are JSON files {"up", "floor_point"}.

    python -m powersim.foamedit.insert --target-config <scene>/config.yaml --source-config <object>/config.yaml \
        --target-frame config/frames/<scene>.json --source-frame config/frames/<object>.json \
        --replace-selection selection.pt --output-dir <new checkpoint dir>"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from powersim.core.checkpoint import load_powerfoam_checkpoint
from powersim.foamedit._io import render_view, save_edited_checkpoint
from powersim.foamedit.edit import (
    bbox_diag,
    compose_scene,
    crop_adjacency,
    crop_primitives,
    expand_mask_k_hops,
    load_frame,
    mad_inlier_mask,
    neutralize_insert_radii_near_boundary,
    remap_adjacency,
    remove_outliers,
    rotation_aligning_vectors,
    transform_cropped_primitives,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-config", required=True)
    ap.add_argument("--source-config", required=True)
    ap.add_argument("--target-frame", required=True)
    ap.add_argument("--source-frame", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--replace-selection", default=None, help="boolean mask over the target's primitives to replace")
    ap.add_argument("--mask-hops", type=int, default=0, help="grow the replaced selection by this many distance-bounded hops")
    ap.add_argument("--hop-radius-multiple", type=float, default=2.0)
    ap.add_argument("--anchor-world", type=float, nargs=3, default=None, help="placement anchor when not replacing")
    ap.add_argument("--scale", type=float, default=None, help="isotropic scale (overrides --scale-multiple)")
    ap.add_argument("--scale-multiple", type=float, default=1.0, help="object bbox diagonal = this x the replaced selection's")
    ap.add_argument("--yaw-degrees", type=float, default=0.0, help="extra rotation about the target's up after alignment")
    ap.add_argument("--density-scale", type=float, default=None, help="default 1/scale (keeps density x radius constant)")
    ap.add_argument("--remove-outliers", action="store_true", help="drop source primitives far from the centroid (MAD test)")
    ap.add_argument("--no-preserve-adjacency", action="store_true",
                    help="rebuild one adjacency over the merged set instead of keeping the object's own neighbour graph")
    ap.add_argument("--neutralize-radii", action="store_true",
                    help="with --no-preserve-adjacency: shrink inserted primitives whose radius dwarfs nearby target ones")
    ap.add_argument("--neutralize-max-ratio", type=float, default=3.0)
    ap.add_argument("--neutralize-search-multiple", type=float, default=4.0)
    ap.add_argument("--carry-mask", nargs="*", default=[], metavar="NAME=MASK.pt",
                    help="masks over the source's primitives to re-index onto the composite")
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--split", default="train")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    if a.neutralize_radii:
        a.no_preserve_adjacency = True
    if a.replace_selection is None and a.anchor_world is None:
        ap.error("give --replace-selection (replacement) or --anchor-world (insertion)")
    if a.replace_selection is None and a.scale is None:
        ap.error("--anchor-world placement needs --scale")
    device = a.device

    target = load_powerfoam_checkpoint(a.target_config, device=device, split=a.split)
    source = load_powerfoam_checkpoint(a.source_config, device=device, split="train")
    t_scene, s_scene = target.scene, source.scene
    camera = target.data_handler.cameras[a.camera_index]
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[insert] target {t_scene.points.shape[0]} primitives, source {s_scene.points.shape[0]}", flush=True)
    render_view(t_scene, camera, out / "before.png")

    # ---- what is replaced (if anything) ----
    replace_mask, base_points = None, None
    if a.replace_selection is not None:
        replace_mask = torch.load(a.replace_selection, map_location=device).to(torch.bool)
        # scale / placement anchor = the UNgrown selection, minus stray far-away primitives (a few
        # strays would otherwise dominate its bbox diagonal and centroid)
        base_points = t_scene.points.data[replace_mask]
        inlier = mad_inlier_mask(base_points)
        print(f"[insert] placement statistics from {int(inlier.sum())}/{base_points.shape[0]} selected primitives (MAD inliers)", flush=True)
        base_points = base_points[inlier]
        if a.mask_hops > 0:
            before = int(replace_mask.sum())
            replace_mask = expand_mask_k_hops(t_scene, replace_mask, a.mask_hops, a.hop_radius_multiple)
            print(f"[insert] replaced selection grown {a.mask_hops} hops: {before} -> {int(replace_mask.sum())}", flush=True)

    # ---- crop the whole source, keep its own adjacency ----
    whole = torch.ones(s_scene.points.shape[0], dtype=torch.bool, device=device)
    cropped = crop_primitives(s_scene, whole)
    ins_adj, ins_off = crop_adjacency(s_scene, whole)
    keep_src = torch.ones(len(cropped), dtype=torch.bool, device=device)
    if a.remove_outliers:
        cropped, keep_src = remove_outliers(cropped, return_mask=True)
        ins_adj, ins_off = remap_adjacency(ins_adj, ins_off, keep_src)
        print(f"[insert] outlier removal kept {int(keep_src.sum())}/{keep_src.shape[0]} source primitives", flush=True)

    # ---- similarity transform ----
    t_up, t_floor = load_frame(a.target_frame)
    s_up, _ = load_frame(a.source_frame)
    R = rotation_aligning_vectors(s_up, t_up)
    if a.yaw_degrees != 0.0:
        R = Rotation.from_rotvec(t_up * np.radians(a.yaw_degrees)).as_matrix() @ R
    crop_center = cropped.points.mean(dim=0)
    source_diag = bbox_diag(cropped.points)
    if a.scale is not None:
        scale = a.scale
    else:
        scale = a.scale_multiple * bbox_diag(base_points) / source_diag
    centered = (cropped.points - crop_center).cpu().numpy()
    rotated_scaled = (centered @ R.T) * scale
    heights = rotated_scaled @ t_up
    if base_points is not None:
        anchor = base_points.mean(dim=0).cpu().numpy()
        horizontal_anchor = anchor - np.dot(anchor, t_up) * t_up
        floor_height = np.dot(t_floor, t_up)
    else:
        anchor = np.array(a.anchor_world, dtype=np.float64)
        horizontal_anchor = anchor - np.dot(anchor, t_up) * t_up
        floor_height = np.dot(anchor, t_up)
    translation = torch.tensor(horizontal_anchor + (floor_height - heights.min()) * t_up, dtype=cropped.points.dtype, device=device)
    transformed = transform_cropped_primitives(cropped, R=R, scale=scale, translation=translation, center=crop_center)
    density_scale = a.density_scale if a.density_scale is not None else 1.0 / scale
    transformed.density = transformed.density * density_scale
    print(f"[insert] scale {scale:.4f} (source diag {source_diag:.3f}), yaw {a.yaw_degrees} deg, density x{density_scale:.3f}, "
          f"translation {translation.cpu().numpy().round(4).tolist()}", flush=True)
    if a.neutralize_radii:
        transformed = neutralize_insert_radii_near_boundary(
            t_scene, transformed, max_ratio=a.neutralize_max_ratio, search_radius_multiple=a.neutralize_search_multiple)

    # ---- compose ----
    n_kept = int((~replace_mask).sum()) if replace_mask is not None else t_scene.points.shape[0]
    compose_scene(t_scene, transformed, replace_mask=replace_mask,
                  preserve_source_adjacency=None if a.no_preserve_adjacency else (ins_adj, ins_off))
    n_ins = len(transformed)
    print(f"[insert] composite: {t_scene.points.shape[0]} primitives ({n_kept} kept + {n_ins} inserted)", flush=True)
    render_view(t_scene, camera, out / "after.png")

    # ---- masks over the composite ----
    inserted = torch.zeros(t_scene.points.shape[0], dtype=torch.bool)
    inserted[n_kept:] = True
    torch.save(inserted, out / "inserted_mask.pt")
    for spec in a.carry_mask:
        name, path = spec.split("=", 1)
        m = torch.load(path, map_location="cpu").to(torch.bool)
        if m.shape[0] != keep_src.shape[0]:
            raise ValueError(f"--carry-mask {name}: {m.shape[0]} entries but the source has {keep_src.shape[0]} primitives")
        comp = torch.cat([torch.zeros(n_kept, dtype=torch.bool), m[keep_src.cpu()]])
        torch.save(comp, out / f"{name}.pt")
        print(f"[insert] carried mask {name}: {int(comp.sum())} primitives -> {out / (name + '.pt')}", flush=True)
    with open(out / "placement.json", "w") as f:
        json.dump({"source": str(a.source_config), "target": str(a.target_config), "scale": float(scale),
                   "rotation": np.asarray(R).tolist(), "translation": translation.cpu().numpy().tolist(),
                   "center": crop_center.cpu().numpy().tolist(), "density_scale": float(density_scale),
                   "n_kept": n_kept, "n_inserted": n_ins, "replaced": int(replace_mask.sum()) if replace_mask is not None else 0,
                   "preserve_source_adjacency": not a.no_preserve_adjacency}, f, indent=2)
    save_edited_checkpoint(t_scene, target.checkpoint_dir, out)
    print(f"[insert] wrote checkpoint, masks and before/after renders to {out}", flush=True)


if __name__ == "__main__":
    main()
