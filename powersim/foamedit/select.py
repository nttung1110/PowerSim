"""Select an object's primitives from 2D masks by render-weighted voting (paper section 4.3).

One backward pass of the rasterizer per view gives every primitive's compositing weight inside
and outside the mask; a primitive is kept when inside > beta * outside. Unseen primitives take
the majority label of their neighbours.

    python -m powersim.foamedit.select --checkpoint-config <ckpt>/config.yaml \
        --masks masks/vase masks/frond --output selection.pt

Each --masks directory is one object class and holds one binary image per training view, named
after the view's image."""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from powersim.core.checkpoint import load_powerfoam_checkpoint
from powersim.foamedit.edit import expand_mask_k_hops, extend_selection, floor_distance_fn, mad_inlier_mask


class FeatureRenderer:
    """Composite an arbitrary per-primitive 3-vector through the rasterizer in place of colour."""

    def __init__(self, scene):
        self.scene = scene
        with torch.no_grad():
            self.normals = scene.get_normals()
            tangents, bitangent = scene.get_tangents()
            self.radii = scene.get_radii()
            offsets = scene.texel_sites * self.radii[:, None, None]
            offsets = offsets[..., 0:1] * tangents[:, None, :] + offsets[..., 1:2] * bitangent[:, None, :]
            self.texel_sites = scene.points[:, None, :] + offsets
            self.texel_height = scene.texel_height * self.radii[:, None]
            self.density = scene.get_density()
            self.points = scene.points.detach()
        self.num_texels = scene.args.num_texel_sites
        self.dtype = self.texel_height.dtype

    def _forward(self, camera, feat):
        s = self.scene
        texel_rgb = feat.to(self.dtype)[:, None, :].expand(-1, self.num_texels, -1)
        return s.rasterizer.forward(
            camera, None, self.points, self.radii, self.density, self.normals,
            self.texel_sites, texel_rgb.contiguous() if not feat.requires_grad else texel_rgb, self.texel_height,
            s.adjacency, s.adjacency_offsets, None, False,
        )

    def composite(self, camera, feat):
        """feat (N, 3) -> rendered feature image (H, W, 3) and opacity (H, W)"""
        with torch.no_grad():
            out = self._forward(camera, feat)
        return out[0].float(), out[1].float().reshape(camera.height, camera.width)

    def weighted_votes(self, camera, target):
        """target (H, W, 3): per-pixel indicator of up to 3 classes."""
        n = self.points.shape[0]
        feat = torch.zeros(n, 3, device=self.points.device, dtype=self.dtype, requires_grad=True)
        out = self._forward(camera, feat)
        (out[0].float() * target).sum().backward()
        return feat.grad.float()


def _find_mask_file(directory: Path, stem: str):
    for ext in (".png", ".PNG", ".jpg", ".JPG", ".jpeg"):
        p = directory / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def load_class_map(mask_dirs, class_map_dir, stem, W, H, device):
    """Per-pixel class ids (H, W) for one view, or None when the view has no mask."""
    if class_map_dir is not None:
        p = _find_mask_file(Path(class_map_dir), stem)
        if p is None:
            return None
        m = Image.open(p).convert("L")
        if m.size != (W, H):
            m = m.resize((W, H), Image.NEAREST)
        return torch.from_numpy(np.array(m)).to(device).long()
    cls = None
    for k, d in enumerate(mask_dirs, start=1):
        p = _find_mask_file(Path(d), stem)
        if p is None:
            continue
        m = Image.open(p).convert("L")
        if m.size != (W, H):
            m = m.resize((W, H), Image.NEAREST)
        inside = torch.from_numpy(np.array(m) > 128).to(device)
        if cls is None:
            cls = torch.zeros(H, W, dtype=torch.long, device=device)
        cls[inside] = k
    return cls


def vote(scene, renderer, cameras, names, mask_dirs, class_map_dir, device, bg_weight=0.5,
         fill_unseen=True, fill_iters=5, max_views=None, verify_views=0, verify_dir=None):
    """Render-weighted vote over the views that have masks -> (labels, conf, totals, voted view names)"""
    n = renderer.points.shape[0]
    num_classes = 256 if class_map_dir is not None else len(mask_dirs) + 1
    counts = torch.zeros(n, num_classes, dtype=torch.float32, device=device)
    voted = []
    for i, name in enumerate(names):
        if max_views is not None and len(voted) >= max_views:
            break
        cam = cameras[i].to_device(device)
        W, H = cam.width, cam.height
        cls_map = load_class_map(mask_dirs, class_map_dir, os.path.splitext(name)[0], W, H, device)
        if cls_map is None:
            continue
        present = cls_map.unique().tolist()
        for g in range(0, len(present), 3):
            grp = present[g:g + 3]
            target = torch.zeros(H, W, 3, device=device)
            for j, c in enumerate(grp):
                target[..., j] = (cls_map == c).float()
            votes = renderer.weighted_votes(cam, target)
            counts[:, grp] += votes[:, :len(grp)]
        voted.append((i, name, cls_map))
        if len(voted) % 20 == 0:
            print(f"[select] voted {len(voted)} views", flush=True)
    if not voted:
        raise RuntimeError("no view had a mask file; check --masks / --class-maps and the image names")

    totals = counts.sum(1)
    scores = counts.clone()
    scores[:, 0] *= bg_weight  # s+ > beta * s-  <=>  arg-max with the background vote discounted
    labels = scores.argmax(1)
    conf = scores.max(1).values / scores.sum(1).clamp(min=1e-6)
    seen = totals > 0
    labels[~seen] = 0
    n_unseen = int((~seen).sum())
    if fill_unseen and n_unseen:
        adj, off = scene.adjacency.long(), scene.adjacency_offsets.long()
        src = torch.repeat_interleave(torch.arange(n, device=device), off[1:] - off[:-1])
        for _ in range(fill_iters):
            unseen = (~seen).nonzero(as_tuple=True)[0]
            if unseen.numel() == 0:
                break
            local = torch.full((n,), -1, dtype=torch.long, device=device)
            local[unseen] = torch.arange(unseen.numel(), device=device)
            em = (~seen[src]) & seen[adj]
            cnt = torch.zeros(unseen.numel(), num_classes, dtype=torch.int32, device=device)
            cnt.index_put_((local[src[em]], labels[adj[em]]), torch.ones(int(em.sum()), dtype=torch.int32, device=device), accumulate=True)
            has = cnt.sum(1) > 0
            if not has.any():
                break
            labels[unseen[has]] = cnt.argmax(1)[has]
            seen[unseen[has]] = True
        print(f"[select] {n_unseen - int((~seen).sum())}/{n_unseen} never-seen primitives labelled from their neighbours", flush=True)
    print(f"[select] {len(voted)} views voted; classes present: {labels.unique().tolist()}", flush=True)

    if verify_dir is not None and verify_views > 0:
        verify_dir = Path(verify_dir)
        verify_dir.mkdir(parents=True, exist_ok=True)
        for i, name, cls_map in voted[:verify_views]:
            cam = cameras[i].to_device(device)
            feat = torch.zeros(n, 3, device=device)
            feat[:, 0] = (labels != 0).float()
            color, _ = renderer.composite(cam, feat)
            pred = (color[..., 0].clamp(0, 1) * 255).byte().cpu().numpy()
            mask = ((cls_map != 0).float() * 255).byte().cpu().numpy()
            stem = os.path.splitext(name)[0]
            Image.fromarray(np.concatenate([mask, pred], axis=1)).save(verify_dir / f"{stem}_mask_vs_selected.png")
        print(f"[select] verification renders (input mask | selected primitives) in {verify_dir}", flush=True)
    return labels, conf, totals, [name for _, name, _ in voted]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-config", required=True)
    ap.add_argument("--masks", nargs="*", default=[], help="one directory per object class (class ids 1, 2, ... in this order)")
    ap.add_argument("--class-maps", default=None, help="directory of per-view class-id maps instead of --masks")
    ap.add_argument("--split", default="train", help="camera split the masks belong to")
    ap.add_argument("--bg-weight", type=float, default=0.5, help="beta: discount on the outside/background vote")
    ap.add_argument("--no-fill-unseen", action="store_true", help="do not label unobserved primitives from their neighbours")
    ap.add_argument("--max-views", type=int, default=None)
    ap.add_argument("--classes", type=int, nargs="*", default=None, help="class ids that form the output selection (default: all non-zero)")
    ap.add_argument("--drop-outliers", action="store_true",
                    help="drop selected primitives whose distance from the selection's centroid is a MAD outlier (stray far-away votes)")
    ap.add_argument("--outlier-k", type=float, default=6.0)
    ap.add_argument("--extend-hops", type=int, default=0,
                    help="grow the selection by this many power-diagram adjacency hops, keeping only extras within --extend-max-distance of it and above the floor of --extend-frame")
    ap.add_argument("--extend-max-distance", type=float, default=0.1)
    ap.add_argument("--extend-frame", default=None, help="the sim config the selection will be simulated with (its surface_collider is the floor, in MPM space) or a frame JSON (world space)")
    ap.add_argument("--extend-floor-margin", type=float, default=0.05)
    ap.add_argument("--dilate-hops", type=int, default=0, help="grow the selection by this many distance-bounded hops")
    ap.add_argument("--dilate-radius-multiple", type=float, default=2.0)
    ap.add_argument("--output", required=True, help="boolean mask (.pt) over the checkpoint's primitives")
    ap.add_argument("--labels-out", default=None, help="save {labels, conf, votes} (.pt)")
    ap.add_argument("--verify-dir", default=None)
    ap.add_argument("--verify-views", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    if not a.masks and a.class_maps is None:
        ap.error("give --masks DIR [DIR ...] or --class-maps DIR")

    ck = load_powerfoam_checkpoint(a.checkpoint_config, device=a.device, split=a.split)
    scene = ck.scene
    names = getattr(ck.data_handler, "image_names", None)
    if names is None:
        raise RuntimeError("the checkpoint's cameras carry no image names; re-export cameras_<split>.pt with "
                           "scripts/prep/export_cameras.py (or load the dataset) so masks can be matched to views")
    renderer = FeatureRenderer(scene)
    labels, conf, totals, voted = vote(
        scene, renderer, ck.data_handler.cameras, names, a.masks, a.class_maps, a.device,
        bg_weight=a.bg_weight, fill_unseen=not a.no_fill_unseen, max_views=a.max_views,
        verify_views=a.verify_views, verify_dir=a.verify_dir,
    )
    classes = a.classes if a.classes else [c for c in labels.unique().tolist() if c != 0]
    selection = torch.isin(labels, torch.tensor(classes, device=labels.device))
    print(f"[select] classes {classes}: {int(selection.sum())}/{selection.shape[0]} primitives", flush=True)
    if a.drop_outliers:
        idx = selection.nonzero(as_tuple=True)[0]
        keep = mad_inlier_mask(scene.points.data[idx], a.outlier_k)
        selection[idx[~keep]] = False
        print(f"[select] dropped {int((~keep).sum())} stray primitives (MAD k={a.outlier_k}); {int(selection.sum())} remain", flush=True)
    if a.extend_hops > 0:
        before = int(selection.sum())
        unext = Path(a.output).with_name(Path(a.output).stem + "_unextended.pt")
        unext.parent.mkdir(parents=True, exist_ok=True)
        torch.save(selection.cpu(), unext)
        print(f"[select] unextended selection kept at {unext} (use it as the simulator's --recenter-selection)", flush=True)
        fd = floor_distance_fn(a.extend_frame, scene.points.data[selection]) if a.extend_frame else None
        selection = extend_selection(scene, selection, a.extend_hops, a.extend_max_distance, fd, a.extend_floor_margin)
        print(f"[select] extended {a.extend_hops} adjacency hop(s) (filtered): {before} -> {int(selection.sum())}", flush=True)
    if a.dilate_hops > 0:
        before = int(selection.sum())
        selection = expand_mask_k_hops(scene, selection, a.dilate_hops, a.dilate_radius_multiple)
        print(f"[select] dilated {a.dilate_hops} hops: {before} -> {int(selection.sum())}", flush=True)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(selection.cpu(), a.output)
    print(f"[select] wrote {a.output}", flush=True)
    if a.labels_out:
        torch.save({"labels": labels.cpu(), "conf": conf.cpu(), "votes": totals.cpu(), "views": voted,
                    "classes": {k: str(d) for k, d in enumerate(a.masks, start=1)}, "bg_weight": a.bg_weight}, a.labels_out)
        print(f"[select] wrote {a.labels_out}", flush=True)


if __name__ == "__main__":
    main()
