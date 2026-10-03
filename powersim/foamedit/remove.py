"""Remove the selected primitives from a checkpoint and rebuild the adjacency.

    python -m powersim.foamedit.remove --checkpoint-config <ckpt>/config.yaml \
        --selection selection.pt --hops 5 --output-dir <new checkpoint dir>"""

import argparse
from pathlib import Path

import torch

from powersim.core.checkpoint import load_powerfoam_checkpoint
from powersim.foamedit._io import render_view, save_edited_checkpoint
from powersim.foamedit.edit import compose_scene, crop_primitives, expand_mask_k_hops


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-config", required=True)
    ap.add_argument("--selection", required=True, help="boolean mask (.pt) of the primitives to remove")
    ap.add_argument("--hops", type=int, default=0, help="grow the selection by this many distance-bounded hops before removing")
    ap.add_argument("--hop-radius-multiple", type=float, default=2.0)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--split", default="train")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    ck = load_powerfoam_checkpoint(a.checkpoint_config, device=a.device, split=a.split)
    scene = ck.scene
    camera = ck.data_handler.cameras[a.camera_index]
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[remove] {scene.points.shape[0]} primitives", flush=True)
    render_view(scene, camera, out / "before.png")

    mask = torch.load(a.selection, map_location=a.device).to(torch.bool)
    if mask.shape[0] != scene.points.shape[0]:
        raise ValueError(f"selection has {mask.shape[0]} entries, checkpoint has {scene.points.shape[0]} primitives")
    if a.hops > 0:
        before = int(mask.sum())
        mask = expand_mask_k_hops(scene, mask, a.hops, a.hop_radius_multiple)
        print(f"[remove] selection grown {a.hops} hops: {before} -> {int(mask.sum())}", flush=True)
    empty = crop_primitives(scene, torch.zeros(scene.points.shape[0], dtype=torch.bool, device=a.device))
    compose_scene(scene, empty, replace_mask=mask)
    print(f"[remove] removed {int(mask.sum())}; {scene.points.shape[0]} primitives remain", flush=True)

    render_view(scene, camera, out / "after.png")
    torch.save(mask.cpu(), out / "removed_mask.pt")
    save_edited_checkpoint(scene, ck.checkpoint_dir, out)
    print(f"[remove] wrote checkpoint + before/after renders to {out}", flush=True)


if __name__ == "__main__":
    main()
