"""Export a checkpoint's dataset cameras to cameras_<split>.pt so it loads without training images.

    python scripts/prep/export_cameras.py --powerfoam-root third_party/powerfoam \
        --config <ckpt>/config.yaml --out-dir data/powerfoam_ckpt/<name> --splits train test"""
import argparse
import os
import sys
from pathlib import Path

import torch


def split_image_names(args, split):
    """Image file names of a split, in the loader's camera order (None if unavailable)"""
    import json
    import os
    data_dir = os.path.join(args.data_path, args.scene)
    try:
        if args.dataset == "colmap":
            import pycolmap
            rec = pycolmap.Reconstruction()
            rec.read(os.path.join(data_dir, "sparse/0/"))
            names = sorted(str(im.name) for im in rec.images.values())
            if split in ("train", "test"):  # data_loader/colmap.py: every 8th image is the test split
                names = [nm for i, nm in enumerate(names) if (i % 8 == 0) == (split == "test")]
            return names
        if args.dataset == "blender":
            with open(os.path.join(data_dir, f"transforms_{split}.json")) as f:
                return [os.path.basename(fr["file_path"]) + ".png" for fr in json.load(f)["frames"]]
    except Exception as e:  # names are a convenience; never fail the export over them
        print(f"[export_cameras] image names unavailable ({type(e).__name__}: {e})")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--powerfoam-root", required=True, help="PowerFoam checkout whose data/ holds the datasets")
    ap.add_argument("--config", required=True, help="checkpoint config.yaml")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--verify", type=int, default=5, help="number of views whose ray maps are checked")
    a = ap.parse_args()

    root = Path(a.powerfoam_root).resolve()
    sys.path.insert(0, str(root))
    os.chdir(root)  # config.yaml's data_path is relative to the PowerFoam checkout
    import configargparse
    from configs import Params, add_group
    from data_loader import DataHandler

    parser = configargparse.ArgParser()
    get_params = add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True)
    args = get_params(parser.parse_args(["-c", str(Path(a.config).resolve())]))
    out_dir = Path(a.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in a.splits:
        dh = DataHandler(args)
        try:
            dh.reload(split, downsample=args.downsample[-1])
        except Exception as e:  # e.g. a dataset without a test split
            print(f"[export_cameras] {split}: SKIP ({type(e).__name__}: {e})")
            continue
        image_names = split_image_names(args, split)
        cams = dh.cameras
        poses = dh.c2ws.clone()
        n = len(cams)
        W, H = dh.img_wh
        eyes = torch.stack([c.eye for c in cams]).clone()
        rights = torch.stack([c.right for c in cams]).clone()
        ups = torch.stack([c.up for c in cams]).clone()
        cam_ray_dirs = None
        if cams[0].ray_maps is not None:
            R0 = poses[0, :3, :3]
            world0 = cams[0].ray_maps.reshape(-1, 6)
            cam_ray_dirs = (world0[:, 3:] @ R0).contiguous()  # world = cam @ R^T  =>  cam = world @ R
            for i in torch.linspace(0, n - 1, min(a.verify, n)).round().long().tolist():
                Ri = poses[i, :3, :3]
                rebuilt = cam_ray_dirs @ Ri.T
                ref = cams[i].ray_maps.reshape(-1, 6)
                err_d = (rebuilt - ref[:, 3:]).abs().max().item()
                err_o = (ref[:, :3] - eyes[i][None]).abs().max().item()
                assert err_d < 1e-5 and err_o < 1e-6, f"ray-map mismatch on view {i}: dirs {err_d:.2e} origins {err_o:.2e}"
        if image_names is not None and len(image_names) != n:
            print(f"[export_cameras] WARNING: {len(image_names)} image names vs {n} cameras; dropping names")
            image_names = None
        bundle = {
            "format": "powersim.cameras.v1",
            "split": split,
            "img_wh": [int(W), int(H)],
            "poses": poses,
            "eyes": eyes,
            "rights": rights,
            "ups": ups,
            "cam_ray_dirs": cam_ray_dirs,
            "dataset": args.dataset,
            "scene": args.scene,
            "downsample": args.downsample[-1],
            "is_pinhole": bool(args.is_pinhole),
            "image_names": image_names,
        }
        path = out_dir / f"cameras_{split}.pt"
        torch.save(bundle, path)
        mb = path.stat().st_size / 1e6
        print(f"[export_cameras] {split}: {n} cameras {W}x{H} ray_dirs={'yes' if cam_ray_dirs is not None else 'pinhole'} -> {path} ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
