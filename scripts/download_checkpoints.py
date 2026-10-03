"""Download PowerSim's released checkpoints and assets from Hugging Face into data/.

    python scripts/download_checkpoints.py                 # everything (~14 GB)
    python scripts/download_checkpoints.py garden_bonsai   # only what one scene needs
    python scripts/download_checkpoints.py --list

Files land exactly where the run scripts expect them (data/powerfoam_ckpt/<name>/...,
data/mat_prop_ckpt/..., data/selections/<name>/...). Re-running skips files already present.
"""
import argparse
from pathlib import Path

REPO = "Tung11/PowerSim"

# scene -> the checkpoint folders it needs (selections / material fields are keyed by checkpoint)
SCENES = {
    "garden_bonsai": ["garden_v1_bonsai_foreground_edit"],
    "bread_roll": ["bread_roll@ec2ced65"],
    "ficus": ["ficus@48d229db"],
    "pillow": ["pillow2sofa_gs_fewer_points"],
    "coke_can": ["coke_can_mesh_fewer_points"],
    "wolf": ["wolf_gs_filled"],
    "telephone": ["telephone_sfm@2ed8d7fa"],
    "mic": ["mic@84daf179"],
    "carnation": ["carnations_sfm@ae005aa7"],  # + gt_video/carnation and mat_prop_ckpt/carnation
    "garden_edit": ["garden_v1", "bonsai_foreground"],  # + masks/garden_v1 (foamedit example)
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenes", nargs="*", help=f"subset of {sorted(SCENES)}; default: everything")
    ap.add_argument("--list", action="store_true", help="list the files in the repo and exit")
    ap.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[1] / "data"))
    a = ap.parse_args()

    from huggingface_hub import HfApi, snapshot_download

    if a.list:
        for f in sorted(HfApi().list_repo_files(REPO, repo_type="dataset")):
            print(f)
        return

    patterns = None
    if a.scenes:
        unknown = [s for s in a.scenes if s not in SCENES]
        if unknown:
            ap.error(f"unknown scene(s) {unknown}; choose from {sorted(SCENES)}")
        patterns = ["README.md", "mat_prop_ckpt/*", "mat_prop_ckpt/**"]
        for s in a.scenes:
            for ckpt in SCENES[s]:
                patterns += [f"powerfoam_ckpt/{ckpt}/*", f"selections/{ckpt}/*"]
            if s == "carnation":
                patterns += ["gt_video/carnation/*"]
            if s == "garden_edit":
                patterns += ["masks/garden_v1/*/*", "images/garden_v1/*", "selections/garden_v1/*", "selections/bonsai_foreground/*"]
    path = snapshot_download(REPO, repo_type="dataset", local_dir=a.data_dir, allow_patterns=patterns)
    print(f"downloaded to {path}")


if __name__ == "__main__":
    main()
