"""Load a trained PowerFoam checkpoint (config.yaml + model.pt). Cameras come from the shipped
cameras_<split>.pt when present, so no training images are needed; otherwise from the dataset."""

from dataclasses import dataclass
from pathlib import Path

import torch

from powersim.core.thirdparty_paths import ensure_powerfoam_on_path


@dataclass
class LoadedCheckpoint:
    scene: "PowerfoamScene"  # noqa: F821 - imported lazily, see load_powerfoam_checkpoint
    data_handler: "DataHandler"  # noqa: F821 - a DataHandler or a CameraBundle (same camera API)
    args: object
    checkpoint_dir: Path


class _LazyCameras:
    """Sequence of TorchCamera over a camera bundle."""

    def __init__(self, bundle: dict):
        from powerfoam.camera import TorchCamera

        self._TorchCamera = TorchCamera
        self.img_wh = tuple(int(v) for v in bundle["img_wh"])
        self.poses = bundle["poses"].float()
        self.eyes = bundle["eyes"].float()
        self.rights = bundle["rights"].float()
        self.ups = bundle["ups"].float()
        self.cam_ray_dirs = bundle["cam_ray_dirs"]
        if self.cam_ray_dirs is not None:
            self.cam_ray_dirs = self.cam_ray_dirs.float()

    def __len__(self):
        return self.eyes.shape[0]

    def _light(self, i):
        W, H = self.img_wh
        pin = (lambda t: t.pin_memory()) if torch.cuda.is_available() else (lambda t: t)
        return self._TorchCamera(eye=pin(self.eyes[i].clone()), right=pin(self.rights[i].clone()), up=pin(self.ups[i].clone()), width=W, height=H)

    def __iter__(self):
        for i in range(len(self)):
            yield self._light(i)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        if i < 0:
            i += len(self)
        if not 0 <= i < len(self):
            raise IndexError(i)
        cam = self._light(i)
        if self.cam_ray_dirs is not None:
            R = self.poses[i, :3, :3]
            dirs = self.cam_ray_dirs @ R.T  # the loader's einsum("ij,kj->ik", cam_dirs, R)
            origins = self.eyes[i][None, :].expand_as(dirs)
            ray_maps = torch.cat([origins, dirs], dim=-1).reshape(self.img_wh[1], self.img_wh[0], 6).contiguous()
            # The rasterizer's non-pinhole kernels read ray_maps straight from host memory (the
            # upstream loader stores them pinned, which CUDA kernels can address)
            cam.ray_maps = ray_maps.pin_memory() if torch.cuda.is_available() else ray_maps
        return cam


class CameraBundle:
    """Stand-in for powerfoam's DataHandler built from cameras_<split>.pt (cameras only, no images)"""

    def __init__(self, path):
        bundle = torch.load(path, map_location="cpu")
        if bundle.get("format") != "powersim.cameras.v1":
            raise ValueError(f"{path}: not a PowerSim camera bundle")
        self.path = Path(path)
        self.split = bundle["split"]
        self.cameras = _LazyCameras(bundle)
        self.image_names = bundle.get("image_names")  # per camera, same order (None if the export lacked them)
        self.img_wh = self.cameras.img_wh
        self.c2ws = self.cameras.poses
        self.points3D = None
        self.points3D_colors = None
        self.rgbs = None
        self.alphas = None
        self.normals = None


def camera_bundle_path(checkpoint_dir, split: str) -> Path:
    return Path(checkpoint_dir) / f"cameras_{split}.pt"


def load_powerfoam_checkpoint(
    config_path: str,
    device: str = "cuda",
    split: str = "train",
) -> LoadedCheckpoint:
    """Load a PowerFoam checkpoint given the path to its config.yaml."""
    ensure_powerfoam_on_path()

    import configargparse
    import warp as wp

    from configs import Params, add_group
    from powerfoam.scene import PowerfoamScene

    wp.init()

    checkpoint_dir = Path(config_path).resolve().parent

    parser = configargparse.ArgParser()
    get_params = add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True, help="Path to config file")
    args = parser.parse_args(["-c", str(config_path)])
    args = get_params(args)

    bundle_path = camera_bundle_path(checkpoint_dir, split)
    if bundle_path.exists():
        data_handler = CameraBundle(bundle_path)
        # initialize_from_dataset() draws `init_points` random points that load_pt() replaces; keep
        # that scaffolding cheap and independent of the SfM point cloud we do not ship
        args.init_type = "random_bounded"
        args.init_points = min(int(args.init_points), 4096)
    else:
        from data_loader import DataHandler

        data_handler = DataHandler(args)
        data_handler.reload(split, downsample=args.downsample[-1])

    scene = PowerfoamScene(args)
    scene.initialize_from_dataset(data_handler, device=device)
    scene.load_pt(str(checkpoint_dir / "model.pt"))

    return LoadedCheckpoint(
        scene=scene,
        data_handler=data_handler,
        args=args,
        checkpoint_dir=checkpoint_dir,
    )
