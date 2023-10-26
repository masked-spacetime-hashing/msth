from __future__ import annotations

import json
import math
import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import List, Optional, Type
from typing import *
import cv2

import numpy as np
import torch
from PIL import Image
from rich.console import Console
from typing_extensions import Literal

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json
from torchtyping import TensorType

CONSOLE = Console(width=120)
MAX_AUTO_RESOLUTION = 1600


@dataclass
class VideoDataParserOutputs:
    data_dir: Path
    video_filenames: List[Path]
    start_frame: int
    num_frames: int
    """Dataparser outputs for the which will be used by the DataManager
    for creating RayBundle and RayGT objects."""

    """Filenames for the images."""
    cameras: Cameras
    """Camera object storing collection of camera information in dataset."""
    alpha_color: Optional[TensorType[3]] = None
    """Color of dataset background."""
    scene_box: SceneBox = SceneBox()
    """Scene box of dataset. Used to bound the scene or provide the scene scale depending on model."""
    mask_filenames: Optional[List[Path]] = None
    """Filenames for any masks that are required"""
    metadata: Dict[str, Any] = to_immutable_dict({})
    """Dictionary of any metadata that be required for the given experiment.
    Will be processed by the InputDataset to create any additional tensors that may be required.
    """
    dataparser_transform: TensorType[3, 4] = torch.eye(4)[:3, :]
    """Transform applied by the dataparser."""
    dataparser_scale: float = 1.0
    """Scale applied by the dataparser."""

    def as_dict(self) -> dict:
        """Returns the dataclass as a dictionary."""
        return vars(self)

    def save_dataparser_transform(self, path: Path):
        """Save dataparser transform to json file. Some dataparsers will apply a transform to the poses,
        this method allows the transform to be saved so that it can be used in other applications.

        Args:
            path: path to save transform to
        """
        data = {
            "transform": self.dataparser_transform.tolist(),
            "scale": float(self.dataparser_scale),
        }
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        with open(path, "w", encoding="UTF-8") as file:
            json.dump(data, file, indent=4)


@dataclass
class VideoDataParserConfig(DataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: VideoDataParser)
    """target class to instantiate"""
    data: Path = Path("/opt/czl/nerf/data/flame_salmon_videos")
    """Directory or explicit json file path specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: Optional[int] = 1
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    orientation_method: Literal["pca", "up", "none"] = "up"
    """The method to use for orientation."""
    center_poses: bool = True
    """Whether to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    train_split_fraction: float = 0.9
    """The fraction of images to use for training. The remaining images are for eval."""
    depth_unit_scale_factor: float = 1e-3
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""
    use_llff_poses: bool = False


@dataclass
class VideoDataParser(DataParser):
    """Video dataparser"""

    config: VideoDataParserConfig
    downscale_factor: Optional[int] = None

    def _generate_dataparser_outputs(self, split: str = "train") -> VideoDataParserOutputs:
        # print(list(os.listdir(self.config.data)))
        use_separate_file = False
        if "transforms_train.json" in list(os.listdir(self.config.data)):
            use_separate_file = True
            CONSOLE.log("Using separated config files for train and eval")
            meta_train = load_from_json(self.config.data / "transforms_train.json")
            meta_val = load_from_json(self.config.data / "transforms_test.json")

            num_train_cams = len(meta_train["frames"])
            num_val_cams = len(meta_val["frames"])
            meta = deepcopy(meta_train)
            meta["frames"].extend(meta_val["frames"])
        else:
            meta = load_from_json(self.config.data / "transforms.json")
            num_tot_cams = len(meta["frames"])
            num_train_cams = math.ceil(num_tot_cams * self.config.train_split_fraction)
        data_dir = self.config.data
        print(self.config.data)
        # exit(0)

        video_filenames = []
        poses = []

        fx_fixed = "fl_x" in meta
        fy_fixed = "fl_y" in meta
        cx_fixed = "cx" in meta
        cy_fixed = "cy" in meta
        height_fixed = "h" in meta
        width_fixed = "w" in meta
        distort_fixed = False
        for distort_key in ["k1", "k2", "k3", "p1", "p2"]:
            if distort_key in meta:
                distort_fixed = True
                break
        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        num_frames = meta["num_frames"]
        start_frame = meta.get("start_frame", 0)

        for frame in meta["frames"]:
            filepath = PurePath(frame["file_path"])
            assert filepath.suffix in [".mp4", ".mov", ".mkv"]

            if not fx_fixed:
                assert "fl_x" in frame, "fx not specified in frame"
                fx.append(float(frame["fl_x"]))
            if not fy_fixed:
                assert "fl_y" in frame, "fy not specified in frame"
                fy.append(float(frame["fl_y"]))
            if not cx_fixed:
                assert "cx" in frame, "cx not specified in frame"
                cx.append(float(frame["cx"]))
            if not cy_fixed:
                assert "cy" in frame, "cy not specified in frame"
                cy.append(float(frame["cy"]))
            if not height_fixed:
                assert "h" in frame, "height not specified in frame"
                height.append(int(frame["h"]))
            if not width_fixed:
                assert "w" in frame, "width not specified in frame"
                width.append(int(frame["w"]))
            if not distort_fixed:
                distort.append(
                    camera_utils.get_distortion_params(
                        k1=float(frame["k1"]) if "k1" in frame else 0.0,
                        k2=float(frame["k2"]) if "k2" in frame else 0.0,
                        k3=float(frame["k3"]) if "k3" in frame else 0.0,
                        k4=float(frame["k4"]) if "k4" in frame else 0.0,
                        p1=float(frame["p1"]) if "p1" in frame else 0.0,
                        p2=float(frame["p2"]) if "p2" in frame else 0.0,
                    )
                )

            video_filenames.append(self.config.data / filepath)
            poses.append(np.array(frame["transform_matrix"]))

        num_tot_cams = len(video_filenames)
        num_eval_cams = num_tot_cams - num_train_cams

        if not use_separate_file:
            i_all = np.arange(num_tot_cams)
            i_train = np.arange(num_train_cams)

            i_eval = np.setdiff1d(i_all, i_train)
            # i_eval = i_all[-1:]
            # i_train = i_all[:-1]
        else:
            i_all = np.arange(num_tot_cams)
            i_train = i_all[:-1]
            i_eval = i_all[-1:]

        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        if split != "train":
            CONSOLE.print(f"Eval camera names: {(video_filenames[i_eval[0]])}")
        else:
            CONSOLE.print("Train camera names:")
            for ii in i_train:
                CONSOLE.print(f"{video_filenames[i_train[ii]]}")

        if "orientation_override" in meta:
            orientation_method = meta["orientation_override"]
            CONSOLE.log(f"[yellow] Dataset is overriding orientation method to {orientation_method}")
        else:
            orientation_method = self.config.orientation_method

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=orientation_method,
            center_poses=self.config.center_poses,
        )

        if self.config.use_llff_poses:
            poses_path = Path(self.config.data) / "poses.npy"
            poses = np.load(poses_path)
            poses = torch.from_numpy(poses).to(torch.float32)
            perm = [i for i in range(1, len(i_train) + 1)]
            perm.append(0)
            poses = poses[perm]

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor

        poses[:, :3, 3] *= scale_factor

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        video_filenames = [video_filenames[i] for i in indices]
        poses = poses[indices]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        if "camera_model" in meta:
            camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]
        else:
            camera_type = CameraType.PERSPECTIVE

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        fx = float(meta["fl_x"]) if fx_fixed else torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        fy = float(meta["fl_y"]) if fy_fixed else torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        cx = float(meta["cx"]) if cx_fixed else torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        cy = float(meta["cy"]) if cy_fixed else torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        height = int(meta["h"]) if height_fixed else torch.tensor(height, dtype=torch.int32)[idx_tensor]
        width = int(meta["w"]) if width_fixed else torch.tensor(width, dtype=torch.int32)[idx_tensor]
        if distort_fixed:
            distortion_params = camera_utils.get_distortion_params(
                k1=float(meta["k1"]) if "k1" in meta else 0.0,
                k2=float(meta["k2"]) if "k2" in meta else 0.0,
                k3=float(meta["k3"]) if "k3" in meta else 0.0,
                k4=float(meta["k4"]) if "k4" in meta else 0.0,
                p1=float(meta["p1"]) if "p1" in meta else 0.0,
                p2=float(meta["p2"]) if "p2" in meta else 0.0,
            )
        else:
            distortion_params = torch.stack(distort, dim=0)[idx_tensor]

        ## debug
        if split != "train":
            print(poses)

        def get_elem(t):
            if isinstance(t, torch.Tensor):
                return t[0][0].item()
            return t

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
        )

        ## TODO: check this workaround
        self.downscale_factor = self.config.downscale_factor
        ## check if downscale needed
        cap = cv2.VideoCapture(str(video_filenames[0]))
        sample_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        sample_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        assert np.isclose(sample_width / sample_height, get_elem(width) / get_elem(height))
        calced_downscale_factor = get_elem(height) / sample_height

        print(f"loaded video size: ({sample_height}, {sample_width})")

        if not np.isclose(calced_downscale_factor, self.downscale_factor):
            print("downscale provided is incorrect, changed to the real one calulated using the loaded video")

        self.downscale_factor = calced_downscale_factor

        assert self.downscale_factor is not None
        cameras.rescale_output_resolution(scaling_factor=1.0 / self.downscale_factor)

        dataparser_outputs = VideoDataParserOutputs(
            data_dir=self.config.data,
            video_filenames=video_filenames,
            num_frames=num_frames,
            start_frame=start_frame,
            cameras=cameras,
            scene_box=scene_box,
            # mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata={
                # "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
                # "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
            },
        )
        return dataparser_outputs
