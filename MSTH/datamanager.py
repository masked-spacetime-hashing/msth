import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union

try:
    from typing import Literal, Callable
except ImportError:
    from typing_extensions import Literal

from pathlib import Path

import torch
import os
from rich.progress import Console
from torch.nn.parameter import Parameter
from torch.utils.data.dataloader import DataLoader

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig
from nerfstudio.data.pixel_samplers import PixelSampler
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.model_components.ray_generators import RayGenerator
from MSTH.dataparser import (
    VideoDataParser,
    VideoDataParserConfig,
    VideoDataParserOutputs,
)

# from MSTH.dataset import EvalVideoDataset, VideoDataset
from MSTH.dataset import VideoDataset, VideoDatasetWithFeature, VideoDatasetAllCached, VideoDatasetAllCachedUint8
from MSTH.sampler import (
    CompletePixelSampler,
    CompletePixelSamplerIter,
    PixelTimeSampler,
    PixelTimeUniformSampler,
    spacetime_samplers,
    spacetime_samplers_default_args,
    PixelTimeUniformSampler_origin,
    SpatioTemporalSampler,
)
from MSTH.utils import Timer

CONSOLE = Console(width=120)


@dataclass
class VideoDataManagerConfig(DataManagerConfig):
    """Video Data Manager config"""

    _target: Type = field(default_factory=lambda: VideoDataManager)
    dataparser: VideoDataParserConfig = VideoDataParserConfig()

    collate_fn = staticmethod(nerfstudio_collate)
    """Specifies the collate function to use for the train and eval dataloaders."""
    camera_res_scale_factor: float = 1.0
    train_num_rays_per_batch: int = 1024
    eval_num_rays_per_batch: int = 1024

    camera_optimizer: CameraOptimizerConfig = CameraOptimizerConfig()

    mask_extend_radius: int = 5
    next_n_frames: int = 1
    """mask extend radius for gaussian filter"""


class VideoDataManager(DataManager):
    train_dataset: VideoDataset
    eval_dataset: VideoDataset
    train_dataparser_outputs: VideoDataParserOutputs
    train_pixel_sampler: Optional[CompletePixelSampler] = None
    eval_dynamic_pixel_sampler: CompletePixelSampler
    eval_all_pixel_sampler: CompletePixelSampler

    def __init__(
        self,
        config: VideoDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        self.static_train_count = 0
        self.dynamic_train_count = 0
        self.dynamic_train_count_inverse = 0
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"
        self.dataparser_config = self.config.dataparser
        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data
        self.dataparser = self.dataparser_config.setup()
        self.train_dataparser_outputs = self.dataparser.get_dataparser_outputs(split="train")

        self.train_dataset = self.create_train_dataset()
        self.eval_dataset = self.create_eval_dataset()
        super().__init__()

    def create_train_dataset(self) -> VideoDataset:
        return VideoDataset(
            self.train_dataparser_outputs,
            self.config.camera_res_scale_factor,
            self.config.mask_extend_radius,
            self.config.next_n_frames,
        )

    def create_eval_dataset(self) -> VideoDataset:
        return VideoDataset(
            self.dataparser.get_dataparser_outputs(split=self.test_split),
            self.config.camera_res_scale_factor,
            self.config.mask_extend_radius,
            self.config.next_n_frames,
        )

    def _get_train_pixel_sampler(self, cur_frame_data):
        return CompletePixelSampler(self.config.train_num_rays_per_batch, cur_frame_data)

    @property
    def train_num_rays_per_batch(self):
        return self.config.train_num_rays_per_batch

    # def _get_eval_pixel_sampler(self):

    def setup_train(self):
        # TODO: finish this
        CONSOLE.print("Setting up stuffs for training")
        self.train_pixel_sampler = self._get_train_pixel_sampler(self.train_dataset.get_all_data(self.device))
        self.train_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.train_dataset.cameras.size, device=self.device
        )
        self.train_ray_generator = RayGenerator(
            self.train_dataset.cameras.to(self.device),
            self.train_camera_optimizer,
        )

    def setup_eval(self):
        """set up eval preparation"""
        # TODO: finish here
        CONSOLE.print("Setting up stuffs for evaluating")
        # self.eval_pixel_sampler = self._get_train_pixel_sampler(self.eval_dataset.get_all_data(self.device))
        self.eval_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.eval_dataset.cameras.size, device=self.device
        )
        self.eval_dynamic_pixel_sampler = CompletePixelSampler(
            self.config.eval_num_rays_per_batch, self.eval_dataset.get_all_data(self.device)
        )
        self.eval_all_pixel_sampler = CompletePixelSampler(
            self.config.eval_num_rays_per_batch, self.eval_dataset.get_all_data(self.device), use_mask=False
        )
        assert self.eval_dataset.cur_frame == 1
        assert self.eval_all_pixel_sampler is not None
        assert self.eval_dynamic_pixel_sampler is not None

        self.eval_ray_generator = RayGenerator(self.eval_dataset.cameras.to(self.device), self.eval_camera_optimizer)

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Return next batch"""
        self.dynamic_train_count += 1
        assert self.train_pixel_sampler is not None

        batch = self.train_pixel_sampler.sample()
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle, batch

    def next_train_inverse(self, step: int) -> Tuple[RayBundle, Dict]:
        """Return next batch"""
        self.dynamic_train_count_inverse += 1
        assert self.train_pixel_sampler is not None

        batch = self.train_pixel_sampler.sample_inverse()
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)

        return ray_bundle, batch

    def next_eval_all(self, step: int):
        """Return next eval batch"""
        ## TODO: impl here
        self.eval_count += 1
        assert self.eval_all_pixel_sampler is not None

        batch = self.eval_all_pixel_sampler.sample()
        ray_indices = batch["indices"]
        ray_bundle = self.eval_ray_generator(ray_indices)

        return ray_bundle, batch

    def next_eval_dynamic(self, step: int):
        assert self.eval_dynamic_pixel_sampler is not None

        batch = self.eval_dynamic_pixel_sampler.sample()
        ray_indices = batch["indices"]
        ray_bundle = self.eval_ray_generator(ray_indices)

        return ray_bundle, batch

    def next_eval_image(self, idx: int):
        # TODO: impl this
        image_idx = idx
        if idx < 0:
            image_idx = random.randint(0, self.eval_dataset.num_cams - 1)
        camera_ray_bundle = self.eval_ray_generator.cameras.generate_rays(camera_indices=image_idx, keep_shape=True)
        batch = self.eval_dataset[image_idx]
        return image_idx, camera_ray_bundle, batch

    def get_train_rays_per_batch(self) -> int:
        return self.config.train_num_rays_per_batch

    def get_eval_rays_per_batch(self) -> int:
        return self.config.eval_num_rays_per_batch

    def get_datapath(self) -> Path:
        return self.config.dataparser.data

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        return {}

    def tick(self, tick_n_frames=False):
        # TODO: make sure have done everything needed to tick
        if tick_n_frames:
            self.train_dataset.tick_n_frames()
            self.eval_dataset.tick_n_frames()
        else:
            self.train_dataset.tick()
            self.eval_dataset.tick()
        assert self.train_pixel_sampler is not None
        self.train_pixel_sampler.set_batch(self.train_dataset.get_all_data(self.device))
        assert self.eval_all_pixel_sampler is not None
        assert self.eval_dynamic_pixel_sampler is not None
        self.eval_all_pixel_sampler.set_batch(self.eval_dataset.get_all_data(self.device))
        self.eval_dynamic_pixel_sampler.set_batch(self.eval_dataset.get_all_data(self.device))


@dataclass
class VideoFeatureDataManagerConfig(DataManagerConfig):
    """Video Data Manager config"""

    _target: Type = field(default_factory=lambda: VideoFeatureDataManager)
    dataparser: VideoDataParserConfig = VideoDataParserConfig()

    collate_fn = staticmethod(nerfstudio_collate)
    """Specifies the collate function to use for the train and eval dataloaders."""
    camera_res_scale_factor: float = 1.0
    train_num_rays_per_batch: int = 1024
    eval_num_rays_per_batch: int = 1024

    camera_optimizer: CameraOptimizerConfig = CameraOptimizerConfig()

    mask_extend_radius: int = 5
    pretrained_path = "/data/czl/nerf/MSTH/MSTH/ibrnet/model_255000.pth"
    next_n_frames: int = 1
    fe_device: str = "cpu"
    """mask extend radius for gaussian filter"""


class VideoFeatureDataManager(DataManager):
    train_dataset: VideoDatasetWithFeature
    eval_dataset: VideoDatasetWithFeature
    train_dataparser_outputs: VideoDataParserOutputs
    train_pixel_sampler: Optional[CompletePixelSampler] = None
    eval_dynamic_pixel_sampler: CompletePixelSampler
    eval_all_pixel_sampler: CompletePixelSampler

    def __init__(
        self,
        config: VideoFeatureDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        self.static_train_count = 0
        self.dynamic_train_count = 0
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"
        self.dataparser_config = self.config.dataparser
        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data
        self.dataparser = self.dataparser_config.setup()
        self.train_dataparser_outputs = self.dataparser.get_dataparser_outputs(split="train")

        self.train_dataset = self.create_train_dataset()
        self.eval_dataset = self.create_eval_dataset()
        super().__init__()

    def create_train_dataset(self) -> VideoDatasetWithFeature:
        return VideoDatasetWithFeature(
            self.train_dataparser_outputs,
            self.config.camera_res_scale_factor,
            self.config.mask_extend_radius,
            1,
            self.config.pretrained_path,
            self.config.fe_device,
        )

    def create_eval_dataset(self) -> VideoDatasetWithFeature:
        return VideoDatasetWithFeature(
            self.dataparser.get_dataparser_outputs(split=self.test_split),
            self.config.camera_res_scale_factor,
            self.config.mask_extend_radius,
            1,
            self.config.pretrained_path,
            self.config.fe_device,
        )

    def _get_train_pixel_sampler(self, cur_frame_data):
        return CompletePixelSampler(self.config.train_num_rays_per_batch, cur_frame_data)

    @property
    def train_num_rays_per_batch(self):
        return self.config.train_num_rays_per_batch

    # def _get_eval_pixel_sampler(self):

    def setup_train(self):
        # TODO: finish this
        CONSOLE.print("Setting up stuffs for training")
        self.train_pixel_sampler = self._get_train_pixel_sampler(self.train_dataset.get_all_data(self.device))
        self.train_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.train_dataset.cameras.size, device=self.device
        )
        self.train_ray_generator = RayGenerator(
            self.train_dataset.cameras.to(self.device),
            self.train_camera_optimizer,
        )

    def setup_eval(self):
        """set up eval preparation"""
        # TODO: finish here
        CONSOLE.print("Setting up stuffs for evaluating")
        # self.eval_pixel_sampler = self._get_train_pixel_sampler(self.eval_dataset.get_all_data(self.device))
        self.eval_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.eval_dataset.cameras.size, device=self.device
        )
        self.eval_dynamic_pixel_sampler = CompletePixelSampler(
            self.config.eval_num_rays_per_batch, self.eval_dataset.get_all_data(self.device)
        )
        self.eval_all_pixel_sampler = CompletePixelSampler(
            self.config.eval_num_rays_per_batch, self.eval_dataset.get_all_data(self.device), use_mask=False
        )
        assert self.eval_dataset.cur_frame == 1
        assert self.eval_all_pixel_sampler is not None
        assert self.eval_dynamic_pixel_sampler is not None

        self.eval_ray_generator = RayGenerator(self.eval_dataset.cameras.to(self.device), self.eval_camera_optimizer)

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Return next batch"""
        self.dynamic_train_count += 1
        assert self.train_pixel_sampler is not None

        batch = self.train_pixel_sampler.sample()
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle, batch

    def next_train_inverse(self, step: int) -> Tuple[RayBundle, Dict]:
        """Return next batch"""
        self.dynamic_train_count_inverse += 1
        assert self.train_pixel_sampler is not None

        batch = self.train_pixel_sampler.sample_inverse()
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)

        return ray_bundle, batch

    def next_eval_all(self, step: int):
        """Return next eval batch"""
        ## TODO: impl here
        self.eval_count += 1
        assert self.eval_all_pixel_sampler is not None

        batch = self.eval_all_pixel_sampler.sample()
        ray_indices = batch["indices"]
        ray_bundle = self.eval_ray_generator(ray_indices)

        return ray_bundle, batch

    def next_eval_dynamic(self, step: int):
        assert self.eval_dynamic_pixel_sampler is not None

        batch = self.eval_dynamic_pixel_sampler.sample()
        ray_indices = batch["indices"]
        ray_bundle = self.eval_ray_generator(ray_indices)

        return ray_bundle, batch

    def next_eval_image(self, idx: int):
        # TODO: impl this
        image_idx = idx
        if idx < 0:
            image_idx = random.randint(0, self.eval_dataset.num_cams - 1)
        camera_ray_bundle = self.eval_ray_generator.cameras.generate_rays(camera_indices=image_idx, keep_shape=True)
        batch = self.eval_dataset[image_idx]
        return image_idx, camera_ray_bundle, batch

    def get_train_rays_per_batch(self) -> int:
        return self.config.train_num_rays_per_batch

    def get_eval_rays_per_batch(self) -> int:
        return self.config.eval_num_rays_per_batch

    def get_datapath(self) -> Path:
        return self.config.dataparser.data

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        return {}

    def tick(self, tick_n_frames=False):
        # TODO: make sure have done everything needed to tick
        if tick_n_frames:
            self.train_dataset.tick_n_frames()
            self.eval_dataset.tick_n_frames()
        else:
            self.train_dataset.tick()
            self.eval_dataset.tick()
        assert self.train_pixel_sampler is not None
        # extract features
        self.train_dataset.extract_cur_frame_feature()
        self.train_pixel_sampler.set_batch(self.train_dataset.get_all_data(self.device))
        assert self.eval_all_pixel_sampler is not None
        assert self.eval_dynamic_pixel_sampler is not None
        self.eval_all_pixel_sampler.set_batch(self.eval_dataset.get_all_data(self.device))
        self.eval_dynamic_pixel_sampler.set_batch(self.eval_dataset.get_all_data(self.device))

    @property
    def get_num_dynamic_rays(self):
        return self.train_pixel_sampler.all_indices.size(0)


@dataclass
class SpaceTimeDataManagerConfig(DataManagerConfig):
    _target: Type = field(default_factory=lambda: SpaceTimeDataManager)
    dataparser: VideoDataParserConfig = VideoDataParserConfig()

    collate_fn = staticmethod(nerfstudio_collate)
    """Specifies the collate function to use for the train and eval dataloaders."""
    camera_res_scale_factor: float = 1.0
    train_num_rays_per_batch: int = 1024
    eval_num_rays_per_batch: int = 1024

    camera_optimizer: CameraOptimizerConfig = CameraOptimizerConfig()
    dataset_mask_precomputed_path: Path = Path("/data/machine/data/flame_salmon_videos_2/masks.pt")

    mask_extend_radius: int = 5
    use_uint8: bool = True
    use_isg_sampler: bool = True
    static_dynamic_sampling_ratio_end: float = 1
    static_ratio_decay_total_steps: int = 6000
    use_all_uniform_sampler: bool = False

    use_stratified_pixel_sampler: bool = False  # deprecated
    spatial_temporal_sampler: Literal["uniform", "stratified", "st"] = "stratified"
    use_static_dynamic_ratio_anealing: bool = False
    static_dynamic_sampling_ratio: float = 1.0
    ratio_anealing_start: int = 0
    ratio_anealing_end: int = 20000
    initial_ratio: float = 50
    final_ratio: float = 10
    # n_time_for_dynamic: int = 1  # parameters for spatial_temporal sampler, 1 is equal to stratified sampler
    n_time_for_dynamic: Callable[
        [int], float
    ] = lambda x: 1  # parameters for spatial_temporal sampler, 1 is equal to stratified sampler
    use_temporal_weight: str = "none"
    use_median: bool = False

    train_sampler_type: str = "spatio"
    train_sampler_args: Type = field(default_factory=lambda: {})


class SpaceTimeDataManager(DataManager):
    train_dataset: Union[VideoDatasetAllCached, VideoDatasetAllCachedUint8]
    eval_dataset: Union[VideoDatasetAllCached, VideoDatasetAllCachedUint8]
    train_dataparser_outputs: VideoDataParserOutputs
    train_pixel_sampler: Union[PixelTimeUniformSampler, PixelTimeSampler, SpatioTemporalSampler]
    eval_pixel_sampler: Union[PixelTimeUniformSampler, PixelTimeSampler]

    def __init__(
        self,
        config: SpaceTimeDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        self.train_count = 0
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"
        self.dataparser_config = self.config.dataparser
        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data
        self.dataparser = self.dataparser_config.setup()
        self.train_dataparser_outputs = self.dataparser.get_dataparser_outputs(split="train")

        self.train_dataset = self.create_train_dataset()
        self.eval_dataset = self.create_eval_dataset()
        super().__init__()

    def create_train_dataset(self):
        if not self.config.use_uint8:
            return VideoDatasetAllCached(
                self.train_dataparser_outputs, self.config.camera_res_scale_factor, self.config.mask_extend_radius
            )
        else:
            return VideoDatasetAllCachedUint8(
                self.train_dataparser_outputs,
                self.config.camera_res_scale_factor,
                self.config.mask_extend_radius,
                use_median=self.config.use_median,
            )

    def create_eval_dataset(self):
        if not self.config.use_uint8:
            return VideoDatasetAllCached(
                self.dataparser.get_dataparser_outputs(split=self.test_split),
                self.config.camera_res_scale_factor,
                self.config.mask_extend_radius,
            )
        else:
            return VideoDatasetAllCachedUint8(
                self.dataparser.get_dataparser_outputs(split=self.test_split),
                self.config.camera_res_scale_factor,
                self.config.mask_extend_radius,
                use_mask=False,
            )

    def setup_train(self):
        CONSOLE.print("Setting up for training")
        # sampler_cls = spacetime_samplers[self.config.sampler]
        # sampler_args = {"dataset": self.train_dataset, "num_rays_per_batch": self.config.train_num_rays_per_batch}
        # sampler_args.update(self.config.sampler_extra_args)
        # self.train_pixel_sampler = sampler_cls(**sampler_args)
        # if not self.config.use_stratified_pixel_sampler:
        sampler_args = {
            "dataset": self.train_dataset,
            "num_rays_per_batch": self.config.train_num_rays_per_batch,
            "device": self.device,
        }

        if self.config.spatial_temporal_sampler == "uniform":
            if not self.config.use_all_uniform_sampler:
                self.train_pixel_sampler = PixelTimeUniformSampler(
                    self.train_dataset, self.config.train_num_rays_per_batch
                )
            else:
                self.train_pixel_sampler = PixelTimeUniformSampler_origin(
                    self.train_dataset, self.config.train_num_rays_per_batch
                )

        elif self.config.spatial_temporal_sampler == "stratified":
            self.train_pixel_sampler = PixelTimeSampler(
                self.train_dataset,
                self.config.train_num_rays_per_batch,
                static_dynamic_ratio=self.config.static_dynamic_sampling_ratio,
                static_dynamic_ratio_end=self.config.static_dynamic_sampling_ratio_end,
                total_steps=self.config.static_ratio_decay_total_steps,
            )
        elif self.config.spatial_temporal_sampler == "st":
            extra_args = dict(
                static_dynamic_ratio=self.config.static_dynamic_sampling_ratio,
                static_dynamic_ratio_end=self.config.static_dynamic_sampling_ratio_end,
                total_steps=self.config.static_ratio_decay_total_steps,
                n_time_for_dynamic=self.config.n_time_for_dynamic,
                use_temporal_weight=self.config.use_temporal_weight,
            )
            sampler_args.update(extra_args)
            del sampler_args["device"]

        sampler_type = spacetime_samplers[self.config.train_sampler_type]
        sampler_args.update(spacetime_samplers_default_args[self.config.train_sampler_type])
        sampler_args.update(self.config.train_sampler_args)

        self.train_pixel_sampler = sampler_type(**sampler_args)

        self.train_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.train_dataset.cameras.size, device=self.device
        )
        self.train_ray_generator = RayGenerator(
            self.train_dataset.cameras.to(self.device),
            self.train_camera_optimizer,
        )

    def setup_eval(self):
        CONSOLE.print("Setting up for evaluating")
        self.eval_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.eval_dataset.cameras.size, device=self.device
        )

        self.eval_pixel_sampler = PixelTimeUniformSampler(
            self.eval_dataset,
            self.config.eval_num_rays_per_batch,
        )

        self.eval_ray_generator = RayGenerator(self.eval_dataset.cameras.to(self.device), self.eval_camera_optimizer)

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        if self.config.use_static_dynamic_ratio_anealing:
            assert self.ratio is not None
            assert isinstance(self.train_pixel_sampler, PixelTimeSampler)
            # self.train_pixel_sampler.set_static_dynamic_ratio(self.ratio(step))
        self.train_count += 1
        batch = self.train_pixel_sampler.sample()
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        ray_bundle.times = batch["time"][..., None]

        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        self.eval_count += 1
        batch = self.eval_pixel_sampler.sample()
        ray_indices = batch["indices"]
        ray_bundle = self.eval_ray_generator(ray_indices)
        ray_bundle.times = batch["time"][..., None]

        return ray_bundle, batch

    def get_train_rays_per_batch(self) -> int:
        return self.config.train_num_rays_per_batch

    def get_eval_rays_per_batch(self) -> int:
        return self.config.eval_num_rays_per_batch

    def get_datapath(self) -> Path:
        return self.config.dataparser.data

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        return {}

    # def next_eval_image(self, idx: int):
    #     # TODO: impl this
    #     image_idx = idx
    #     if idx < 0:
    #         image_idx = random.randint(0, self.eval_dataset.num_cams - 1)
    #     camera_ray_bundle = self.eval_ray_generator.cameras.generate_rays(camera_indices=image_idx, keep_shape=True)
    #     batch = self.eval_dataset[image_idx]
    #     return image_idx, camera_ray_bundle, batch

    def next_eval_image(self, frame_idx: int):
        camera_ray_bundle = self.eval_ray_generator.cameras.generate_rays(camera_indices=0, keep_shape=True)
        batch = self.eval_dataset.frames[0, frame_idx]
        if batch.dtype != torch.float32:
            batch = batch.to(torch.float32) / 255.0
        batch = {"image": batch}
        batch["time"] = frame_idx / self.eval_dataset.num_frames
        camera_ray_bundle.times = torch.zeros_like(camera_ray_bundle.origins[..., :1]).to(camera_ray_bundle.origins)
        camera_ray_bundle.times.fill_(batch["time"])

        return frame_idx, camera_ray_bundle, batch

    def next_eval_image_incremental(self, frame_idx, camera_ray_bundle):
        batch = self.eval_dataset.frames[0, frame_idx]
        if batch.dtype != torch.float32:
            batch = batch.to(torch.float32) / 255.0
        batch = {"image": batch}
        batch["time"] = frame_idx / self.eval_dataset.num_frames
        camera_ray_bundle.times.fill_(batch["time"])

        return frame_idx, camera_ray_bundle, batch

    # def eval_all_images()
