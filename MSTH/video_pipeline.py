from __future__ import annotations

import random
import typing
from abc import abstractmethod
from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, List, Mapping, Optional, Type, Union, cast

import cv2
import imageio
import numpy as np
import torch
import torch.distributed as dist
import wandb
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from torch import nn
from torch.nn import Parameter
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.io import write_video
from tqdm import tqdm, trange
from typing_extensions import Literal

from nerfstudio.configs import base_config as cfg
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipelineConfig
from nerfstudio.utils import colormaps, profiler
from MSTH.datamanager import (
    SpaceTimeDataManager,
    SpaceTimeDataManagerConfig,
    VideoDataManager,
    VideoDataManagerConfig,
    VideoFeatureDataManager,
    VideoFeatureDataManagerConfig,
)
from MSTH.SpaceTimeHashing.render import get_render_cameras
from MSTH.utils import Timer, get_chart


def module_wrapper(ddp_or_model: Union[DDP, Model]) -> Model:
    """
    If DDP, then return the .module. Otherwise, return the model.
    """
    if isinstance(ddp_or_model, DDP):
        return cast(Model, ddp_or_model.module)
    return ddp_or_model


@dataclass
class VideoPipelineConfig(cfg.InstantiateConfig):
    _target: Type = field(default_factory=lambda: VideoPipeline)
    datamanager: Union[
        VideoDataManagerConfig, VideoFeatureDataManagerConfig, DataManagerConfig
    ] = VideoDataManagerConfig()
    model: ModelConfig = ModelConfig()


class VideoPipeline(Pipeline):
    def __init__(
        self,
        config: VanillaPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
    ):
        super().__init__()
        self.config = config
        self.test_mode = test_mode
        self.datamanager: VideoDataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        self.datamanager.to(device)
        # TODO(ethan): get rid of scene_bounds from the model
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        extra_args = {}
        if isinstance(self.datamanager, VideoFeatureDataManager):
            extra_args["train_cameras"] = self.datamanager.train_dataset.cameras
            extra_args["eval_cameras"] = self.datamanager.eval_dataset.cameras
            extra_args["train_dataset"] = self.datamanager.train_dataset
            extra_args["device"] = "cuda"

        print("extra_args: ", extra_args)
        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            **extra_args,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.model.device

    @profiler.time_function
    def get_static_train_loss_dict(self, step: int):
        # TODO: add static specific training procedures here
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_dynamic_train_loss_dict(self, step: int):
        # TODO: add dynamic specific training procedures here
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict

    def hash_reinitialize(self, step: int, std: float):
        with Timer(des="next_train"):
            ray_bundle, batch = self.datamanager.next_train(step)
        with Timer(des="reinit"):
            self.model.hash_reinitialize(ray_bundle, std)
            # print("hello")

    def set_static(self, step: int):
        ray_bundle, batch = self.datamanager.next_train_inverse(step)
        self.model.set_static(ray_bundle)

    # @profiler.time_function
    # def get_static_eval_loss_dict(self, step: int):
    #     self.eval()
    #     ray_bundle, batch = self.datamanager.next_eval_all(step)
    #     model_outputs = self.model(ray_bundle)
    #     metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
    #     loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
    #     self.train()
    #     return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_static_eval_loss_dict(self, step: int):
        self.eval()
        ray_bundle, batch = self.datamanager.next_eval_all(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_dynamic_eval_loss_dict(self, step: int):
        self.eval()
        ray_bundle, batch = self.datamanager.next_eval_dynamic(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        self.eval()
        # -1 for random sampling in eval dataset
        image_idx, camera_ray_bundle, batch = self.datamanager.next_eval_image(-1)
        outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
        assert "image_idx" not in metrics_dict
        metrics_dict["image_idx"] = image_idx
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = len(camera_ray_bundle)
        self.train()
        return metrics_dict, images_dict

    def get_cur_frame_eval_mask(self):
        return self.datamanager.eval_dataset[0]["mask"]

    @profiler.time_function
    def get_average_eval_image_metrics(self, step: Optional[int] = None):
        self.eval()
        metrics_dict_list = []
        num_images = len(self.datamanager.eval_dataset)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            for i in range(num_images):
                _, camera_ray_bundle, batch = self.datamanager.next_eval_image(idx)
                # for camera_ray_bundle, batch in self.datamanager.fixed_indices_eval_dataloader:
                # time this the following line
                inner_start = time()
                height, width = camera_ray_bundle.shape
                num_rays = height * width
                outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                metrics_dict, _ = self.model.get_image_metrics_and_images(outputs, batch)
                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = num_rays / (time() - inner_start)
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = metrics_dict["num_rays_per_sec"] / (height * width)
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
        # average the metrics list
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            metrics_dict[key] = float(
                torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
            )
        self.train()
        return metrics_dict

    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
            step: training step of the loaded checkpoint
        """
        state = {key.replace("module.", ""): value for key, value in loaded_state.items()}
        self._model.update_to_step(step)
        self.load_state_dict(state, strict=True)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        # TODO: requires careful check here: whether or not to distinguish dynamic callback from static one if they do exist
        """Returns the training callbacks from both the Dataloader and the Model."""
        datamanager_callbacks = self.datamanager.get_training_callbacks(training_callback_attributes)
        model_callbacks = self.model.get_training_callbacks(training_callback_attributes)
        callbacks = datamanager_callbacks + model_callbacks
        return callbacks

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the pipeline.

        Returns:
            A list of dictionaries containing the pipeline's param groups.
        """
        datamanager_params = self.datamanager.get_param_groups()
        model_params = self.model.get_param_groups()
        # TODO(ethan): assert that key names don't overlap
        return {**datamanager_params, **model_params}

    def tick(self):
        self.datamanager.tick(tick_n_frames=(self.datamanager.config.next_n_frames > 1))

    @property
    def cur_frame(self):
        return self.datamanager.train_dataset.cur_frame

    @property
    def num_dynamic_rays(self):
        assert self.datamanager.train_pixel_sampler is not None
        return self.datamanager.train_pixel_sampler.all_indices.size(0)

    @property
    def num_static_rays(self):
        assert self.datamanager.train_pixel_sampler is not None
        return self.datamanager.train_pixel_sampler.all_indices_inverted.size(0)

    def get_eval_last_frame(self):
        return torch.from_numpy(self.datamanager.eval_dataset.prev_frame_buffer).squeeze()

    def get_metric(self, image, rgb):
        psnr = self.model.psnr(image, rgb)
        ssim = self.model.ssim(image, rgb)
        lpips = self.model.lpips(image, rgb)

        return {"psnr": psnr, "ssim": ssim, "lpips": lpips}


@dataclass
class SpaceTimePipelineConfig(cfg.InstantiateConfig):
    _target: Type = field(default_factory=lambda: SpaceTimePipeline)
    datamanager: SpaceTimeDataManagerConfig = SpaceTimeDataManagerConfig()
    model: ModelConfig = ModelConfig()
    use_error_map_sampler: bool = False


class SpaceTimePipeline(Pipeline):
    def __init__(
        self,
        config: SpaceTimePipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
    ):
        super().__init__()
        self.config = config
        self.test_mode = test_mode
        self.datamanager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        self.datamanager.to(device)
        self.use_error_map_sampler = self.datamanager.config.train_sampler_type == "error_map"

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            # TODO: check: will num_train_data affect training steps
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.model.device

    def get_train_loss_dict(self, step: int):
        ray_bundle, batch = self.datamanager.next_train(step)
        # ray_bundle.times = batch["time"]
        model_outputs = self.model(ray_bundle)

        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        if self.use_error_map_sampler:
            self.datamanager.train_pixel_sampler.update(batch, loss_dict["rgb_loss_ray_wise"])
            del loss_dict["rgb_loss_ray_wise"]

        if "rgb_loss_ray_wise" in loss_dict:
            del loss_dict["rgb_loss_ray_wise"]

        return model_outputs, loss_dict, metrics_dict

    def get_eval_loss_dict(self, step: int):
        ray_bundle, batch = self.datamanager.next_eval(step)
        # ray_bundle.times = batch["time"]
        model_outputs = self.model(ray_bundle)

        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict

    # def get_eval_image_metrics_and_images(self, step: int):
    #     # TODO: eval several frame at one call
    #     self.eval()
    #     n_f = self.datamanager.eval_dataset.frames.size(1)
    #     frame_idx = random.sample([0, n_f // 2, n_f - 1], 1)[0]
    #     image_idx, camera_ray_bundle, batch = self.datamanager.next_eval_image(frame_idx)
    #     outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
    #     metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
    #     assert "image_idx" not in metrics_dict
    #     metrics_dict["image_idx"] = image_idx
    #     assert "num_rays" not in metrics_dict
    #     metrics_dict["num_rays"] = len(camera_ray_bundle)
    #     self.train()
    #     return metrics_dict, images_dict

    def get_eval_image_metrics_and_images(self, step: int, interval=10, use_fast=False):
        # TODO: eval several frame at one call
        if use_fast:
            return self.get_eval_image_metrics_and_images_fast(step, interval, thresh=0.9)
        self.eval()
        n_f = self.datamanager.eval_dataset.frames.size(1)
        frame_idxs = list(range(0, n_f, interval))

        metrics_dicts = []
        images_dicts = []
        masks = []
        for _i, frame_idx in enumerate(frame_idxs):
            if _i == 0:
                image_idx, camera_ray_bundle, batch = self.datamanager.next_eval_image(frame_idx)
            else:
                image_idx, camera_ray_bundle, batch = self.datamanager.next_eval_image_incremental(
                    frame_idx, camera_ray_bundle
                )
            outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)

            assert "image_idx" not in metrics_dict
            metrics_dict["image_idx"] = image_idx
            assert "num_rays" not in metrics_dict
            metrics_dict["num_rays"] = len(camera_ray_bundle)
            if "mask_val" in images_dict:
                masks.append(images_dict["mask_val"])
                del images_dict["mask_val"]
            metrics_dicts.append(metrics_dict)
            images_dicts.append(images_dict)

        mean_metrics_dict = {}
        total_metrics_dict = {}
        for k in ["lpips", "psnr", "ssim", "ssim2"]:
            total_metrics_dict[k] = [m[k] for m in metrics_dicts]
            print(total_metrics_dict[k])
            mean_metrics_dict[k] = np.mean(total_metrics_dict[k])
            wandb.log({f"eval/{k}_chart": get_chart(np.array(frame_idxs), np.array(total_metrics_dict[k]))}, step=step)

        if self.model.config.mask_type.startswith("global"):
            wandb.log({f"eval/mask_hist_field": wandb.Histogram(outputs["mask"].cpu().numpy())}, step=step)
            # wandb.log(
            #     {
            #         f"eval/mask_hist_proposal": wandb.Histogram(
            #             self.model.proposal_networks[0].mlp_base.mask_detached.cpu().numpy()
            #         )
            #     },
            #     step=step,
            # )

        display_frame_idx = random.sample(list(range(len(frame_idxs))), 1)[0]
        mean_metrics_dict["image_idx"] = metrics_dicts[display_frame_idx]["image_idx"]
        mean_metrics_dict["num_rays"] = metrics_dicts[display_frame_idx]["num_rays"]

        self.train()
        return mean_metrics_dict, images_dicts[display_frame_idx]

    def get_eval_image_metrics_and_images_fast(self, step: int, interval=10, thresh=0.9):
        # TODO: eval several frame at one call
        self.eval()
        n_f = self.datamanager.eval_dataset.frames.size(1)
        frame_idxs = list(range(0, n_f, interval))

        metrics_dicts = []
        images_dicts = []
        for _i, frame_idx in enumerate(frame_idxs):
            if _i == 0:
                image_idx, camera_ray_bundle, batch = self.datamanager.next_eval_image(frame_idx)
                outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                frame_one_outputs = outputs
                ray_mask = (outputs["mask"] > thresh).all(dim=-1).flatten()
                print("static_ratio:")
                print(ray_mask.sum() / ray_mask.numel())
            else:
                image_idx, camera_ray_bundle, batch = self.datamanager.next_eval_image_incremental(
                    frame_idx, camera_ray_bundle
                )
                outputs = self.model.get_outputs_for_camera_ray_bundle_incremental(
                    camera_ray_bundle, ray_mask, frame_one_outputs
                )

            metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
            assert "image_idx" not in metrics_dict
            metrics_dict["image_idx"] = image_idx
            assert "num_rays" not in metrics_dict
            metrics_dict["num_rays"] = len(camera_ray_bundle)
            metrics_dicts.append(metrics_dict)
            images_dicts.append(images_dict)

        mean_metrics_dict = {}
        total_metrics_dict = {}
        for k in ["lpips", "psnr", "ssim"]:
            total_metrics_dict[k] = [m[k] for m in metrics_dicts]
            print(total_metrics_dict[k])
            mean_metrics_dict[k] = np.mean(total_metrics_dict[k])
            if step is not None:
                wandb.log(
                    {f"eval/{k}_chart": get_chart(np.array(frame_idxs), np.array(total_metrics_dict[k]))}, step=step
                )

        if step is not None:
            if self.model.config.mask_type == "global":
                wandb.log(
                    {f"eval/mask_hist_field": wandb.Histogram(self.model.field.mlp_base.mask_detached.cpu().numpy())},
                    step=step,
                )
                wandb.log(
                    {
                        f"eval/mask_hist_proposal": wandb.Histogram(
                            self.model.proposal_networks[0].mlp_base.mask_detached.cpu().numpy()
                        )
                    },
                    step=step,
                )

        display_frame_idx = random.sample(list(range(len(frame_idxs))), 1)[0]
        mean_metrics_dict["image_idx"] = metrics_dicts[display_frame_idx]["image_idx"]
        mean_metrics_dict["num_rays"] = metrics_dicts[display_frame_idx]["num_rays"]

        self.train()
        return mean_metrics_dict, images_dicts[display_frame_idx]

    def get_average_eval_image_metrics(self, step: Optional[int] = None):
        num_frames = self.datamanager.eval_dataset.num_frames

    def get_eval_video(self, num_frames=None):
        self.eval()
        if num_frames is None:
            num_frames = self.datamanager.eval_dataset.num_frames
        frames = torch.zeros([num_frames] + list(self.datamanager.eval_dataset.frames.shape[2:]), dtype=torch.uint8)
        depth_frames = torch.zeros(
            [num_frames] + list(self.datamanager.eval_dataset.frames.shape[2:]), dtype=torch.uint8
        )
        for idx in trange(num_frames):
            _, camera_ray_bundle, batch = self.datamanager.next_eval_image(idx)
            assert hasattr(self.model, "render_one_image")
            with Timer("forward"):
                outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            frames[idx] = (outputs["rgb"] * 255).to(torch.uint8)
            depth_frames[idx] = (
                (colormaps.apply_depth_colormap(outputs["depth"], outputs["accumulation"]) * 255.0)
                .to("cpu")
                .to(torch.uint8)
            )

        fps = 30
        frames = frames.cpu().numpy().astype(np.uint8)
        depth_frames = depth_frames.cpu().numpy().astype(np.uint8)
        imageio.mimwrite(f"eval_{wandb.run.name}.mp4", frames, fps=fps, quality=10)
        imageio.mimwrite(f"eval_depth_{wandb.run.name}.mp4", depth_frames, fps=fps, quality=10)

        self.train()
        return metrics_dict, frames

    def render_from_cameras(
        self,
        near=1.0,
        far=5.0,
        num_frames=None,
        cameras=None,
        save_path=None,
        fps=None,
        offset=None,
        render_depth=True,
    ):
        self.eval()
        if num_frames is None:
            num_frames = self.datamanager.train_dataset.num_frames
        if cameras is None:
            if offset is None:
                cameras = get_render_cameras(
                    self.datamanager.train_dataset.cameras,
                    self.datamanager.eval_dataset.cameras,
                    near,
                    far,
                    n_frames=num_frames,
                    # rads_scale=1.0,
                    rads_scale=0.5,
                )
            else:
                cameras = get_render_cameras(
                    self.datamanager.train_dataset.cameras,
                    self.datamanager.eval_dataset.cameras,
                    near,
                    far,
                    n_frames=num_frames,
                    # rads_scale=1.0,
                    rads_scale=0.5,
                    offset=offset,
                )
        # frames = torch.zeros([num_frames] + list(self.datamanager.train_dataset.frames.shape[2:]), dtype=torch.uint8)
        frames = torch.zeros(
            [
                num_frames,
                cameras.height if isinstance(cameras.height, int) else cameras.height[0, 0].item(),
                cameras.width if isinstance(cameras.width, int) else cameras.width[0, 0].item(),
                3,
            ],
            dtype=torch.uint8,
        )
        depth_frames = torch.zeros(
            [
                num_frames,
                cameras.height if isinstance(cameras.height, int) else cameras.height[0, 0].item(),
                cameras.width if isinstance(cameras.width, int) else cameras.width[0, 0].item(),
                3,
            ],
            dtype=torch.uint8,
        )
        # [T, H, W, C]
        for t in trange(num_frames):
            ray_bundle = cameras.generate_rays(camera_indices=t, keep_shape=True).to("cuda")
            print(ray_bundle.shape)
            outputs = self.model.get_outputs_for_camera_ray_bundle(ray_bundle)
            frames[t] = (outputs["rgb"] * 255).to("cpu").to(torch.uint8)
            depth_frames[t] = (
                (colormaps.apply_depth_colormap(outputs["depth"], outputs["accumulation"]) * 255)
                .to("cpu")
                .to(torch.uint8)
            )

        # cv2.imwrite("test.png", frames[0].numpy())
        # torch.save(frames, "frames_v1.2.pt")

        if save_path is not None:
            if fps is None:
                fps = 30
            frames = frames.cpu().numpy().astype(np.uint8)
            depth_frames = depth_frames.cpu().numpy().astype(np.uint8)
            depth_video_path = save_path.parent / f"spiral_depth_{wandb.run.name}.mp4"
            imageio.mimwrite(save_path, frames, fps=fps, quality=10)
            imageio.mimwrite(depth_video_path, depth_frames, fps=fps, quality=10)

        # write_video(save_path, frames, fps=fps)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        datamanager_params = self.datamanager.get_param_groups()
        model_params = self.model.get_param_groups()
        # TODO(ethan): assert that key names don't overlap
        return {**datamanager_params, **model_params}

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        datamanager_callbacks = self.datamanager.get_training_callbacks(training_callback_attributes)
        model_callbacks = self.model.get_training_callbacks(training_callback_attributes)
        callbacks = datamanager_callbacks + model_callbacks
        return callbacks

    def mock_eval(self):
        print(self.datamanager.eval_dataset.cameras.camera_to_worlds)
        metric, image = self.get_eval_image_metrics_and_images(0)
        print(metric)

    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
            step: training step of the loaded checkpoint
        """
        state = {key.replace("module.", ""): value for key, value in loaded_state.items()}
        self._model.update_to_step(step)
        self.load_state_dict(state, strict=True)
