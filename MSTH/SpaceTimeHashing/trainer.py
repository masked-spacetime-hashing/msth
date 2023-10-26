from __future__ import annotations

import dataclasses
import functools
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from rich.console import Console
from torch.cuda.amp.grad_scaler import GradScaler
from typing_extensions import Literal

from nerfstudio.configs.experiment_config import ExperimentConfig
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.pipelines.base_pipeline import VanillaPipeline
from nerfstudio.utils import profiler, writer
from nerfstudio.utils.decorators import (
    check_eval_enabled,
    check_main_thread,
    check_viewer_enabled,
)
from nerfstudio.utils.misc import step_check
from nerfstudio.utils.writer import EventName, TimeWriter
from nerfstudio.viewer.server import viewer_utils
from MSTH.utils import Timer
import yappi

CONSOLE = Console(width=120)

TRAIN_INTERATION_OUTPUT = Tuple[  # pylint: disable=invalid-name
    torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]
]
TORCH_DEVICE = Union[torch.device, str]  # pylint: disable=invalid-name

from MSTH.video_pipeline import (
    VideoPipeline,
    VideoPipelineConfig,
    SpaceTimeDataManagerConfig,
    SpaceTimePipelineConfig,
    SpaceTimePipeline,
)

from nerfstudio.engine.trainer import Trainer, TrainerConfig
import wandb


@dataclass
class SpaceTimeHashingTrainerConfig(TrainerConfig):
    """Configuration for training regimen"""

    _target: Type = field(default_factory=lambda: SpaceTimeHashingTrainer)
    pipeline: SpaceTimePipelineConfig
    """target class to instantiate"""
    steps_per_save: int = 1000
    """Number of steps between saves."""
    steps_per_eval_batch: int = 500
    """Number of steps between randomly sampled batches of rays."""
    steps_per_eval_image: int = 2000
    """Number of steps between single eval images."""
    steps_per_eval_all_images: int = 25000
    """Number of steps between eval all images."""
    max_num_iterations: int = 1000000
    """Maximum number of iterations to run."""
    mixed_precision: bool = False
    """Whether or not to use mixed precision for training."""
    save_only_latest_checkpoint: bool = True
    """Whether to only save the latest checkpoint or all checkpoints."""
    # optional parameters if we want to resume training
    load_dir: Optional[Path] = None
    """Optionally specify a pre-trained model directory to load from."""
    load_step: Optional[int] = None
    """Optionally specify model step to load from; if none, will find most recent model in load_dir."""
    load_config: Optional[Path] = None
    """Path to config YAML file."""
    log_gradients: bool = False
    """Optionally log gradients during training"""

    wandb_name: str = "none"
    steps_full_video: int = 10000000000
    eval_total_frames: Optional[int] = None
    save_eval_video: bool = False

    render_camera_offset: Optional[List[float]] = None


class SpaceTimeHashingTrainer(Trainer):
    config: SpaceTimeHashingTrainerConfig
    pipeline: SpaceTimePipeline
    optimizers: Optimizers
    callbacks: List[TrainingCallback]

    @profiler.time_function
    def train_iteration(self, step: int) -> TRAIN_INTERATION_OUTPUT:
        """Run one iteration with a batch of inputs. Returns dictionary of model losses.

        Args:
            step: Current training step.
        """
        self.optimizers.zero_grad_all()
        self.pipeline.train()
        cpu_or_cuda_str: str = self.device.split(":")[0]
        with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision):
            _, loss_dict, metrics_dict = self.pipeline.get_train_loss_dict(step=step)
            loss = functools.reduce(torch.add, loss_dict.values())
        self.grad_scaler.scale(loss).backward()  # type: ignore
        # TODO remove this
        # self.pipeline.model.field.mlp_base.spatial_net.grad_total_variation(weight=1e-2, B=10000)
        # print(self.pipeline.get_param_groups()["proposal_networks"][0].grad)
        self.optimizers.optimizer_scaler_step_all(self.grad_scaler)

        if self.config.log_gradients:
            total_grad = 0
            for tag, value in self.pipeline.model.named_parameters():
                assert tag != "Total"
                if value.grad is not None:
                    grad = value.grad.norm()
                    metrics_dict[f"Gradients/{tag}"] = grad
                    total_grad += grad

            metrics_dict["Gradients/Total"] = total_grad

        self.grad_scaler.update()
        self.optimizers.scheduler_step_all(step)

        # Merging loss and metrics dict into a single output.
        return loss, loss_dict, metrics_dict

    @check_eval_enabled
    @profiler.time_function
    def eval_iteration(self, step: int) -> None:
        """Run one iteration with different batch/image/all image evaluations depending on step size.

        Args:
            step: Current training step.
        """
        # a batch of eval rays
        # if step_check(step, self.config.steps_per_eval_batch):
        #     _, eval_loss_dict, eval_metrics_dict = self.pipeline.get_eval_loss_dict(step=step)
        #     eval_loss = functools.reduce(torch.add, eval_loss_dict.values())
        #     writer.put_scalar(name="Eval Loss", scalar=eval_loss, step=step)
        #     writer.put_dict(name="Eval Loss Dict", scalar_dict=eval_loss_dict, step=step)
        #     writer.put_dict(name="Eval Metrics Dict", scalar_dict=eval_metrics_dict, step=step)

        # one eval image
        if step_check(step, self.config.steps_per_eval_image):
            with TimeWriter(writer, EventName.TEST_RAYS_PER_SEC, write=False) as test_t:
                metrics_dict, images_dict = self.pipeline.get_eval_image_metrics_and_images(step=step)
            writer.put_time(
                name=EventName.TEST_RAYS_PER_SEC,
                duration=metrics_dict["num_rays"] / test_t.duration,
                step=step,
                avg_over_steps=True,
            )
            writer.put_dict(name="Eval Images Metrics", scalar_dict=metrics_dict, step=step)
            group = "Eval Images"
            for image_name, image in images_dict.items():
                writer.put_image(name=group + "/" + image_name, image=image, step=step)

        if step_check(step, self.config.steps_full_video):
            metrics_dict, frames = self.pipeline.get_eval_video(
                step, self.config.eval_total_frames, self.config.save_eval_video
            )
            wandb.log({"eval_video": wandb.Video(frames)})
            writer.put_dict(name="Eval Full Video Metrics", scalar_dict=metrics_dict, step=step)

        # if step_check(step, 500):
        #     metrics_dict, images_dict = self.pipeline.get_eval_image_metrics_and_images(step=step)
        #     print(metrics_dict)
        #     self.pipeline.render_from_cameras(1.0, 5.0, save_path="/data/czl/tmp/test.mp4", fps=5, num_frames=5)
        #     exit(0)

        # all eval images
        # if step_check(step, self.config.steps_per_eval_all_images):
        #     metrics_dict = self.pipeline.get_average_eval_image_metrics(step=step)
        #     writer.put_dict(name="Eval Images Metrics Dict (all images)", scalar_dict=metrics_dict, step=step)

    def mock_eval(self):
        metrics_dict, _ = self.pipeline.get_eval_image_metrics_and_images(step=0)
        print(metrics_dict)
