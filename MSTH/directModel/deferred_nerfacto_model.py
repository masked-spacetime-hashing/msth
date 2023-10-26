# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NeRF implementation that combines many recent advancements.
"""
from __future__ import annotations
import pdb

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type, Union, Optional

import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.nerfacto_field import TCNNNerfactoField
from MSTH.directModel.deferred_nerfacto_field import TCNNDeferredNerfactoField
from MSTH.directModel.density_field_nomlp import HashDensityField

from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
)
from nerfstudio.model_components.ray_samplers import (
    ProposalNetworkSampler,
    UniformSampler,
    UniformLinDispPiecewiseSampler,
)
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    NormalsRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from MSTH.dataset import VideoDataset, VideoDatasetWithFeature, VideoDatasetAllCached


def sparsity_loss(density):
    return torch.mean(torch.log(1 + 2 * (density**2)))


@dataclass
class DeferredNerfactoModelConfig(ModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: DeferredNerfactoModel)
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""
    background_color: Literal["random", "last_sample", "black", "white"] = "last_sample"
    """Whether to randomize the background color."""
    hidden_dim: int = 64
    """Dimension of hidden layers"""
    hidden_dim_color: int = 64
    """Dimension of hidden layers for color network"""
    hidden_dim_transient: int = 64
    """Dimension of hidden layers for transient network"""
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {
                "hidden_dim": 16,
                "log2_hashmap_size": 17,
                "num_levels": 5,
                "max_res": 128,
                "use_linear": False,
                "base_res": 16,
                "gridtype": "hash",
            },
            {
                "hidden_dim": 16,
                "log2_hashmap_size": 17,
                "num_levels": 5,
                "max_res": 256,
                "use_linear": False,
                "base_res": 16,
                "gridtype": "hash",
            },
        ]
    )
    """Arguments for the proposal density fields."""
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    orientation_loss_mult: float = 0.0001
    """Orientation loss multiplier on computed normals."""
    pred_normal_loss_mult: float = 0.001
    """Predicted normal loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    predict_normals: bool = False
    """Whether to predict normals or not."""
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""

    """ feng add """
    use_appearance_embedding: bool = True

    gridtype: str = "hash"
    """ encoding grid type hash or tiled """
    base_res: Tuple[int, ...] = (16, 16, 16)
    max_res: Tuple[int, ...] = (2048, 2048, 2048)
    features_per_level: int = 2
    sparsity_loss_weight: float = 0.0
    """ /feng add"""

    """required by IBR"""
    train_cameras: Optional[Cameras] = None
    eval_cameras: Optional[Cameras] = None
    train_dataset: Optional[VideoDatasetWithFeature] = None
    device: Optional[str] = "cuda"


class DeferredNerfactoModel(Model):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: DeferredNerfactoModelConfig

    @torch.no_grad()
    def hash_reinitialize(self, ray_bundle, std):
        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        ray_samples = self.reinitial_uniform_sampler(ray_bundle, num_samples=256)
        """Set up proposal networks and field to restart training for the next frame"""
        self.field.hash_reinitialize(ray_samples, std)
        # for prop_net in self.proposal_networks:
        # prop_net.hash_reinitialize(ray_samples)
        # for proposal networks, sample the changing pixels, and re-init the value of HashMLP density grid

    @torch.no_grad()
    def set_static(self, ray_bundle):
        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        ray_samples = self.reinitial_uniform_sampler(ray_bundle, num_samples=512)

        self.field.set_static(ray_samples)

        # field_density, field_diffuse_color = self.field.get_density(ray_samples)
        # weights = ray_samples.get_weights(field_density)

    def reset_proposal_samplers(self):
        for network in self.proposal_networks:
            network.reset_parameters()

    def populate_modules(self):
        """Set the fields and modules."""
        self.disable_mlp = False
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        # Fields
        is_cube = np.unique(self.config.base_res).shape[0] == 1 and np.unique(self.config.max_res).shape[0] == 1
        self.field = TCNNDeferredNerfactoField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            use_appearance_embedding=self.config.use_appearance_embedding,
            gridtype=self.config.gridtype,
            base_res=self.config.base_res[0] if is_cube else self.config.base_res,
            max_res=self.config.max_res[0] if is_cube else self.config.max_res,
            features_per_level=self.config.features_per_level,
        )

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashDensityField(self.scene_box.aabb, spatial_distortion=scene_contraction, **prop_net_args)
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # Samplers
        update_schedule = lambda step: np.clip(
            np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
            1,
            self.config.proposal_update_every,
        )

        # Change proposal network initial sampler if uniform
        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)

        self.reinitial_uniform_sampler = UniformLinDispPiecewiseSampler(single_jitter=self.config.use_single_jitter)
        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        self.renderer_normals = NormalsRenderer()

        # shaders
        self.normals_shader = NormalsShader()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        print(list(self.field.encoding.parameters()))
        print(list(self.field.encoding.parameters())[0].shape)
        param_groups["fields_grid"] = (
            list(self.field.direction_encoding.parameters())
            + list(self.field.position_encoding.parameters())
            + list(self.field.encoding.parameters())
            + list(self.field.mlp_after_encoding.parameters())
        )

        if not self.disable_MLP:
            param_groups["fields_mlp"] = list(filter(lambda x: x.requires_grad, self.field.mlp_head.parameters()))

        param_groups["proposal_networks"] = list(filter(lambda x: x.requires_grad, self.proposal_networks.parameters()))

        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)
                bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)

        # field output
        # field_density, field_diffuse_color, field_density_embedding = self.field.get_density(ray_samples)
        field_density, field_diffuse_color = self.field.get_density(ray_samples)
        weights = ray_samples.get_weights(field_density)
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        # embedding integral
        intrinsic_rgb = self.renderer_rgb(rgb=field_diffuse_color, weights=weights)

        # field_embedding = self.renderer_rgb(rgb=field_density_embedding, weights=weights)
        # residual_rgb = self.field.get_outputs(ray_bundle, density_embedding=None, rgb_intrinsic=intrinsic_rgb)
        # rgb = (intrinsic_rgb + residual_rgb)/2.
        # rgb = intrinsic_rgb + residual_rgb
        rgb = intrinsic_rgb

        if not self.training:
            rgb = torch.nan_to_num(rgb)
            torch.clamp_(rgb, min=0.0, max=1.0)
        # rgb = residual_rgb

        # rgb = intrinsic_rgb
        outputs = {"rgb": rgb, "accumulation": accumulation, "depth": depth, "density": field_density}

        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        return outputs

    def disable_MLP(self):
        self.disable_mlp = True
        for parameter in self.field.mlp_head.parameters():
            parameter.requires_grad = False
        # for parameter in self.proposal_networks.parameters():
        # parameter.requires_grad = False

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        # image = image[...,0:1] * 0.3 + image[...,1:2]*0.59 + image[...,2:3]*0.11
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        # image = image[...,0:1] * 0.3 + image[...,1:2]*0.59 + image[...,2:3]*0.11
        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])
        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )

        if self.config.sparsity_loss_weight > 0:
            loss_dict["density_loss"] = self.config.sparsity_loss_weight * sparsity_loss(outputs["density"])
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)
        # image = image[...,0:1] * 0.3 + image[...,1:2]*0.59 + image[...,2:3]*0.11
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        # ssim = self.ssim(image, rgb)
        # lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item())}  # type: ignore
        # metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        # metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict
