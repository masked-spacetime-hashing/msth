import torch
import torch.nn as nn
import numpy as np
from typing import *
from nerfacc import ContractionType, contract
from torch.nn.parameter import Parameter
from torchtyping import TensorType
from dataclasses import dataclass, field

from nerfstudio.cameras.rays import RaySamples, Frustums, RayBundle
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.base_field import Field
from MSTH.gridencoder import GridEncoder
from MSTH.SpaceTimeHashing.ray_samplers import UniformSamplerSpatial
from nerfstudio.model_components.ray_samplers import (
    ProposalNetworkSampler,
    # UniformSampler,
)
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    NormalsRenderer,
    RGBRenderer,
)
from nerfstudio.field_components.spatial_distortions import (
    SceneContraction,
    SpatialDistortion,
)
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
)
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
import tinycudann as tcnn
from MSTH.SpaceTimeHashing.ray_samplers import ProposalNetworkSamplerSpatial, spacetime, spacetime_concat
from rich.console import Console
from MSTH.SpaceTimeHashing.stmodel_components import STModule

CONSOLE = Console(width=120)


def get_normalized_directions(directions: TensorType["bs":..., 3]) -> TensorType["bs":..., 3]:
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


class DSpaceTimeDensityFieldWithBase(Field):
    def __init__(
        self,
        aabb: TensorType,
        num_layers: int = 2,
        hidden_dim: int = 64,
        spatial_distortion: Optional[SpatialDistortion] = None,
        use_linear: bool = False,
        num_levels: int = 8,
        max_res: tuple = (1024, 1024, 1024, 300),
        base_res: tuple = (16, 16, 16, 4),
        log2_hashmap_size_spatial: int = 18,
        log2_hashmap_size_temporal: int = 18,
        features_per_level: int = 2,
        mode: str = "mst",
        mask_reso: tuple = (256, 256, 256),
        mask_log2_hash_size: int = 24,
    ) -> None:
        super().__init__()
        self.register_buffer("aabb", aabb)
        self.spatial_distortion = spatial_distortion
        self.use_linear = use_linear

        assert isinstance(base_res, (tuple, list))
        per_level_scale = (
            (1.0, 1.0, 1.0, 1.0)
            if num_levels == 1
            else tuple(np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1)))
        )
        CONSOLE.log(f"density grid per_level_scale: {per_level_scale}")

        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size_spatial", torch.tensor(log2_hashmap_size_spatial))
        self.register_buffer("log2_hashmap_size_temporal", torch.tensor(log2_hashmap_size_temporal))

        if not self.use_linear:
            self.mlp_base = STModule(
                1,
                num_levels,
                features_per_level,
                per_level_scale,
                base_res,
                log2_hashmap_size_spatial,
                log2_hashmap_size_temporal,
                hidden_dim,
                num_layers,
                mask_reso,
                mask_log2_hash_size,
                mode,
            )
        else:
            raise NotImplementedError

    # @torch.cuda.amp.autocast(enabled=False)
    def get_sparse_loss(self, mult_h=0.0001, mult_f=0.0001):
        # dh_loss = self.mlp_base.dh.abs().mean()
        dh_loss = torch.nn.L1Loss(size_average=None, reduce=None, reduction="mean")(
            self.mlp_base.dh, torch.zeros_like(self.mlp_base.dh)
        )
        # df_loss = self.mlp_base.df.abs().mean()
        df_loss = torch.nn.L1Loss(size_average=None, reduce=None, reduction="mean")(
            self.mlp_base.df, torch.zeros_like(self.mlp_base.df)
        )
        return mult_h * dh_loss.to(torch.float32) + mult_f * df_loss.to(torch.float32)

    def get_density(self, ray_samples: RaySamples) -> Tuple[TensorType, None]:
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(ray_samples.frustums.get_positions())
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        # note that now this stands for spacetime
        positions_flat = spacetime_concat(positions, ray_samples.times).view(-1, 4)
        # print(positions_flat)
        # positions_flat = positions.view(-1, 4)

        if not self.use_linear:
            density_before_activation = (
                self.mlp_base(positions_flat)[0].view(*ray_samples.frustums.shape, -1).to(positions)
            )
            density_before_activation_static = (
                self.mlp_base(positions_flat)[1].view(*ray_samples.frustums.shape, -1).to(positions)
            )

        else:
            raise NotImplementedError
            # x = self.encoding(positions_flat).to(positions)
            # density_before_activation = self.linear(x).view(*ray_samples.frustums.shape, -1)

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation)
        density = density * selector[..., None]

        density_static = trunc_exp(density_before_activation_static)
        density_static = density_static * selector[..., None]
        return density, None, density_static, None

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None) -> dict:
        return {}

    def density_fn(self, positions: TensorType["bs":..., 4]) -> TensorType["bs":..., 1]:
        pos = positions[..., :3]
        times = positions[..., 3:]
        ray_samples = RaySamples(
            frustums=Frustums(
                origins=pos,
                directions=torch.ones_like(pos),
                starts=torch.zeros_like(pos[..., :1]),
                ends=torch.zeros_like(pos[..., :1]),
                pixel_area=torch.ones_like(pos[..., :1]),
            ),
            times=times,
        )
        density, _, density_static, _ = self.get_density(ray_samples)
        return density, density_static


class SpaceTimeMixture(nn.Module):
    def __init__(self, spatial_net, temporal_net, temporal_prod_net, mode="mst"):
        """
        mode: mst or mt, mst is for m * s + (1-m) * t, mt is for s + (1-m) * t
        """
        super().__init__()
        self.spatial_net = spatial_net
        self.temporal_net = temporal_net
        self.temporal_prod_net = temporal_prod_net
        self.mode = mode

    def forward(self, x):
        # x should be B x 4
        spatial_component = self.spatial_net(x[..., :3])
        temporal_component = self.temporal_net(x)
        temporal_prod_component = self.temporal_prod_net(x[..., :3]).sigmoid()

        if self.mode == "mst":
            output = spatial_component * temporal_prod_component + (1 - temporal_prod_component) * temporal_component
        else:
            assert self.mode == "mt"
            output = spatial_component + (1 - temporal_prod_component) * temporal_component

        return (
            output,
            # spatial_component * temporal_prod_component + (1 - temporal_prod_component) * temporal_component,
            spatial_component,
        )


class DSpaceTimeHashingFieldWithBase(Field):
    def __init__(
        self,
        aabb: TensorType,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_layers_color: int = 3,
        hidden_dim_color: int = 64,
        use_appearance_embedding: Optional[bool] = False,
        num_images: Optional[int] = None,
        appearance_embedding_dim: int = 32,
        spatial_distortion: SpatialDistortion = None,
        num_levels: int = 16,
        log2_hashmap_size_spatial: int = 19,
        log2_hashmap_size_temporal: int = 19,
        base_res: tuple = (16, 16, 16, 4),
        max_res: tuple = (2048, 2048, 2048, 300),
        mask_reso: tuple = (256, 256, 256),
        mask_log2_hash_size: int = 24,
        mode: str = "mst",
    ) -> None:
        super().__init__()
        self.aabb = Parameter(aabb, requires_grad=False)
        self.geo_feat_dim = geo_feat_dim
        self.use_appearance_embedding = use_appearance_embedding
        self.appearance_embedding_dim = appearance_embedding_dim
        # self.contraction_type = contraction_type
        self.spatial_distortion = spatial_distortion

        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        if isinstance(base_res, (tuple, list)):
            per_level_scale = (
                (1.0, 1.0, 1.0, 1.0)
                if num_levels == 1
                else tuple(np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1)))
            )

        features_per_level = 2
        self.mlp_base = STModule(
            1 + self.geo_feat_dim,
            num_levels,
            features_per_level,
            per_level_scale,
            base_res,
            log2_hashmap_size_spatial,
            log2_hashmap_size_temporal,
            hidden_dim,
            num_layers,
            mask_reso,
            mask_log2_hash_size,
            mode,
        )

        in_dim = self.direction_encoding.n_output_dims + self.geo_feat_dim
        if self.use_appearance_embedding:
            in_dim += self.appearance_embedding_dim
        self.mlp_head = tcnn.Network(
            n_input_dims=in_dim,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            },
        )

    # @torch.cuda.amp.autocast(enabled=False)
    def get_sparse_loss(self, mult_h=0.0001, mult_f=0.0001):
        # dh_loss = self.mlp_base.dh.abs().mean()
        dh_loss = torch.nn.L1Loss(size_average=None, reduce=None, reduction="mean")(
            self.mlp_base.dh, torch.zeros_like(self.mlp_base.dh)
        )
        print(dh_loss)
        # df_loss = self.mlp_base.df.abs().mean()
        df_loss = torch.nn.L1Loss(size_average=None, reduce=None, reduction="mean")(
            self.mlp_base.df, torch.zeros_like(self.mlp_base.df)
        )
        print(df_loss)
        return mult_h * dh_loss.to(torch.float32) + mult_f * df_loss.to(torch.float32)

    def get_density(self, ray_samples: RaySamples) -> Tuple[TensorType, TensorType]:
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        # [num_rays_per_batch, num_samples, 3]

        # spacetime = torch.cat([positions_flat, times], dim=-1)
        st = spacetime_concat(positions, ray_samples.times).view(-1, 4)

        _spatial_temporal_output = self.mlp_base(st)
        h = _spatial_temporal_output[0].view(*ray_samples.frustums.shape, -1)
        static_h = _spatial_temporal_output[1].view(*ray_samples.frustums.shape, -1)

        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)

        density_before_activation_static, base_mlp_out_static = torch.split(static_h, [1, self.geo_feat_dim], dim=-1)

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(positions))
        density_static = trunc_exp(density_before_activation_static.to(positions))
        return density, base_mlp_out, density_static, base_mlp_out_static

    def get_outputs(
        self,
        ray_samples: RaySamples,
        density_embedding: Optional[TensorType] = None,
        density_embedding_static: Optional[TensorType] = None,
    ) -> Dict[FieldHeadNames, TensorType]:
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)

        d = self.direction_encoding(directions_flat)
        if density_embedding is None:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
            h = torch.cat([d, positions.view(-1, 3)], dim=-1)
        else:
            h = torch.cat([d, density_embedding.view(-1, self.geo_feat_dim)], dim=-1)

        if density_embedding_static is not None:
            static_h = torch.cat([d, density_embedding_static.view(-1, self.geo_feat_dim)], dim=-1)
        else:
            raise NotImplementedError

        # if self.use_appearance_embedding:
        #     if ray_samples.camera_indices is None:
        #         raise AttributeError("Camera indices are not provided.")
        #     camera_indices = ray_samples.camera_indices.squeeze()
        #     if self.training:
        #         embedded_appearance = self.appearance_embedding(camera_indices)
        #     else:
        #         embedded_appearance = torch.zeros(
        #             (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
        #         )
        #     h = torch.cat([h, embedded_appearance.view(-1, self.appearance_embedding_dim)], dim=-1)

        rgb = self.mlp_head(h).view(*ray_samples.frustums.directions.shape[:-1], -1).to(directions)
        rgb_static = self.mlp_head(static_h).view(*ray_samples.frustums.directions.shape[:-1], -1).to(directions)
        return {FieldHeadNames.RGB: rgb, "rgb_static": rgb_static}

    def density_fn(self, positions: TensorType["bs":..., 4]) -> TensorType["bs":..., 1]:
        pos = positions[..., :3]
        times = positions[..., 3:]
        ray_samples = RaySamples(
            frustums=Frustums(
                origins=pos,
                directions=torch.ones_like(pos),
                starts=torch.zeros_like(pos[..., :1]),
                ends=torch.zeros_like(pos[..., :1]),
                pixel_area=torch.ones_like(pos[..., :1]),
            ),
            times=times,
        )
        density, _, _, _ = self.get_density(ray_samples)
        return density

    def forward(self, ray_samples: RaySamples, compute_normals: bool = False) -> Dict[FieldHeadNames, TensorType]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        if compute_normals:
            with torch.enable_grad():
                density, density_embedding = self.get_density(ray_samples)
        else:
            density, density_embedding, static_density, static_density_embedding = self.get_density(ray_samples)

        field_outputs = self.get_outputs(
            ray_samples, density_embedding=density_embedding, density_embedding_static=static_density_embedding
        )
        field_outputs[FieldHeadNames.DENSITY] = density  # type: ignore
        field_outputs["density_static"] = static_density  # type: ignore

        if compute_normals:
            with torch.enable_grad():
                normals = self.get_normals()
            field_outputs[FieldHeadNames.NORMALS] = normals  # type: ignore
        return field_outputs


@dataclass
class DSpaceTimeHashingModelConfig(ModelConfig):
    _target: Type = field(default_factory=lambda: DSpaceTimeHashingModel)
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
    max_res: tuple = (2048, 2048, 2048, 300)
    base_res: tuple = (16, 16, 16, 4)
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size_spatial: int = 19
    log2_hashmap_size_temporal: int = 19
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
                "log2_hashmap_size_spatial": 17,
                "log2_hashmap_size_temporal": 17,
                "num_levels": 5,
                "max_res": (128, 128, 128, 150),
                "base_res": (16, 16, 16, 4),
                "use_linear": False,
                "mode": "mst",
                "mask_reso": (256, 256, 256),
                "mask_log2_hash_size": 24,
            },
            {
                "hidden_dim": 16,
                "log2_hashmap_size_spatial": 17,
                "log2_hashmap_size_temporal": 17,
                "num_levels": 5,
                "max_res": (256, 256, 256, 300),
                "base_res": (16, 16, 16, 4),
                "use_linear": False,
                "mode": "mst",
                "mask_reso": (256, 256, 256),
                "mask_log2_hash_size": 24,
            },
        ]
    )
    """Arguments for the proposal density fields."""
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    # distortion_loss_mult: float = 100.0
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
    use_appearance_embedding: bool = False
    """ /feng add"""

    sparse_loss_mult_h: float = 0.0
    sparse_loss_mult_f: float = 0.0

    mask_loss_mult: float = 0.0
    mst_mode: Literal["mst", "mt"] = "mst"

    mask_reso: tuple = (256, 256, 256)
    mask_log2_hash_size: int = 24


class DSpaceTimeHashingModel(Model):
    config: DSpaceTimeHashingModelConfig

    def populate_modules(self):
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        self.field = DSpaceTimeHashingFieldWithBase(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            log2_hashmap_size_spatial=self.config.log2_hashmap_size_spatial,
            log2_hashmap_size_temporal=self.config.log2_hashmap_size_temporal,
            hidden_dim_color=self.config.hidden_dim_color,
            num_images=self.num_train_data,
            spatial_distortion=scene_contraction,
            mode=self.config.mst_mode,
            mask_reso=self.config.mask_reso,
            mask_log2_hash_size=self.config.mask_log2_hash_size,
        )

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        sampler_class = DSpaceTimeDensityFieldWithBase
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = sampler_class(self.scene_box.aabb, spatial_distortion=scene_contraction, **prop_net_args)
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = sampler_class(
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

        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSamplerSpatial(single_jitter=self.config.use_single_jitter)

        self.proposal_sampler = ProposalNetworkSamplerSpatial(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

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

        # debugging
        # print(self.get_param_groups()["proposal_networks"])

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
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
        ray_samples, weights_list, ray_samples_list, weights_list_static = self.proposal_sampler(
            ray_bundle, density_fns=self.density_fns
        )
        field_outputs = self.field(ray_samples)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)
        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)

        weights_static = ray_samples.get_weights(field_outputs["density_static"])
        weights_list_static.append(weights_static)
        rgb_static = self.renderer_rgb(rgb=field_outputs["rgb_static"], weights=weights_static)

        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }
        outputs.update({"rgb_static": rgb_static})

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["weights_list_static"] = weights_list_static
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            # CONSOLE.print(i, weights_list[i].max())
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        # CONSOLE.print("final", weights_list[-1].max())

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])
        if self.training:
            if "is_static" in batch:
                loss_dict["rgb_static_loss"] = self.rgb_loss(
                    image[batch["is_static"]], outputs["rgb_static"][batch["is_static"]]
                )
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )

            #
            if self.config.mask_loss_mult > 0.0:
                loss_dict["mask_loss"] = (
                    self.field.mlp_base.temporal_prod_net.get_mask_loss() * self.config.mask_loss_mult
                )
                for net in self.proposal_networks:
                    loss_dict["mask_loss"] += (
                        net.mlp_base.temporal_prod_net.get_mask_loss() * self.config.mask_loss_mult
                    )

            # + self.config.interlevel_loss_mult * interlevel_loss(
            #     outputs["weights_list_static"], outputs["ray_samples_list"]
            # )
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
            if self.config.sparse_loss_mult_f > 0 or self.config.sparse_loss_mult_h > 0:
                loss_dict["sparse_loss"] = self.field.get_sparse_loss(
                    self.config.sparse_loss_mult_h, self.config.sparse_loss_mult_f
                )
                for _m in self.proposal_networks:
                    loss_dict["sparse_loss"] += _m.get_sparse_loss(
                        self.config.sparse_loss_mult_h, self.config.sparse_loss_mult_f
                    )
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        rgb_static = outputs["rgb_static"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([image, rgb, rgb_static], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict


# class
