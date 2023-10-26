import torch
import torch.nn as nn
import numpy as np
import time
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
from MSTH.SpaceTimeHashing.ray_samplers import UniformSamplerSpatial, UniformLinDispSamplerSpatial
from MSTH.SpaceTimeHashing.mle import MLELoss
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
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.functional import structural_similarity_index_measure
from skimage.metrics import structural_similarity as sk_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.model_components.scene_colliders import SceneCollider
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
from MSTH.SpaceTimeHashing.ray_samplers import (
    ProposalNetworkSamplerSpatial,
    spacetime,
    spacetime_concat,
    PDFSamplerSpatial,
    UniformLinDispPiecewiseSamplerSpatial,
)
from rich.console import Console
from MSTH.SpaceTimeHashing.stmodel_components import (
    STModule,
    STModuleHierarchicay,
    STModuleFuse,
    SampleNetwork,
    STModuleML,
    STModuleTemporalMask,
    STModuleWithTimeQuery,
)
from collections import defaultdict
from MSTH.utils import Timer, gen_ndc_rays, sparse_loss, print_value_range, gen_ndc_rays_looking_at_world_z
from functools import partial

CONSOLE = Console(width=120)


class NearFarCollider(SceneCollider):
    """Sets the nears and fars with fixed values.

    Args:
        near_plane: distance to near plane
        far_plane: distance to far plane
    """

    def __init__(self, near_plane: float, far_plane: float, **kwargs) -> None:
        self.near_plane = near_plane
        self.far_plane = far_plane
        super().__init__(**kwargs)

    def set_nears_and_fars(self, ray_bundle: RayBundle, valnear=0.5) -> RayBundle:
        ones = torch.ones_like(ray_bundle.origins[..., 0:1])
        # near_plane = self.near_plane if self.training else 0.5
        # TODO
        near_plane = self.near_plane if self.training else valnear
        # near_plane = self.near_plane
        ray_bundle.nears = ones * near_plane
        ray_bundle.fars = ones * self.far_plane
        return ray_bundle


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
        mask_type: str = "global",
        st_mlp_mode: str = "independent",
        spatial_only: bool = False,
        interp="linear",
        nosigmoid=False,
        ablation_add=False,
        mask_init_mean=0.0,
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

        if mask_type == "global":
            STM = STModule
        elif mask_type == "global_fuse":
            STM = STModuleFuse
        elif mask_type == "global_multilevel":
            STM = STModuleML
        elif mask_type == "global_multitime":
            STM = STModuleTemporalMask
        elif mask_type == "global_timequery":
            STM = STModuleWithTimeQuery
        else:
            assert mask_type == "hierarchical"
            STM = STModuleHierarchicay
        self.mlp_base = STM(
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
            st_mlp_mode,
            use_linear=self.use_linear,
            spatial_only=spatial_only,
            interp=interp,
            nosigmoid=nosigmoid,
            ablation_add=ablation_add,
            mask_init_mean=mask_init_mean,
        )

    def get_density(self, ray_samples: RaySamples, get_static_one=False) -> Tuple[TensorType, None]:
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

        mlp_out = self.mlp_base(positions_flat)
        density_before_activation = mlp_out[0].view(*ray_samples.frustums.shape, -1).to(positions)
        if get_static_one:
            density_before_activation_static = mlp_out[1].view(*ray_samples.frustums.shape, -1).to(positions)

            # x = self.encoding(positions_flat).to(positions)
            # density_before_activation = self.linear(x).view(*ray_samples.frustums.shape, -1)

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation)
        density = density * selector[..., None]

        if get_static_one:
            density_static = trunc_exp(density_before_activation_static)
            density_static = density_static * selector[..., None]
        else:
            density_static = None
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
        density, _, density_static, _ = self.get_density(ray_samples, get_static_one=False)
        return density, density_static


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
        appearance_embedding_dim: int = 16,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: SpatialDistortion = None,
        num_levels: int = 16,
        log2_hashmap_size_spatial: int = 19,
        log2_hashmap_size_temporal: int = 19,
        base_res: tuple = (16, 16, 16, 4),
        max_res: tuple = (2048, 2048, 2048, 300),
        mask_reso: tuple = (256, 256, 256),
        mask_log2_hash_size: int = 24,
        mode: str = "mst",
        mask_type: str = "global",
        st_mlp_mode="independent",
        use_linear_for_collision=False,
        interp="linear",
        level_one_interp="linear",
        nosigmoid=False,
        ablation_add=False,
        mask_init_mean=0.0,
        customzied_t_reso=None,
    ) -> None:
        super().__init__()
        self.aabb = Parameter(aabb, requires_grad=False)
        self.geo_feat_dim = geo_feat_dim
        self.use_appearance_embedding = use_appearance_embedding
        self.appearance_embedding_dim = appearance_embedding_dim
        # self.contraction_type = contraction_type
        self.spatial_distortion = spatial_distortion
        self.num_images = num_images
        self.use_average_appearance_embedding = use_average_appearance_embedding

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
        if mask_type == "global":
            STM = STModule
        elif mask_type == "global_fuse":
            STM = STModuleFuse
        elif mask_type == "global_multilevel":
            STM = STModuleML
        elif mask_type == "global_multitime":
            STM = STModuleTemporalMask
        elif mask_type == "global_timequery":
            STM = STModuleWithTimeQuery
        else:
            assert mask_type == "hierarchical"
            STM = STModuleHierarchicay
        self.mlp_base = STM(
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
            st_mlp_mode,
            use_linear=use_linear_for_collision,
            interp=interp,
            level_one_interp=level_one_interp,
            nosigmoid=nosigmoid,
            ablation_add=ablation_add,
            mask_init_mean=mask_init_mean,
            customized_t_reso=customzied_t_reso,
        )

        in_dim = self.direction_encoding.n_output_dims + self.geo_feat_dim
        if self.use_appearance_embedding:
            in_dim += self.appearance_embedding_dim
            self.embedding_appearance = Embedding(self.num_images, self.appearance_embedding_dim)
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

    def get_density(
        self, ray_samples: RaySamples, get_static_one=False, temporal_only=False
    ) -> Tuple[TensorType, TensorType]:
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        # [num_rays_per_batch, num_samples, 3]

        # spacetime = torch.cat([positions_flat, times], dim=-1)
        st = spacetime_concat(positions, ray_samples.times).view(-1, 4)

        _spatial_temporal_output = self.mlp_base(st, temporal_only=temporal_only)
        h = _spatial_temporal_output[0].view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        density = trunc_exp(density_before_activation.to(positions))

        if get_static_one:
            static_h = _spatial_temporal_output[1].view(*ray_samples.frustums.shape, -1)
            density_before_activation_static, base_mlp_out_static = torch.split(
                static_h, [1, self.geo_feat_dim], dim=-1
            )
            density_static = trunc_exp(density_before_activation_static.to(positions))
        else:
            density_static = None
            base_mlp_out_static = None

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        return density, base_mlp_out, density_static, base_mlp_out_static, _spatial_temporal_output[2]

    @torch.no_grad()
    def get_mask_density(self, ray_samples: RaySamples):
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(ray_samples.frustums.get_positions())
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        # note that now this stands for spacetime
        # positions_flat = spacetime_concat(positions, ray_samples.times).view(-1, 4)
        # print(positions_flat)
        # positions_flat = positions.view(-1, 4)

        return 1 - torch.nn.functional.sigmoid(self.mlp_base.temporal_prod_net(positions))

    @torch.no_grad()
    def get_mask_color(self, ray_samples: RaySamples):
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(ray_samples.frustums.get_positions())
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        # note that now this stands for spacetime
        # positions_flat = spacetime_concat(positions, ray_samples.times).view(-1, 4)
        # print(positions_flat)
        # positions_flat = positions.view(-1, 4)

        return (1 - torch.nn.functional.sigmoid(self.mlp_base.temporal_prod_net(positions))).repeat(1, 1, 3)

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
            static_h = None

        if self.use_appearance_embedding:
            if ray_samples.camera_indices is None:
                raise AttributeError("Camera indices are not provided.")
            camera_indices = ray_samples.camera_indices.squeeze()
            if self.training:
                embedded_appearance = self.embedding_appearance(camera_indices)
            else:
                if self.use_average_appearance_embedding:
                    embedded_appearance = torch.ones(
                        (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                    ) * self.embedding_appearance.mean(dim=0)
                else:
                    embedded_appearance = torch.zeros(
                        (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                    )
            h = torch.cat([h, embedded_appearance.view(-1, self.appearance_embedding_dim)], dim=-1)

        rgb = self.mlp_head(h).view(*ray_samples.frustums.directions.shape[:-1], -1).to(directions)

        if static_h is not None:
            rgb_static = self.mlp_head(static_h).view(*ray_samples.frustums.directions.shape[:-1], -1).to(directions)
        else:
            rgb_static = None
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
        density, _, _, _, _ = self.get_density(ray_samples)
        return density

    def forward(
        self,
        ray_samples: RaySamples,
        compute_normals: bool = False,
        get_static_one=False,
        temporal_only=False,
    ) -> Dict[FieldHeadNames, TensorType]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        if compute_normals:
            with torch.enable_grad():
                density, density_embedding = self.get_density(ray_samples, temporal_only)
        else:
            density, density_embedding, static_density, static_density_embedding, stmask = self.get_density(
                ray_samples, get_static_one=get_static_one
            )

        field_outputs = self.get_outputs(
            ray_samples, density_embedding=density_embedding, density_embedding_static=static_density_embedding
        )
        field_outputs[FieldHeadNames.DENSITY] = density  # type: ignore
        field_outputs["stmask"] = stmask
        field_outputs["density_static"] = static_density  # type: ignore

        if compute_normals:
            with torch.enable_grad():
                normals = self.get_normals()
            field_outputs[FieldHeadNames.NORMALS] = normals  # type: ignore
        return field_outputs


@dataclass
class DSpaceTimeHashingModelConfig(ModelConfig):
    _target: Type = field(default_factory=lambda: DSpaceTimeHashingModel)
    num_layers_color: int = 3
    num_layers: int = 2
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
    max_res_spatial: tuple = (2048, 2048, 2048)
    base_res_spatial: tuple = (16, 16, 16)
    max_res: tuple = (2048, 2048, 2048, 300)
    base_res: tuple = (16, 16, 16, 15)
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
                "mask_type": "global",
                "st_mlp_mode": "independent",
                "interp": "linear",
                "nosigmoid": False,
                "mask_init_mean": 0.0,
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
                "mask_type": "global",
                "st_mlp_mode": "independent",
                "interp": "linear",
                "nosigmoid": False,
                "mask_init_mean": 0.0,
            },
        ]
    )
    """Arguments for the proposal density fields."""
    proposal_initial_sampler: Literal["piecewise", "uniform", "linspace"] = "piecewise"
    """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    distortion_loss_mult_end: float = 0.002
    distortion_loss_mult_decay: int = 40000
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
    field_num_layers: int = 2

    mask_loss_mult: float = 0.0
    mask_loss_for_proposal: bool = True
    mst_mode: Literal["mst", "mt"] = "mst"

    mask_reso: tuple = (256, 256, 256)
    mask_log2_hash_size: int = 24
    mask_type: str = "global"

    freeze_mask: bool = False
    freeze_mask_step: Optional[int] = None

    render_static: bool = False
    use_loss_static: bool = False
    st_mlp_mode: str = "independent"
    use_sample_network: bool = False
    sample_network_hidden_dim: int = 64
    sample_network_num_layers: int = 2
    sample_network_loss_mult: float = 1.0
    use_linear_for_collision: bool = False
    debug: bool = False
    interp: str = "linear"
    level_one_interp: str = "linear"
    nosigmoid: bool = False
    ablation_add: bool = False
    mask_init_mean: float = 0.0
    contraction_type: str = "inf"
    dist_sharpen: float = 1.0
    middle_distance: float = 1.0

    sparse_loss_mult: float = 0.0
    sparse_loss_mult_end: float = 0.0
    sparse_loss_mult_decay: int = 40000
    use_no_prior_mask: bool = True

    freeze_spatial_steps: int = 100000000

    # cnn for patch
    use_perceptual_loss: bool = False
    perceptual_fields: int = 16

    appearance_embedding_dim: int = 16

    disable_mask_loss_for_dynamic_rays: bool = False
    get_mask_val: bool = False

    customized_t_reso: Optional[Callable] = None
    use_error_map_sampler: bool = False

    use_ndc: bool = False


class DSpaceTimeHashingModel(Model):
    config: DSpaceTimeHashingModelConfig

    def populate_modules(self):
        super().populate_modules()
        self.step = 0

        # a workaround for video viewer
        self.temporal_distortion = True

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            if self.config.contraction_type == "inf":
                scene_contraction = SceneContraction(order=float("inf"))
            else:
                assert self.config.contraction_type == "l2"
                scene_contraction = SceneContraction(order=2)

        if self.config.use_ndc:
            assert "ndc_coeffs" in self.kwargs["metadata"]
            self.ndc_coeffs = self.kwargs["metadata"]["ndc_coeffs"]
            self.use_llff_poses = self.kwargs["metadata"]["use_llff_poses"]
            self.ndc_near = self.kwargs["metadata"]["ndc_near"]
            self.config.near_plane = 0.0
            self.config.far_plane = 1.0
            if "ndc_far_plane" in self.kwargs["metadata"]:
                self.config.far_plane = self.kwargs["metadata"]["ndc_far_plane"]
            if "ndc_near_plane" in self.kwargs["metadata"]:
                self.config.near_plane = self.kwargs["metadata"]["ndc_near_plane"]
            if "orientation" in self.kwargs["metadata"]:
                self.orientation = self.kwargs["metadata"]["orientation"]
            scene_contraction = None
            CONSOLE.log(f"NDC coeffs: {self.ndc_coeffs}")
            CONSOLE.log("set near plane to 0 and far plane to 1 since NDC is enabled")
            if self.use_llff_poses:
                CONSOLE.log("Using llff poses")
            ndc_fn = gen_ndc_rays if not self.orientation == "up" else gen_ndc_rays_looking_at_world_z
            self.convert_to_ndc = partial(ndc_fn, ndc_coeffs=self.ndc_coeffs, near=self.ndc_near, normalize_dir=False)

        self.field = DSpaceTimeHashingFieldWithBase(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_layers_color=self.config.num_layers_color,
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
            mask_type=self.config.mask_type,
            st_mlp_mode=self.config.st_mlp_mode,
            use_linear_for_collision=self.config.use_linear_for_collision,
            interp=self.config.interp,
            level_one_interp=self.config.level_one_interp,
            nosigmoid=self.config.nosigmoid,
            ablation_add=self.config.ablation_add,
            mask_init_mean=self.config.mask_init_mean,
            use_appearance_embedding=self.config.use_appearance_embedding,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=self.config.appearance_embedding_dim,
            customzied_t_reso=self.config.customized_t_reso,
        )

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        sampler_class = DSpaceTimeDensityFieldWithBase
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_sample_network:
            self.sample_network = SampleNetwork(
                hidden_dim=self.config.sample_network_hidden_dim, num_layers=self.config.sample_network_num_layers
            )
            self.sn_initial_sampler = UniformLinDispPiecewiseSamplerSpatial(single_jitter=self.config.use_single_jitter)
            self.sn_pdf_sampler = PDFSamplerSpatial(include_original=False, single_jitter=self.config.use_single_jitter)
            self.mle_loss = MLELoss()

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
        if self.config.use_ndc:
            initial_sampler = UniformSamplerSpatial(single_jitter=self.config.use_single_jitter)
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSamplerSpatial(single_jitter=self.config.use_single_jitter)
        elif self.config.proposal_initial_sampler == "linspace":
            initial_sampler = UniformLinDispSamplerSpatial(single_jitter=self.config.use_single_jitter)

        self.proposal_sampler = ProposalNetworkSamplerSpatial(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
            middle_distance=self.config.middle_distance,
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
        self.ssim2 = StructuralSimilarityIndexMeasure(
            data_range=2.0
        )  # 2.0 is not correct, but is adopted by some prior works, for fair comparisons, we also measure this metrics.

        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        # debugging
        # print(self.get_param_groups()["proposal_networks"])

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.config.use_sample_network:
            param_groups["proposal_networks"] = list(self.proposal_networks.parameters()) + list(
                self.sample_network.parameters()
            )
        else:
            param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        # param_groups["sample_networks"] = list(self.proposal_networks.parameters())
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

        if self.config.freeze_mask:
            assert self.config.freeze_mask_step is not None

            def freeze(step):
                if step == self.config.freeze_mask_step:
                    self.freeze_temporal_prod()

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=freeze,
                )
            )
        return callbacks

    def get_outputs(self, ray_bundle: RayBundle, temporal_only=False):
        if self.config.use_ndc:
            self.convert_to_ndc(ray_bundle)
            print_value_range(ray_bundle.origins)
            print_value_range(ray_bundle.origins + ray_bundle.directions)

        if self.config.use_sample_network:
            if self.training:
                self.step += 1
                assert self.config.num_proposal_iterations == 1
                initial_samples = ray_samples_list[0]
                last_weight = self.proposal_sampler.last_weight
                if self.step >= self.config.freeze_spatial_steps:
                    self.freeze_static()
            else:
                initial_samples = self.sn_initial_sampler(
                    ray_bundle, num_samples=self.config.num_proposal_samples_per_ray[0]
                )

            sn_weights = self.sample_network(ray_bundle.origins, ray_bundle.directions, ray_bundle.times)

            # sn_weights = self.sample_network.get_pdf_at(
            # (initial_samples.frustums.starts + initial_samples.frustums.ends) / 2, sample_dist_mean, sample_dist_std
            # )
            # generate weights

            # if self.step > 8000:
            sn_ray_samples = self.sn_pdf_sampler(
                ray_bundle,
                initial_samples,
                sn_weights.unsqueeze(dim=-1),
                num_samples=self.config.num_nerf_samples_per_ray,
            )
            field_outputs = self.field(sn_ray_samples, get_static_one=self.config.render_static)
            weights = sn_ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

            # ray_samples_list.append(sn_ray_samples)
            ray_samples = sn_ray_samples
            # else:
            # field_outputs = self.field(ray_samples, get_static_one=self.config.render_static)
            # weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
            # ray_samples_list.append(ray_samples)
        else:
            with Timer(des="first"):
                ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
                    ray_bundle, density_fns=self.density_fns
                )
            with Timer(des="second"):
                field_outputs = self.field(
                    ray_samples,
                    get_static_one=self.config.render_static,
                    temporal_only=temporal_only,
                )
                weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

                if self.config.dist_sharpen != 1.0:
                    weights = weights**self.config.dist_sharpen
                    weights = weights / weights.sum(dim=-2, keepdim=True)

            ray_samples_list.append(ray_samples)

            weights_list.append(weights)

        with Timer(des="renderrgb"):
            rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)

        # weights_list_static.append(weights_static)
        if self.config.render_static:
            weights_static = ray_samples.get_weights(field_outputs["density_static"])
            rgb_static = self.renderer_rgb(rgb=field_outputs["rgb_static"], weights=weights_static)
        else:
            rgb_static = None

        if self.config.debug:
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
            accumulation = self.renderer_accumulation(weights=weights)

            outputs = {
                "rgb": rgb,
                "accumulation": accumulation,
                "depth": depth,
            }
        else:
            outputs = {
                "rgb": rgb,
            }

        if self.config.sparse_loss_mult > 0.0:
            outputs.update({"density": field_outputs[FieldHeadNames.DENSITY]})

        with Timer(des="dictupdate"):
            outputs.update({"rgb_static": rgb_static})

        if not self.training and self.config.mask_type.startswith("global"):
            outputs.update({"mask": self.field.mlp_base.mask_detached})

        if self.config.use_sample_network:
            outputs.update({"sn_weights": sn_weights})
            if self.training:
                outputs.update({"last_weight": last_weight.squeeze()})

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            # outputs["weights_list_static"] = weights_list_static
            outputs["ray_samples_list"] = ray_samples_list
            outputs["stmask"] = field_outputs["stmask"]

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        if self.config.debug:
            for i in range(self.config.num_proposal_iterations):
                # CONSOLE.print(i, weights_list[i].max())
                outputs[f"prop_depth_{i}"] = self.renderer_depth(
                    weights=weights_list[i], ray_samples=ray_samples_list[i]
                )
        # outputs[f"prop_depth_sn"] = self.renderer_depth(weights=sn_weights, ray_samples=sn_ray_samples)

        # CONSOLE.print("final", weights_list[-1].max())

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        if self.training and self.config.distortion_loss_mult > 0:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)

        # for error map update
        if self.config.use_error_map_sampler:
            with torch.no_grad():
                rgb_loss_ray_wise = torch.mean(torch.square(image - outputs["rgb"]), dim=-1)
                loss_dict["rgb_loss_ray_wise"] = rgb_loss_ray_wise.detach()

        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])
        if self.training:
            if self.config.use_sample_network:
                if self.step > 1000:
                    loss_dict["sample_network_loss"] = (
                        self.config.sample_network_loss_mult  # * (outputs["last_weight"].detach() - outputs["sn_weights"]).abs().mean()
                        * (-outputs["last_weight"].detach() * (outputs["sn_weights"] + 1e-7).log()).sum(-1).mean()
                    )

            if self.config.sparse_loss_mult > 0.0:
                _sparse_loss_mult = self.config.sparse_loss_mult_end + (
                    self.config.sparse_loss_mult - self.config.sparse_loss_mult_end
                ) * (self.step / self.config.sparse_loss_mult_decay)
                loss_dict["sparse_loss"] = _sparse_loss_mult * sparse_loss(outputs["density"])

            if self.config.render_static and self.config.use_loss_static:
                assert "is_static" in batch, "Must provide is_static in batch for static rendering."
                loss_dict["rgb_static_loss"] = self.rgb_loss(
                    image[batch["is_static"]], outputs["rgb_static"][batch["is_static"]]
                )
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )

            if self.config.mask_type.startswith("global"):
                if self.config.mask_loss_mult > 0.0:
                    loss_dict["mask_loss"] = (
                        # self.field.mlp_base.temporal_prod_net[0].get_mask_loss() * self.config.mask_loss_mult
                        self.field.mlp_base.temporal_prod_net.get_mask_loss()
                        * self.config.mask_loss_mult
                    )
                    if self.config.mask_loss_for_proposal:
                        for net in self.proposal_networks:
                            loss_dict["mask_loss"] += (
                                # net.mlp_base.temporal_prod_net[0].get_mask_loss() * self.config.mask_loss_mult
                                net.mlp_base.temporal_prod_net.get_mask_loss()
                                * self.config.mask_loss_mult
                            )
            else:
                raise NotImplementedError

            # + self.config.interlevel_loss_mult * interlevel_loss(
            #     outputs["weights_list_static"], outputs["ray_samples_list"]
            # )
            if self.config.distortion_loss_mult > 0.0:
                assert metrics_dict is not None and "distortion" in metrics_dict
                _dist_mult = self.config.distortion_loss_mult_end + (
                    self.config.distortion_loss_mult_end - self.config.distortion_loss_mult
                ) * (self.step / self.config.distortion_loss_mult_decay)

                loss_dict["distortion_loss"] = _dist_mult * metrics_dict["distortion"]
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
        if self.config.render_static:
            rgb_static = outputs["rgb_static"]
        else:
            rgb_static = None

        if self.config.debug:
            acc = colormaps.apply_colormap(outputs["accumulation"])
            depth = colormaps.apply_depth_colormap(
                outputs["depth"],
                accumulation=outputs["accumulation"],
            )
            combined_acc = torch.cat([acc], dim=1)
            combined_depth = torch.cat([depth], dim=1)

        if rgb_static is not None:
            combined_rgb = torch.cat([image, rgb, rgb_static], dim=1)
        else:
            combined_rgb = torch.cat([image, rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        ssim2 = self.ssim2(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim), "ssim2": float(ssim2)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        if self.config.debug:
            images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}
        else:
            images_dict = {"img": combined_rgb}

        if not self.config.use_sample_network and self.config.debug:
            for i in range(self.config.num_proposal_iterations):
                key = f"prop_depth_{i}"
                prop_depth_i = colormaps.apply_depth_colormap(
                    outputs[key],
                    accumulation=outputs["accumulation"],
                )
                images_dict[key] = prop_depth_i

        return metrics_dict, images_dict

    def freeze_temporal_prod(self):
        self.field.mlp_base.temporal_prod_net.requires_grad_(False)
        for _m in self.proposal_networks:
            _m.mlp_base.temporal_prod_net.requires_grad_(False)

    def freeze_static(self):
        self.field.mlp_base.spatial_net.requires_grad_(False)
        for _m in self.proposal_networks:
            _m.mlp_base.spatial_net.requires_grad_(False)

    def render_one_image(self, outputs, batch):
        """render a image, assuming camera_ray_bundle is a complete ray_bundle from the same camera"""
        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {
            "img": rgb.squeeze().moveaxis(0, -1),
        }

        return metrics_dict, images_dict

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle_incremental(
        self, camera_ray_bundle: RayBundle, mask, frame_one_output
    ) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        camera_ray_bundle = camera_ray_bundle.flatten()[~mask]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        with Timer("forwarding"):
            _t1 = time.time()
            for i in range(0, num_rays, num_rays_per_chunk):
                start_idx = i
                end_idx = i + num_rays_per_chunk
                ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
                outputs = self.forward(ray_bundle=ray_bundle)
                for output_name, output in outputs.items():  # type: ignore
                    outputs_lists[output_name].append(output)
            print(f"incremental forwarding took {time.time() - _t1} seconds")

        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            if not torch.is_tensor(outputs_list[0]):
                # TODO: handle lists of tensors as well
                continue
            outputs[output_name] = torch.cat(outputs_list).view(
                num_rays, -1
            )  # .view(image_height, image_width, -1)  # type: ignore
            temp = torch.zeros_like(frame_one_output[output_name]).view(image_height * image_width, -1)
            temp[~mask] = outputs[output_name]
            temp[mask] = frame_one_output[output_name].view(image_height * image_width, -1)[mask]
            temp = temp.view(image_height, image_width, -1)
            outputs[output_name] = temp

        return outputs

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle_fixed_pose(
        self, camera_ray_bundle: RayBundle, camera_pose, thre=0.9
    ) -> Dict[str, torch.Tensor]:
        if not hasattr(self, "last_cam_pose"):
            self.last_cam_pose = torch.zeros_like(camera_pose)

        if not torch.allclose(self.last_cam_pose, camera_pose):
            print("slow")
            outputs = self.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            self.frame_one_outputs = outputs
            self.ray_mask = (outputs["mask"] > thre).all(dim=-1).flatten()
            print("static_ratio:")
            print(self.ray_mask.sum() / self.ray_mask.numel())
            self.last_cam_pose = camera_pose.clone()
            self.last_ray_bundle_shape = camera_ray_bundle.shape
        else:
            print("rendering with fast mode")
            assert camera_ray_bundle.shape == self.last_ray_bundle_shape
            outputs = self.get_outputs_for_camera_ray_bundle_incremental(
                camera_ray_bundle, self.ray_mask, self.frame_one_outputs
            )

        return outputs

    def forward(self, ray_bundle: RayBundle, temporal_only=False) -> Dict[str, torch.Tensor]:
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """

        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        return self.get_outputs(ray_bundle, temporal_only)


# class
