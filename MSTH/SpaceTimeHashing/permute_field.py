import torch
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
from nerfstudio.model_components.ray_samplers import (
    ProposalNetworkSampler,
    UniformSampler,
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

CONSOLE = Console(width=120)

class SpaceTimeDensityFieldWithPermutation(Field):
    def __init__(
        self,
        aabb: TensorType,
        num_layers: int = 2,
        hidden_dim: int = 64,
        spatial_distortion: Optional[SpatialDistortion] = None,
        use_linear: bool = False,
        num_levels: int = 8,
        max_res: int = 1024,
        base_res: int = 16,
        log2_hashmap_size: int = 18,
        features_per_level: int = 2,
    ) -> None:
        super().__init__()
        self.register_buffer("aabb", aabb)
        self.spatial_distortion = spatial_distortion
        self.use_linear = use_linear
        growth_factor = np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1))

        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))

        config = {
            "encoding": {
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_res,
                "per_level_scale": growth_factor,
            },
            "network": {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        }

        if not self.use_linear:
            self.mlp_base = tcnn.NetworkWithInputEncoding(
                n_input_dims=4,
                n_output_dims=1,
                encoding_config=config["encoding"],
                network_config=config["network"],
            )
            self.spatial_mlp_base = tcnn.NetworkWithInputEncoding(
                n_input_dims=3,
                n_output_dims=1,
                encoding_config=config["encoding"],
                network_config=config["network"],
            )
            self.xyzt = tcnn.NetworkWithInputEncoding(
                n_input_dims=4,
                n_output_dims=1,
                encoding_config=config["encoding"],
                network_config=config["network"],
            )
            self.xtyz = tcnn.NetworkWithInputEncoding(
                n_input_dims=4,
                n_output_dims=1,
                encoding_config=config["encoding"],
                network_config=config["network"],
            )
            self.xytz = tcnn.NetworkWithInputEncoding(
                n_input_dims=4,
                n_output_dims=1,
                encoding_config=config["encoding"],
                network_config=config["network"],
            )
            self.txyz = tcnn.NetworkWithInputEncoding(
                n_input_dims=4,
                n_output_dims=1,
                encoding_config=config["encoding"],
                network_config=config["network"],
            )
            
        else:
            self.encoding = tcnn.Encoding(n_input_dims=4, encoding_config=config["encoding"])
            self.linear = torch.nn.Linear(self.encoding.n_output_dims, 1)

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
                self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1).to(positions) + self.spatial_mlp_base(positions.view(-1, 3)).view(*ray_samples.frustums.shape, -1).to(positions)
            )
        else:
            x = self.encoding(positions_flat).to(positions)
            density_before_activation = self.linear(x).view(*ray_samples.frustums.shape, -1)

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation)
        density = density * selector[..., None]
        return density, None

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None) -> dict:
        return {}

    def density_fn(self, positions: TensorType["bs": ..., 4]) -> TensorType["bs": ..., 1]:
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
        density, _ = self.get_density(ray_samples)
        return density