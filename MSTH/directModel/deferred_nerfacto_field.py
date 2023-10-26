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
Field for compound nerf model, adds scene contraction and image embeddings to instant ngp
"""


from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torchtyping import TensorType

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import Encoding, HashEncoding, SHEncoding
from nerfstudio.field_components.field_heads import (
    DensityFieldHead,
    FieldHead,
    FieldHeadNames,
    PredNormalsFieldHead,
    RGBFieldHead,
    SemanticFieldHead,
    TransientDensityFieldHead,
    TransientRGBFieldHead,
    UncertaintyFieldHead,
)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import (
    SceneContraction,
    SpatialDistortion,
)
from nerfstudio.fields.base_field import Field
from MSTH.gridencoder import GridEncoder
from MSTH.ibrnet.colorizer import Colorizer
from nerfstudio.cameras.cameras import Cameras
from typing import Union
from MSTH.dataset import VideoDataset, VideoDatasetWithFeature

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass


def get_normalized_directions(directions: TensorType["bs":..., 3]) -> TensorType["bs":..., 3]:
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


from einops import rearrange


class AddFeatures(nn.Module):
    def __init__(self, features_per_level):
        super().__init__()
        self.features_per_level = features_per_level

    def forward(self, features):
        features = rearrange(features, "b (l f) -> b l f", f=self.features_per_level)
        features = features.mean(dim=1)
        return features


class TCNNDeferredNerfactoField(Field):
    """Compound Field that uses TCNN

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        geo_feat_dim: output geo feat dimensions
        num_levels: number of levels of the hashmap for the base mlp
        max_res: maximum resolution of the hashmap for the base mlp
        log2_hashmap_size: size of the hashmap for the base mlp
        num_layers_color: number of hidden layers for color network
        num_layers_transient: number of hidden layers for transient network
        hidden_dim_color: dimension of hidden layers for color network
        hidden_dim_transient: dimension of hidden layers for transient network
        appearance_embedding_dim: dimension of appearance embedding
        transient_embedding_dim: dimension of transient embedding
        use_transient_embedding: whether to use transient embedding
        use_semantics: whether to use semantic segmentation
        num_semantic_classes: number of semantic classes
        use_pred_normals: whether to use predicted normals
        use_average_appearance_embedding: whether to use average appearance embedding or zeros for inference
        spatial_distortion: spatial distortion to apply to the scene
    """

    def __init__(
        self,
        aabb: TensorType,
        num_images: int,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_levels: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        num_layers_color: int = 3,
        num_layers_transient: int = 2,
        hidden_dim_color: int = 64,
        hidden_dim_transient: int = 64,
        appearance_embedding_dim: int = 32,
        transient_embedding_dim: int = 16,
        use_transient_embedding: bool = False,
        use_semantics: bool = False,
        num_semantic_classes: int = 100,
        pass_semantic_gradients: bool = False,
        use_pred_normals: bool = False,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: SpatialDistortion = None,
        use_appearance_embedding: bool = True,
        gridtype: str = "hash",
        base_res: int = 16,
        features_per_level=2,
    ) -> None:
        super().__init__()

        self.register_buffer("aabb", aabb)
        self.geo_feat_dim = geo_feat_dim

        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))

        self.spatial_distortion = spatial_distortion
        self.num_images = num_images
        self.appearance_embedding_dim = appearance_embedding_dim

        self.use_appearance_embedding = use_appearance_embedding
        if self.use_appearance_embedding:
            self.embedding_appearance = Embedding(self.num_images, self.appearance_embedding_dim)
        self.use_average_appearance_embedding = use_average_appearance_embedding
        self.use_transient_embedding = use_transient_embedding
        self.use_semantics = use_semantics
        self.use_pred_normals = use_pred_normals
        self.pass_semantic_gradients = pass_semantic_gradients

        base_res = base_res
        if isinstance(base_res, (tuple, list)):
            growth_factor = (
                (1.0, 1.0, 1.0)
                if num_levels == 1
                else tuple(np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1)))
            )
        else:
            growth_factor = (
                1.0 if num_levels == 1 else float(np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1)))
            )

        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        self.position_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={"otype": "Frequency", "n_frequencies": 2},
        )

        self.encoding = GridEncoder(
            input_dim=3,
            num_levels=num_levels,
            level_dim=features_per_level,
            per_level_scale=growth_factor,
            base_resolution=base_res,
            log2_hashmap_size=log2_hashmap_size,
            gridtype=gridtype,
        )

        self.mlp_after_encoding = AddFeatures(features_per_level)

        # tcnn.Network(
        #     n_input_dims=num_levels * features_per_level,
        #     n_output_dims=1 + 1 + self.geo_feat_dim,
        #     network_config={
        #         "otype": "FullyFusedMLP",
        #         "activation": "ReLU",
        #         "output_activation": "None",
        #         "n_neurons": hidden_dim,
        #         "n_hidden_layers": num_layers - 1,
        #     },
        # )

        self.mlp_base = nn.Sequential(self.encoding, self.mlp_after_encoding)

        # self.mlp_base = tcnn.NetworkWithInputEncoding(
        #     n_input_dims=3,
        #     n_output_dims= 1 + 1 + self.geo_feat_dim,
        #     encoding_config={
        #         "otype": "HashGrid",
        #         "n_levels": num_levels,
        #         "n_features_per_level": features_per_level,
        #         "log2_hashmap_size": log2_hashmap_size,
        #         "base_resolution": base_res,
        #         "per_level_scale": growth_factor,
        #     },
        #     network_config={
        #         "otype": "FullyFusedMLP",
        #         "activation": "ReLU",
        #         "output_activation": "None",
        #         "n_neurons": hidden_dim,
        #         "n_hidden_layers": num_layers - 1,
        #     },
        # )

        # transients
        if self.use_transient_embedding:
            self.transient_embedding_dim = transient_embedding_dim
            self.embedding_transient = Embedding(self.num_images, self.transient_embedding_dim)
            self.mlp_transient = tcnn.Network(
                n_input_dims=self.geo_feat_dim + self.transient_embedding_dim,
                n_output_dims=hidden_dim_transient,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": hidden_dim_transient,
                    "n_hidden_layers": num_layers_transient - 1,
                },
            )
            self.field_head_transient_uncertainty = UncertaintyFieldHead(in_dim=self.mlp_transient.n_output_dims)
            self.field_head_transient_rgb = TransientRGBFieldHead(in_dim=self.mlp_transient.n_output_dims)
            self.field_head_transient_density = TransientDensityFieldHead(in_dim=self.mlp_transient.n_output_dims)

        # semantics
        if self.use_semantics:
            self.mlp_semantics = tcnn.Network(
                n_input_dims=self.geo_feat_dim,
                n_output_dims=hidden_dim_transient,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                },
            )
            self.field_head_semantics = SemanticFieldHead(
                in_dim=self.mlp_semantics.n_output_dims, num_classes=num_semantic_classes
            )

        # predicted normals
        if self.use_pred_normals:
            self.mlp_pred_normals = tcnn.Network(
                n_input_dims=self.geo_feat_dim + self.position_encoding.n_output_dims,
                n_output_dims=hidden_dim_transient,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                },
            )
            self.field_head_pred_normals = PredNormalsFieldHead(in_dim=self.mlp_pred_normals.n_output_dims)

        self.mlp_head = tcnn.Network(
            n_input_dims=self.direction_encoding.n_output_dims
            + 1
            + (self.appearance_embedding_dim) * int(self.use_appearance_embedding),
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            },
        )

    def upsample(self, resolution: int):
        self.encoding.upsample(resolution)

    def set_static(self, ray_samples: RaySamples, thresh: float = 0.1) -> None:
        density, _ = self.get_density(ray_samples)
        transmittance = ray_samples.get_transmittance(density).reshape(-1)

        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)

        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        positions_flat = positions.view(-1, 3)

        positions_valid = positions_flat[transmittance > thresh]
        self.encoding.generate_grid_mask(positions_valid)

    def hash_reinitialize(self, ray_samples: RaySamples, std: float) -> None:
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)

        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        positions_flat = positions.view(-1, 3)

        self.encoding.hash_reinitialize(positions_flat, std)

    def get_density(self, ray_samples: RaySamples) -> Tuple[TensorType, TensorType]:
        """Computes and returns the densities."""
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True
        positions_flat = positions.view(-1, 3)
        h = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1)

        density_before_activation, diffuse_color_before_activation = torch.split(h, [1, 1], dim=-1)

        # density_before_activation, diffuse_color_before_activation, base_mlp_out = torch.split(h, [1, 3, self.geo_feat_dim], dim=-1)
        self._density_before_activation = density_before_activation

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(positions))
        diffuse_color = torch.sigmoid(diffuse_color_before_activation)
        density = density * selector[..., None]
        diffuse_color = diffuse_color * selector[..., None]
        # base_mlp_out = torch.sigmoid(base_mlp_out)
        return density, diffuse_color  # , base_mlp_out

    def get_outputs(
        self, ray_bundle: RayBundle, density_embedding: Optional[TensorType] = None, rgb_intrinsic=None
    ) -> Dict[FieldHeadNames, TensorType]:
        outputs = {}
        if ray_bundle.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        camera_indices = ray_bundle.camera_indices.squeeze()

        directions = get_normalized_directions(ray_bundle.directions)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)

        outputs_shape = ray_bundle.directions.shape[:-1]

        # appearance
        if self.use_appearance_embedding:
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

        if self.use_appearance_embedding:
            h = torch.cat(
                [
                    d,
                    # density_embedding.view(-1, self.geo_feat_dim),
                    rgb_intrinsic.view(-1, 1),
                    embedded_appearance.view(-1, self.appearance_embedding_dim),
                ],
                dim=-1,
            )
        else:
            h = torch.cat(
                [
                    d,
                    # density_embedding.view(-1, self.geo_feat_dim),
                    rgb_intrinsic.view(-1, 1),
                ],
                dim=-1,
            )

        rgb = self.mlp_head(h).view(*outputs_shape, -1).to(directions)

        if not self.training:
            rgb = torch.nan_to_num(rgb)
            torch.clamp_(rgb, min=0.0, max=1.0)

        return rgb


class TorchNerfactoField(Field):
    """
    PyTorch implementation of the compound field.
    """

    def __init__(
        self,
        aabb: TensorType,
        num_images: int,
        position_encoding: Encoding = HashEncoding(),
        direction_encoding: Encoding = SHEncoding(),
        base_mlp_num_layers: int = 3,
        base_mlp_layer_width: int = 64,
        head_mlp_num_layers: int = 2,
        head_mlp_layer_width: int = 32,
        appearance_embedding_dim: int = 40,
        skip_connections: Tuple = (4,),
        field_heads: Tuple[FieldHead] = (RGBFieldHead(),),
        spatial_distortion: SpatialDistortion = SceneContraction(),
    ) -> None:
        super().__init__()
        self.aabb = Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        self.num_images = num_images
        self.appearance_embedding_dim = appearance_embedding_dim
        self.embedding_appearance = Embedding(self.num_images, self.appearance_embedding_dim)

        self.position_encoding = position_encoding
        self.direction_encoding = direction_encoding

        self.mlp_base = MLP(
            in_dim=self.position_encoding.get_out_dim(),
            num_layers=base_mlp_num_layers,
            layer_width=base_mlp_layer_width,
            skip_connections=skip_connections,
            out_activation=nn.ReLU(),
        )

        self.mlp_head = MLP(
            in_dim=self.mlp_base.get_out_dim() + self.direction_encoding.get_out_dim() + self.appearance_embedding_dim,
            num_layers=head_mlp_num_layers,
            layer_width=head_mlp_layer_width,
            out_activation=nn.ReLU(),
        )

        self.field_output_density = DensityFieldHead(in_dim=self.mlp_base.get_out_dim())
        self.field_heads = nn.ModuleList(field_heads)
        for field_head in self.field_heads:
            field_head.set_in_dim(self.mlp_head.get_out_dim())  # type: ignore

    def get_density(self, ray_samples: RaySamples) -> Tuple[TensorType, TensorType]:
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
        else:
            positions = ray_samples.frustums.get_positions()
        encoded_xyz = self.position_encoding(positions)
        base_mlp_out = self.mlp_base(encoded_xyz)
        density = self.field_output_density(base_mlp_out)
        return density, base_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None
    ) -> Dict[FieldHeadNames, TensorType]:
        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        camera_indices = ray_samples.camera_indices.squeeze()
        if self.training:
            embedded_appearance = self.embedding_appearance(camera_indices)
        else:
            embedded_appearance = torch.zeros(
                (*outputs_shape, self.appearance_embedding_dim),
                device=ray_samples.frustums.directions.device,
            )

        outputs = {}
        for field_head in self.field_heads:
            encoded_dir = self.direction_encoding(ray_samples.frustums.directions)
            mlp_out = self.mlp_head(
                torch.cat(
                    [
                        encoded_dir,
                        density_embedding,  # type:ignore
                        embedded_appearance.view(-1, self.appearance_embedding_dim),
                    ],
                    dim=-1,  # type:ignore
                )
            )
            outputs[field_head.field_head_name] = field_head(mlp_out)
        return outputs


class IBRDeferredNerfactoField(TCNNDeferredNerfactoField):
    def __init__(
        self,
        aabb: TensorType,
        train_cameras: Cameras,
        eval_cameras: Cameras,
        train_dataset: VideoDatasetWithFeature,
        device: Union[str, torch.device],
        num_images: int,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_levels: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        num_layers_color: int = 3,
        num_layers_transient: int = 2,
        hidden_dim_color: int = 64,
        hidden_dim_transient: int = 64,
        appearance_embedding_dim: int = 32,
        transient_embedding_dim: int = 16,
        use_transient_embedding: bool = False,
        use_semantics: bool = False,
        num_semantic_classes: int = 100,
        pass_semantic_gradients: bool = False,
        use_pred_normals: bool = False,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: SpatialDistortion = None,
        use_appearance_embedding: bool = True,
        gridtype: str = "hash",
        base_res: int = 16,
        features_per_level=2,
    ) -> None:
        super().__init__(
            aabb,
            num_images,
            num_layers,
            hidden_dim,
            geo_feat_dim,
            num_levels,
            max_res,
            log2_hashmap_size,
            num_layers_color,
            num_layers_transient,
            hidden_dim_color,
            hidden_dim_transient,
            appearance_embedding_dim,
            transient_embedding_dim,
            use_transient_embedding,
            use_semantics,
            num_semantic_classes,
            pass_semantic_gradients,
            use_pred_normals,
            use_average_appearance_embedding,
            spatial_distortion,
            use_appearance_embedding,
            gridtype,
            base_res,
            features_per_level,
        )
        self.train_cameras = train_cameras
        self.eval_cameras = eval_cameras
        self.train_dataset = train_dataset
        self.colorizer = Colorizer(cameras=train_cameras, dataset=train_dataset, device=device)

    def train(self, mode):
        super().train(mode)
        if mode:
            self.c2ws = self.train_cameras.camera_to_worlds
        else:
            self.c2ws = self.eval_cameras.camera_to_worlds

    def eval(self, mode=False):
        super().eval()
        self.c2ws = self.eval_cameras.camera_to_worlds

    def get_density(self, ray_samples: RaySamples) -> Tuple[TensorType, TensorType]:
        """Computes and returns the densities."""
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True
        positions_flat = positions.view(-1, 3)
        h = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1)

        density_before_activation, diffuse_color_before_activation = torch.split(h, [1, 1], dim=-1)

        # density_before_activation, diffuse_color_before_activation, base_mlp_out = torch.split(h, [1, 3, self.geo_feat_dim], dim=-1)
        self._density_before_activation = density_before_activation

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(positions))
        # diffuse_color = torch.sigmoid(diffuse_color_before_activation)
        # density = density * selector[..., None]
        # diffuse_color = diffuse_color * selector[..., None]

        rgb = self.colorizer(ray_samples, self.c2ws)

        if not self.training:
            rgb = torch.nan_to_num(rgb)
            torch.clamp_(rgb, min=0.0, max=1.0)

        # base_mlp_out = torch.sigmoid(base_mlp_out)
        return density, rgb  # , base_mlp_out

    def get_outputs(self, ray_bundle: RayBundle, density_embedding: Optional[TensorType] = None):
        if ray_bundle.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")

        rgb = self.colorizer(ray_bundle, self.c2ws)

        if not self.training:
            rgb = torch.nan_to_num(rgb)
            torch.clamp_(rgb, min=0.0, max=1.0)

        return rgb


field_implementation_to_class: Dict[str, Field] = {"tcnn": TCNNDeferredNerfactoField, "torch": TorchNerfactoField}
