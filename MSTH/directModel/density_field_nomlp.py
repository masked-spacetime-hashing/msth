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
Proposal network field.
"""


from typing import Optional, Tuple

import pdb
import numpy as np
import torch
import torch.nn as nn
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field
from MSTH.gridencoder import GridEncoder
from MSTH.directModel.deferred_nerfacto_field import AddFeatures
try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass
from torch import nn


class HashDensityField(Field):
    """A lightweight density field module.

    Args:
        aabb: parameters of scene aabb bounds
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        spatial_distortion: spatial distortion module
        use_linear: whether to skip the MLP and use a single linear layer instead
    """

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
        gridtype: str = 'hash',
    ) -> None:
        super().__init__()
        self.register_buffer("aabb", aabb)
        self.spatial_distortion = spatial_distortion
        self.use_linear = use_linear
        growth_factor = 1 if num_levels == 1 else np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1))

        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))

        self.encoding = GridEncoder(input_dim=3, 
                                    num_levels=num_levels,
                                    level_dim=1,
                                    per_level_scale=growth_factor,
                                    base_resolution=base_res,
                                    log2_hashmap_size=log2_hashmap_size,
                                    gridtype=gridtype,
                                    )

        self.mlp_after_encoding = AddFeatures(features_per_level=1)
        self.mlp_base = nn.Sequential(self.encoding, self.mlp_after_encoding)

    def reset_parameters(self) -> None:
        self.encoding.reset_parameters()

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
    
    def upsample(self, resolution: int):
        self.encoding.upsample(resolution)

    def get_density(self, ray_samples: RaySamples) -> Tuple[TensorType, None]:
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(ray_samples.frustums.get_positions())
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        positions_flat = positions.view(-1, 3)
        density_before_activation = (
            self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1).to(positions)
        )

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation)
        density = density * selector[..., None]
        return density, None

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None) -> dict:
        return {}
