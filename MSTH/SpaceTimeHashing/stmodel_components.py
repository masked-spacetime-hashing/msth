import torch
import torch.nn as nn
from MSTH.gridencoder import GridEncoder
from MSTH.gridencoder import SpatialTemporalGridEncoder
import tinycudann as tcnn
from MSTH.utils import Timer
from nerfstudio.field_components.activations import trunc_exp
from rich.console import Console
from einops import rearrange


class SampleNetwork(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=2):
        super().__init__()
        # ray_o
        self.position_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={"otype": "Frequency", "n_frequencies": 2},
        )
        # ray_d
        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )
        # ray_t
        self.time_encoding = tcnn.Encoding(
            n_input_dims=1,
            encoding_config={"otype": "Frequency", "n_frequencies": 2},
        )

        self.network = tcnn.Network(
            n_input_dims=self.position_encoding.n_output_dims
            + self.direction_encoding.n_output_dims
            + self.time_encoding.n_output_dims,
            n_output_dims=128,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                # "output_activation": "ReLU",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        )

    # def forward(self, rays_o, rays_d, rays_t):
    #     # rays_o: B x 3
    #     # rays_d: B x 3
    #     # rays_t: B x 1
    #     if len(rays_t.shape) == 1:
    #         rays_t = rays_t.unsqueeze(-1)
    #     o_rep = self.position_encoding(rays_o)
    #     d_rep = self.direction_encoding(rays_d)
    #     t_rep = self.time_encoding(rays_t)
    #     input_rep = torch.cat([o_rep, d_rep, t_rep], dim=-1)
    #     out = trunc_exp(self.network(input_rep))
    #     # out = self.network(input_rep)
    #     mean = out[..., 0:1]
    #     std = out[..., 1:2]
    #     return mean, std

    def forward(self, rays_o, rays_d, rays_t):
        # rays_o: B x 3
        # rays_d: B x 3
        # rays_t: B x 1
        if len(rays_t.shape) == 1:
            rays_t = rays_t.unsqueeze(-1)
        o_rep = self.position_encoding(rays_o)
        d_rep = self.direction_encoding(rays_d)
        t_rep = self.time_encoding(rays_t)
        input_rep = torch.cat([o_rep, d_rep, t_rep], dim=-1)
        out = torch.nn.functional.softmax(self.network(input_rep), dim=-1)
        # out = self.network(input_rep)
        return out

    # def get_pdf_at(self, rel_pos, mean, std):
    #     rel_pos = rel_pos.squeeze()
    #     eps = 1e-7
    #     # rel_pos B x N, mean B x 1, std B x 1
    #     P = 1 / (2.5066282532517663 * (std + eps)) * torch.exp(-0.5 * (rel_pos - mean) ** 2 / (std**2 + eps))
    #     return P


class STModule(nn.Module):
    def __init__(
        self,
        n_output_dim,
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
        st_mlp_mode="independent",
        use_linear=False,
        spatial_only=False,
        interp="linear",
        level_one_interp="linear",
        nosigmoid=False,
        ablation_add=False,
        mask_init_mean=0.0,
        mask_multi_level=False,
        customized_t_reso=None,
    ):
        super().__init__()
        assert st_mlp_mode == "independent" or st_mlp_mode == "shared"
        self.ablation_add = ablation_add
        self.st_mlp_mode = st_mlp_mode
        self.spatial_only = spatial_only
        if st_mlp_mode == "independent":
            self.spatial_net = nn.Sequential(
                GridEncoder(
                    input_dim=3,
                    num_levels=num_levels,
                    level_dim=features_per_level,
                    per_level_scale=per_level_scale[:3],
                    base_resolution=base_res[:3],
                    log2_hashmap_size=log2_hashmap_size_spatial,
                    std=1e-4,
                    interpolation=interp,
                ),
                nn.Linear(features_per_level * num_levels, n_output_dim)
                if use_linear
                else tcnn.Network(
                    n_input_dims=features_per_level * num_levels,
                    n_output_dims=n_output_dim,
                    network_config={
                        "otype": "FullyFusedMLP",
                        "activation": "ReLU",
                        "output_activation": "None",
                        "n_neurons": hidden_dim,
                        "n_hidden_layers": num_layers - 1,
                    },
                ),
            )

            self.temporal_net = nn.Sequential(
                GridEncoder(
                    input_dim=4,
                    num_levels=num_levels,
                    level_dim=features_per_level,
                    per_level_scale=per_level_scale,
                    base_resolution=base_res,
                    log2_hashmap_size=log2_hashmap_size_temporal,
                    std=1e-4,
                    interpolation=interp,
                ),
                nn.Linear(features_per_level * num_levels, n_output_dim)
                if use_linear
                else tcnn.Network(
                    n_input_dims=features_per_level * num_levels,
                    n_output_dims=n_output_dim,
                    network_config={
                        "otype": "FullyFusedMLP",
                        "activation": "ReLU",
                        "output_activation": "None",
                        "n_neurons": hidden_dim,
                        "n_hidden_layers": num_layers - 1,
                    },
                ),
            )
        else:
            self.spatial_net = GridEncoder(
                input_dim=3,
                num_levels=num_levels,
                level_dim=features_per_level,
                per_level_scale=per_level_scale[:3],
                base_resolution=base_res[:3],
                log2_hashmap_size=log2_hashmap_size_spatial,
                std=1e-4,
                interpolation=interp,
            )

            self.temporal_net = GridEncoder(
                input_dim=4,
                num_levels=num_levels,
                level_dim=features_per_level,
                per_level_scale=per_level_scale,
                base_resolution=base_res,
                log2_hashmap_size=log2_hashmap_size_temporal,
                std=1e-4,
                interpolation=interp,
                customized_t_reso=customized_t_reso,
            )

            self.mlp_shared = (
                nn.Linear(features_per_level * num_levels, n_output_dim)
                if use_linear
                else tcnn.Network(
                    n_input_dims=features_per_level * num_levels,
                    n_output_dims=n_output_dim,
                    network_config={
                        "otype": "FullyFusedMLP",
                        "activation": "ReLU",
                        "output_activation": "None",
                        "n_neurons": hidden_dim,
                        "n_hidden_layers": num_layers - 1,
                    },
                )
            )
        if mask_multi_level:
            self.temporal_prod_net = GridEncoder(
                input_dim=3,
                num_levels=num_levels,
                level_dim=1,
                per_level_scale=per_level_scale[:3],
                base_resolution=base_res[:3],
                log2_hashmap_size=mask_log2_hash_size,
                std=1e-4,
                gridtype="hash",
                interpolation=interp,
                # interpolation="all_nearest",
                mean=mask_init_mean,
            )
        else:
            self.temporal_prod_net = GridEncoder(
                input_dim=3,
                num_levels=1,
                level_dim=1,
                per_level_scale=1,
                base_resolution=mask_reso,
                log2_hashmap_size=mask_log2_hash_size,
                std=1e-4,
                gridtype="tiled",
                interpolation=interp,
                # interpolation="all_nearest",
                mean=mask_init_mean,
            )
        self.nosigmoid = nosigmoid
        self.mode = mode

    #
    def forward(self, x, temporal_only=False):
        spatial_component = self.spatial_net(x[..., :3])
        if self.spatial_only:
            return spatial_component, spatial_component
        # torch.cuda.synchronize("cuda")
        temporal_component = self.temporal_net(x)

        # torch.cuda.synchronize("cuda")
        temporal_prod_component = self.temporal_prod_net(x[..., :3])
        if not self.nosigmoid:
            temporal_prod_component = temporal_prod_component.sigmoid()

        if not self.training:
            self.mask_detached = temporal_prod_component.detach()
        else:
            self.mask_detached = None
        # torch.cuda.synchronize("cuda")

        if self.ablation_add:
            # print("spatial add temporal")
            output = spatial_component + temporal_component
            if self.st_mlp_mode == "shared":
                # print("shared")
                output = self.mlp_shared(output)
                # spatial_component = self.mlp_shared(spatial_component)
                spatial_component = None
            return (
                output,
                # spatial_component * temporal_prod_component + (1 - temporal_prod_component) * temporal_component,
                spatial_component,
            )

        if self.mode == "mst":
            output = spatial_component * temporal_prod_component + (1 - temporal_prod_component) * temporal_component
        else:
            assert self.mode == "mt"
            output = spatial_component + (1 - temporal_prod_component) * temporal_component
            # TODO remove
            # output = (spatial_component + (1 - temporal_prod_component) * temporal_component) / (
            #     2 - temporal_prod_component
            # )

        if self.st_mlp_mode == "shared":
            output = self.mlp_shared(output)
            spatial_component = self.mlp_shared(spatial_component)
            # spatial_component = None

        return (
            output,
            # spatial_component * temporal_prod_component + (1 - temporal_prod_component) * temporal_component,
            spatial_component,
            temporal_prod_component,
        )


class STModuleFuse(nn.Module):
    def __init__(
        self,
        n_output_dim,
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
        st_mlp_mode="independent",
        use_linear=False,
        spatial_only=False,
        interp="linear",
        level_one_interp="linear",
        nosigmoid=False,
        ablation_add=False,
        **kwargs,
    ):
        super().__init__()
        assert st_mlp_mode == "shared", "only suport fused operation for shared mode"
        assert log2_hashmap_size_spatial == log2_hashmap_size_temporal
        self.st_mlp_mode = st_mlp_mode
        self.backbone = SpatialTemporalGridEncoder(
            input_dim=4,
            num_levels=num_levels,
            level_dim=features_per_level,
            per_level_scale=per_level_scale,
            base_resolution=base_res,
            log2_hashmap_size=log2_hashmap_size_spatial,
            desired_resolution=None,
            gridtype="hash",
            align_corners=False,
            interpolation="linear",
            mask_resolution=mask_reso,
            std=1e-4,
        )

        self.mlp_shared = (
            nn.Linear(features_per_level * num_levels, n_output_dim)
            if use_linear
            else tcnn.Network(
                n_input_dims=features_per_level * num_levels,
                n_output_dims=n_output_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": hidden_dim,
                    "n_hidden_layers": num_layers - 1,
                },
            )
        )

        self.mode = mode

    #
    def forward(self, x):
        output = self.backbone(x)
        output = self.mlp_shared(output)
        # spatial_component = self.mlp_shared(spatial_component)
        spatial_component = None

        return (
            output,
            # spatial_component * temporal_prod_component + (1 - temporal_prod_component) * temporal_component,
            spatial_component,
        )


class STModuleHierarchicay(nn.Module):
    def __init__(
        self,
        n_output_dim,
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
        st_mlp_mode="independent",
        use_linear=False,
        spatial_only=False,
        interp="linear",
        level_one_interp="linear",
        nosigmoid=False,
        ablation_add=False,
        **kwargs,
    ):
        super().__init__()
        assert st_mlp_mode == "independent" or st_mlp_mode == "shared"
        self.st_mlp_mode = st_mlp_mode
        self.level_one_bins = base_res[-1]
        self.features_per_level = features_per_level
        self.num_levels = num_levels
        self.n_output_dim = n_output_dim
        if self.st_mlp_mode == "independent":
            self.spatial_net = nn.Sequential(
                GridEncoder(
                    input_dim=3,
                    num_levels=num_levels,
                    level_dim=features_per_level,
                    per_level_scale=per_level_scale[:3],
                    base_resolution=base_res[:3],
                    log2_hashmap_size=log2_hashmap_size_spatial,
                    std=1e-4,
                    interpolation=interp,
                ),
                nn.Linear(features_per_level * num_levels, n_output_dim)
                if use_linear
                else tcnn.Network(
                    n_input_dims=features_per_level * num_levels,
                    n_output_dims=n_output_dim,
                    network_config={
                        "otype": "FullyFusedMLP",
                        "activation": "ReLU",
                        "output_activation": "None",
                        "n_neurons": hidden_dim,
                        "n_hidden_layers": num_layers - 1,
                    },
                ),
            )

            self.level_one_temporal_net = nn.Sequential(
                GridEncoder(
                    input_dim=4,
                    num_levels=num_levels,
                    level_dim=features_per_level,
                    per_level_scale=(per_level_scale[:3] + (1,)),
                    base_resolution=base_res,
                    log2_hashmap_size=log2_hashmap_size_spatial,
                    std=1e-4,
                    interpolation=level_one_interp,
                ),
                nn.Linear(features_per_level * num_levels, n_output_dim)
                if use_linear
                else tcnn.Network(
                    n_input_dims=features_per_level * num_levels,
                    n_output_dims=n_output_dim,
                    network_config={
                        "otype": "FullyFusedMLP",
                        "activation": "ReLU",
                        "output_activation": "None",
                        "n_neurons": hidden_dim,
                        "n_hidden_layers": num_layers - 1,
                    },
                ),
            )
            self.temporal_net = nn.Sequential(
                GridEncoder(
                    input_dim=4,
                    num_levels=num_levels,
                    level_dim=features_per_level,
                    per_level_scale=per_level_scale,
                    base_resolution=base_res,
                    log2_hashmap_size=log2_hashmap_size_temporal,
                    std=1e-4,
                    interpolation=interp,
                ),
                nn.Linear(features_per_level * num_levels, n_output_dim)
                if use_linear
                else tcnn.Network(
                    n_input_dims=features_per_level * num_levels,
                    n_output_dims=n_output_dim,
                    network_config={
                        "otype": "FullyFusedMLP",
                        "activation": "ReLU",
                        "output_activation": "None",
                        "n_neurons": hidden_dim,
                        "n_hidden_layers": num_layers - 1,
                    },
                ),
            )
        else:
            self.spatial_net = GridEncoder(
                input_dim=3,
                num_levels=num_levels,
                level_dim=features_per_level,
                per_level_scale=per_level_scale[:3],
                base_resolution=base_res[:3],
                log2_hashmap_size=log2_hashmap_size_spatial,
                std=1e-4,
                interpolation=interp,
            )

            self.level_one_temporal_net = GridEncoder(
                input_dim=4,
                num_levels=num_levels,
                level_dim=features_per_level,
                per_level_scale=(per_level_scale[:3] + (1,)),
                base_resolution=base_res,
                log2_hashmap_size=log2_hashmap_size_spatial,
                std=1e-4,
                interpolation=level_one_interp,
            )

            self.temporal_net = GridEncoder(
                input_dim=4,
                num_levels=num_levels,
                level_dim=features_per_level,
                per_level_scale=per_level_scale,
                base_resolution=base_res,
                log2_hashmap_size=log2_hashmap_size_temporal,
                std=1e-4,
                interpolation=interp,
            )

            self.mlp_shared = (
                nn.Linear(features_per_level * num_levels, n_output_dim)
                if use_linear
                else tcnn.Network(
                    n_input_dims=features_per_level * num_levels,
                    n_output_dims=n_output_dim,
                    network_config={
                        "otype": "FullyFusedMLP",
                        "activation": "ReLU",
                        "output_activation": "None",
                        "n_neurons": hidden_dim * 2,
                        "n_hidden_layers": num_layers - 1,
                    },
                )
            )
        self.temporal_prod_net = nn.Sequential(
            GridEncoder(
                input_dim=3,
                num_levels=1,
                level_dim=2,
                per_level_scale=1,
                base_resolution=mask_reso,
                log2_hashmap_size=mask_log2_hash_size,
                std=1e-4,
                interpolation=interp,
            ),
            nn.Linear(2, 1 + self.level_one_bins)
            if use_linear
            else tcnn.Network(
                n_input_dims=2,
                n_output_dims=1 + self.level_one_bins,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": ("None" if nosigmoid else "Sigmoid"),
                    "n_neurons": 16,
                    "n_hidden_layers": num_layers - 1,
                },
            ),
        )

        self.mode = mode

    def forward(self, x, **kwargs):
        # if not self.training:
        #     return self.forward_eval(x)
        spatial_component = self.spatial_net(x[..., :3])
        level_one_component = self.level_one_temporal_net(x)
        temporal_component = self.temporal_net(x)
        temporal_prod_component = self.temporal_prod_net(x[..., :3])

        level_one_indices = (x[..., 3] * self.level_one_bins).long() + 1
        level_one_mask = temporal_prod_component.gather(1, level_one_indices.unsqueeze(dim=-1)).squeeze(dim=-1)

        if self.mode == "mst":
            output = spatial_component * temporal_prod_component[..., 0:1] + (1 - temporal_prod_component[..., 0:1]) * (
                level_one_component * level_one_mask.unsqueeze(dim=-1)
                + temporal_component * (1 - level_one_mask).unsqueeze(dim=-1)
            )
        else:
            assert self.mode == "mt"
            output = spatial_component + (1 - temporal_prod_component[..., 0:1]) * (
                level_one_component + (1 - level_one_mask).unsqueeze(dim=-1) * temporal_component
            )

        if self.st_mlp_mode == "shared":
            output = self.mlp_shared(output)
            spatial_component = self.mlp_shared(spatial_component)

        return (
            output,
            # spatial_component * temporal_prod_component + (1 - temporal_prod_component) * temporal_component,
            spatial_component,
            level_one_mask,
            # temporal_prod_component,
        )

    @torch.no_grad()
    def forward_eval(self, x, thresh_high=0.9):
        thresh_low = 1.0 - thresh_high

        temporal_prod_component = self.temporal_prod_net(x[..., :3])
        level_one_indices = (x[..., 3] * self.level_one_bins).long() + 1
        level_one_mask = temporal_prod_component.gather(1, level_one_indices.unsqueeze(dim=-1)).squeeze(dim=-1)
        level_zero_mask = temporal_prod_component[..., 0]

        mask_l0_only_l0 = level_zero_mask > thresh_high  # the last result is determined by l0
        mask_l0_not_only_l0 = (level_zero_mask <= thresh_high) & (
            level_zero_mask >= thresh_low
        )  # requires computations on l0 and l1 ...

        # early stop at the first layer
        # output = torch.zeros(
        #     x.shape[0],
        #     (self.n_output_dim if self.st_mlp_mode == "independent" else self.features_per_level * self.num_levels),
        #     device=x.device,
        # )

        if self.mode == "mst":
            # output shoudl store the mask_l0 as well as the ~(mask_l0|mask_l1)
            _mask = mask_l0_only_l0 | mask_l0_not_only_l0
            output[_mask] = level_zero_mask[_mask].unsqueeze(-1) * self.spatial_net(x[_mask, :3])
        else:
            assert self.mode == "mt"
            spatial_component = self.spatial_net(x[..., :3])
            # output += 1 * spatial_component
            output = spatial_component

        mask_l1_only_l1 = (~mask_l0_only_l0) & (level_one_mask > thresh_high)
        mask_l1_not_only_l1 = (~mask_l0_only_l0) & (level_one_mask <= thresh_high) & (level_one_mask >= thresh_low)

        if self.mode == "mst":
            # output shoudl store the mask_l0 as well as the ~(mask_l0|mask_l1)
            _mask = mask_l1_only_l1 | mask_l1_not_only_l1
            output[_mask] += ((1.0 - level_zero_mask[_mask]) * (level_one_mask[_mask])).unsqueeze(
                -1
            ) * self.level_one_temporal_net(x[_mask])
        else:
            assert self.mode == "mt"
            _mask = ~mask_l0_only_l0
            output[_mask] += (1.0 - level_zero_mask[_mask]).unsqueeze(-1) * self.level_one_temporal_net(x[_mask])

        mask_l2_only_l2 = (~mask_l0_only_l0) & (~mask_l1_only_l1)
        if self.mode == "mst" or self.mode == "mt":
            output[mask_l2_only_l2] += (
                (1.0 - level_zero_mask[mask_l2_only_l2]) * (1.0 - level_one_mask[mask_l2_only_l2])
            ).unsqueeze(-1) * self.temporal_net(x[mask_l2_only_l2])

        if self.st_mlp_mode == "shared":
            output = self.mlp_shared(output)
            # spatial_component = self.mlp_shared(spatial_component)
        return output, spatial_component


class STModuleML(nn.Module):
    def __init__(
        self,
        n_output_dim,
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
        st_mlp_mode="independent",
        use_linear=False,
        spatial_only=False,
        interp="linear",
        level_one_interp="linear",
        nosigmoid=False,
        ablation_add=False,
        mask_init_mean=0.0,
    ):
        super().__init__()
        assert st_mlp_mode == "independent" or st_mlp_mode == "shared"
        self.ablation_add = ablation_add
        self.st_mlp_mode = st_mlp_mode
        self.spatial_only = spatial_only
        self.num_levels = num_levels
        if st_mlp_mode == "independent":
            self.spatial_net = nn.Sequential(
                GridEncoder(
                    input_dim=3,
                    num_levels=num_levels,
                    level_dim=features_per_level,
                    per_level_scale=per_level_scale[:3],
                    base_resolution=base_res[:3],
                    log2_hashmap_size=log2_hashmap_size_spatial,
                    std=1e-4,
                    interpolation=interp,
                ),
                nn.Linear(features_per_level * num_levels, n_output_dim)
                if use_linear
                else tcnn.Network(
                    n_input_dims=features_per_level * num_levels,
                    n_output_dims=n_output_dim,
                    network_config={
                        "otype": "FullyFusedMLP",
                        "activation": "ReLU",
                        "output_activation": "None",
                        "n_neurons": hidden_dim,
                        "n_hidden_layers": num_layers - 1,
                    },
                ),
            )

            self.temporal_net = nn.Sequential(
                GridEncoder(
                    input_dim=4,
                    num_levels=num_levels,
                    level_dim=features_per_level,
                    per_level_scale=per_level_scale,
                    base_resolution=base_res,
                    log2_hashmap_size=log2_hashmap_size_temporal,
                    std=1e-4,
                    interpolation=interp,
                ),
                nn.Linear(features_per_level * num_levels, n_output_dim)
                if use_linear
                else tcnn.Network(
                    n_input_dims=features_per_level * num_levels,
                    n_output_dims=n_output_dim,
                    network_config={
                        "otype": "FullyFusedMLP",
                        "activation": "ReLU",
                        "output_activation": "None",
                        "n_neurons": hidden_dim,
                        "n_hidden_layers": num_layers - 1,
                    },
                ),
            )
        else:
            self.spatial_net = GridEncoder(
                input_dim=3,
                num_levels=num_levels,
                level_dim=features_per_level,
                per_level_scale=per_level_scale[:3],
                base_resolution=base_res[:3],
                log2_hashmap_size=log2_hashmap_size_spatial,
                std=1e-4,
                interpolation=interp,
            )

            self.temporal_net = GridEncoder(
                input_dim=4,
                num_levels=num_levels,
                level_dim=features_per_level,
                per_level_scale=per_level_scale,
                base_resolution=base_res,
                log2_hashmap_size=log2_hashmap_size_temporal,
                std=1e-4,
                interpolation=interp,
            )

            self.mlp_shared = (
                nn.Linear(features_per_level * num_levels, n_output_dim)
                if use_linear
                else tcnn.Network(
                    n_input_dims=features_per_level * num_levels,
                    n_output_dims=n_output_dim,
                    network_config={
                        "otype": "FullyFusedMLP",
                        "activation": "ReLU",
                        "output_activation": "None",
                        "n_neurons": hidden_dim,
                        "n_hidden_layers": num_layers - 1,
                    },
                )
            )
        self.temporal_prod_net = GridEncoder(
            input_dim=3,
            num_levels=num_levels,
            level_dim=1,
            per_level_scale=per_level_scale[:3],
            base_resolution=base_res[:3],
            log2_hashmap_size=mask_log2_hash_size,
            std=1e-4,
            gridtype="hash",
            interpolation=interp,
            # interpolation="all_nearest",
            mean=mask_init_mean,
        )
        self.nosigmoid = nosigmoid
        self.mode = mode

    #
    def forward(self, x):
        spatial_component = self.spatial_net(x[..., :3])
        if self.spatial_only:
            return spatial_component, spatial_component
        # torch.cuda.synchronize("cuda")
        temporal_component = self.temporal_net(x)

        # torch.cuda.synchronize("cuda")
        temporal_prod_component = self.temporal_prod_net(x[..., :3])
        if not self.nosigmoid:
            temporal_prod_component = temporal_prod_component.sigmoid()
        # torch.cuda.synchronize("cuda")

        if self.ablation_add:
            # print("spatial add temporal")
            output = spatial_component + temporal_component
            if self.st_mlp_mode == "shared":
                # print("shared")
                output = self.mlp_shared(output)
                # spatial_component = self.mlp_shared(spatial_component)
                spatial_component = None
            return (
                output,
                # spatial_component * temporal_prod_component + (1 - temporal_prod_component) * temporal_component,
                spatial_component,
            )

        if self.mode == "mst":
            _mask = rearrange(temporal_prod_component, "b (l c) -> b l c", l=self.num_levels)
            output = rearrange(spatial_component, "b (l c) -> b l c", l=self.num_levels) * _mask + (
                1 - _mask
            ) * rearrange(temporal_component, "b (l c) -> b l c", l=self.num_levels)
        else:
            assert self.mode == "mt"
            _mask = rearrange(temporal_prod_component, "b (l c) -> b l c", l=self.num_levels)
            output = rearrange(spatial_component, "b (l c) -> b l c", l=self.num_levels) + (1 - _mask) * rearrange(
                temporal_component, "b (l c) -> b l c", l=self.num_levels
            )
        output = rearrange(output, "b l c -> b (l c)")

        if self.st_mlp_mode == "shared":
            output = self.mlp_shared(output)
            # spatial_component = self.mlp_shared(spatial_component)
            spatial_component = None

        return (
            output,
            # spatial_component * temporal_prod_component + (1 - temporal_prod_component) * temporal_component,
            spatial_component,
        )


class STModuleTemporalMask(nn.Module):
    def __init__(
        self,
        n_output_dim,
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
        st_mlp_mode="independent",
        use_linear=False,
        spatial_only=False,
        interp="linear",
        level_one_interp="linear",
        nosigmoid=False,
        ablation_add=False,
        mask_init_mean=0.0,
        mask_multi_level=False,
    ):
        super().__init__()
        assert st_mlp_mode == "independent" or st_mlp_mode == "shared"
        self.ablation_add = ablation_add
        self.st_mlp_mode = st_mlp_mode
        self.spatial_only = spatial_only
        if st_mlp_mode == "independent":
            self.spatial_net = nn.Sequential(
                GridEncoder(
                    input_dim=3,
                    num_levels=num_levels,
                    level_dim=features_per_level,
                    per_level_scale=per_level_scale[:3],
                    base_resolution=base_res[:3],
                    log2_hashmap_size=log2_hashmap_size_spatial,
                    std=1e-4,
                    interpolation=interp,
                ),
                nn.Linear(features_per_level * num_levels, n_output_dim)
                if use_linear
                else tcnn.Network(
                    n_input_dims=features_per_level * num_levels,
                    n_output_dims=n_output_dim,
                    network_config={
                        "otype": "FullyFusedMLP",
                        "activation": "ReLU",
                        "output_activation": "None",
                        "n_neurons": hidden_dim,
                        "n_hidden_layers": num_layers - 1,
                    },
                ),
            )

            self.temporal_net = nn.Sequential(
                GridEncoder(
                    input_dim=4,
                    num_levels=num_levels,
                    level_dim=features_per_level,
                    per_level_scale=per_level_scale,
                    base_resolution=base_res,
                    log2_hashmap_size=log2_hashmap_size_temporal,
                    std=1e-4,
                    interpolation=interp,
                ),
                nn.Linear(features_per_level * num_levels, n_output_dim)
                if use_linear
                else tcnn.Network(
                    n_input_dims=features_per_level * num_levels,
                    n_output_dims=n_output_dim,
                    network_config={
                        "otype": "FullyFusedMLP",
                        "activation": "ReLU",
                        "output_activation": "None",
                        "n_neurons": hidden_dim,
                        "n_hidden_layers": num_layers - 1,
                    },
                ),
            )
        else:
            self.spatial_net = GridEncoder(
                input_dim=3,
                num_levels=num_levels,
                level_dim=features_per_level,
                per_level_scale=per_level_scale[:3],
                base_resolution=base_res[:3],
                log2_hashmap_size=log2_hashmap_size_spatial,
                std=1e-4,
                interpolation=interp,
            )

            self.temporal_net = GridEncoder(
                input_dim=4,
                num_levels=num_levels,
                level_dim=features_per_level,
                per_level_scale=per_level_scale,
                base_resolution=base_res,
                log2_hashmap_size=log2_hashmap_size_temporal,
                std=1e-4,
                interpolation=interp,
            )

            self.mlp_shared = (
                nn.Linear(features_per_level * num_levels, n_output_dim)
                if use_linear
                else tcnn.Network(
                    n_input_dims=features_per_level * num_levels,
                    n_output_dims=n_output_dim,
                    network_config={
                        "otype": "FullyFusedMLP",
                        "activation": "ReLU",
                        "output_activation": "None",
                        "n_neurons": hidden_dim,
                        "n_hidden_layers": num_layers - 1,
                    },
                )
            )
        self.temporal_prod_net = GridEncoder(
            input_dim=4,
            num_levels=1,
            level_dim=1,
            per_level_scale=1,
            base_resolution=mask_reso,
            log2_hashmap_size=mask_log2_hash_size,
            std=1e-4,
            gridtype="tiled",
            interpolation="last_nearest",
            # interpolation="all_nearest",
            mean=mask_init_mean,
        )
        self.nosigmoid = nosigmoid
        self.mode = mode

    #
    def forward(self, x):
        spatial_component = self.spatial_net(x[..., :3])
        if self.spatial_only:
            return spatial_component, spatial_component
        # torch.cuda.synchronize("cuda")
        temporal_component = self.temporal_net(x)

        # torch.cuda.synchronize("cuda")
        temporal_prod_component = self.temporal_prod_net(x)
        if not self.nosigmoid:
            temporal_prod_component = temporal_prod_component.sigmoid()
        # torch.cuda.synchronize("cuda")

        if self.ablation_add:
            # print("spatial add temporal")
            output = spatial_component + temporal_component
            if self.st_mlp_mode == "shared":
                # print("shared")
                output = self.mlp_shared(output)
                # spatial_component = self.mlp_shared(spatial_component)
                spatial_component = None
            return (
                output,
                # spatial_component * temporal_prod_component + (1 - temporal_prod_component) * temporal_component,
                spatial_component,
            )

        if self.mode == "mst":
            output = spatial_component * temporal_prod_component + (1 - temporal_prod_component) * temporal_component
        else:
            assert self.mode == "mt"
            output = spatial_component + (1 - temporal_prod_component) * temporal_component

        if self.st_mlp_mode == "shared":
            output = self.mlp_shared(output)
            spatial_component = self.mlp_shared(spatial_component)
            # spatial_component = None

        return (
            output,
            # spatial_component * temporal_prod_component + (1 - temporal_prod_component) * temporal_component,
            spatial_component,
        )


class STModuleWithTimeQuery(nn.Module):
    def __init__(
        self,
        n_output_dim,
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
        st_mlp_mode="independent",
        use_linear=False,
        spatial_only=False,
        interp="linear",
        level_one_interp="linear",
        nosigmoid=False,
        ablation_add=False,
        mask_init_mean=0.0,
        mask_multi_level=False,
    ):
        super().__init__()
        assert st_mlp_mode == "independent" or st_mlp_mode == "shared"
        self.ablation_add = ablation_add
        self.st_mlp_mode = st_mlp_mode
        self.spatial_only = spatial_only
        self.time_query = tcnn.Encoding(
            n_input_dims=1,
            encoding_config={"otype": "Frequency", "n_frequencies": 6},
        )
        assert st_mlp_mode == "shared", "only for shared arch"
        if st_mlp_mode == "independent":
            self.spatial_net = nn.Sequential(
                GridEncoder(
                    input_dim=3,
                    num_levels=num_levels,
                    level_dim=features_per_level,
                    per_level_scale=per_level_scale[:3],
                    base_resolution=base_res[:3],
                    log2_hashmap_size=log2_hashmap_size_spatial,
                    std=1e-4,
                    interpolation=interp,
                ),
                nn.Linear(features_per_level * num_levels, n_output_dim)
                if use_linear
                else tcnn.Network(
                    n_input_dims=features_per_level * num_levels,
                    n_output_dims=n_output_dim,
                    network_config={
                        "otype": "FullyFusedMLP",
                        "activation": "ReLU",
                        "output_activation": "None",
                        "n_neurons": hidden_dim,
                        "n_hidden_layers": num_layers - 1,
                    },
                ),
            )

            self.temporal_net = nn.Sequential(
                GridEncoder(
                    input_dim=4,
                    num_levels=num_levels,
                    level_dim=features_per_level,
                    per_level_scale=per_level_scale,
                    base_resolution=base_res,
                    log2_hashmap_size=log2_hashmap_size_temporal,
                    std=1e-4,
                    interpolation=interp,
                ),
                nn.Linear(features_per_level * num_levels, n_output_dim)
                if use_linear
                else tcnn.Network(
                    n_input_dims=features_per_level * num_levels,
                    n_output_dims=n_output_dim,
                    network_config={
                        "otype": "FullyFusedMLP",
                        "activation": "ReLU",
                        "output_activation": "None",
                        "n_neurons": hidden_dim,
                        "n_hidden_layers": num_layers - 1,
                    },
                ),
            )
        else:
            self.spatial_net = GridEncoder(
                input_dim=3,
                num_levels=num_levels,
                level_dim=features_per_level,
                per_level_scale=per_level_scale[:3],
                base_resolution=base_res[:3],
                log2_hashmap_size=log2_hashmap_size_spatial,
                std=1e-4,
                interpolation=interp,
            )

            self.temporal_net = GridEncoder(
                input_dim=4,
                num_levels=num_levels,
                level_dim=features_per_level,
                per_level_scale=per_level_scale,
                base_resolution=base_res,
                log2_hashmap_size=log2_hashmap_size_temporal,
                std=1e-4,
                interpolation=interp,
            )

            self.mlp_shared = (
                nn.Linear(features_per_level * num_levels + self.time_query.n_output_dims, n_output_dim)
                if use_linear
                else tcnn.Network(
                    n_input_dims=features_per_level * num_levels + self.time_query.n_output_dims,
                    n_output_dims=n_output_dim,
                    network_config={
                        "otype": "FullyFusedMLP",
                        "activation": "ReLU",
                        "output_activation": "None",
                        "n_neurons": hidden_dim,
                        "n_hidden_layers": num_layers - 1,
                    },
                )
            )
        if mask_multi_level:
            self.temporal_prod_net = GridEncoder(
                input_dim=3,
                num_levels=num_levels,
                level_dim=1,
                per_level_scale=per_level_scale[:3],
                base_resolution=base_res[:3],
                log2_hashmap_size=mask_log2_hash_size,
                std=1e-4,
                gridtype="hash",
                interpolation=interp,
                # interpolation="all_nearest",
                mean=mask_init_mean,
            )
        else:
            self.temporal_prod_net = GridEncoder(
                input_dim=3,
                num_levels=1,
                level_dim=1,
                per_level_scale=1,
                base_resolution=mask_reso,
                log2_hashmap_size=mask_log2_hash_size,
                std=1e-4,
                gridtype="tiled",
                interpolation=interp,
                # interpolation="all_nearest",
                mean=mask_init_mean,
            )
        self.nosigmoid = nosigmoid
        self.mode = mode

    #
    def forward(self, x, temporal_only=False):
        spatial_component = self.spatial_net(x[..., :3])
        if self.spatial_only:
            return spatial_component, spatial_component
        # torch.cuda.synchronize("cuda")
        temporal_component = self.temporal_net(x)

        time_embedding = self.time_query(x[..., 3:4])
        # torch.cuda.synchronize("cuda")
        temporal_prod_component = self.temporal_prod_net(x[..., :3])
        if not self.nosigmoid:
            temporal_prod_component = temporal_prod_component.sigmoid()

        if not self.training:
            self.mask_detached = temporal_prod_component.detach()
        else:
            self.mask_detached = None
        # torch.cuda.synchronize("cuda")

        if self.ablation_add:
            # print("spatial add temporal")
            output = spatial_component + temporal_component
            if self.st_mlp_mode == "shared":
                # print("shared")
                output = self.mlp_shared(output)
                # spatial_component = self.mlp_shared(spatial_component)
                spatial_component = None
            return (
                output,
                # spatial_component * temporal_prod_component + (1 - temporal_prod_component) * temporal_component,
                spatial_component,
            )

        if self.mode == "mst":
            output = spatial_component * temporal_prod_component + (1 - temporal_prod_component) * temporal_component
        else:
            assert self.mode == "mt"
            output = spatial_component + (1 - temporal_prod_component) * temporal_component

        if self.st_mlp_mode == "shared":
            output = self.mlp_shared(torch.cat([output, time_embedding], dim=-1))
            # spatial_component = self.mlp_shared(spatial_component)
            spatial_component = None

        return (
            output,
            # spatial_component * temporal_prod_component + (1 - temporal_prod_component) * temporal_component,
            spatial_component,
        )
