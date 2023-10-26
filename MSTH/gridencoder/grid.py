import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd
import torch.nn.functional as F
from einops import rearrange

try:
    import _gridencoder as _backend
except ImportError:
    from .backend import _backend

_gridtype_to_id = {
    "hash": 0,
    "tiled": 1,
}

_interp_to_id = {
    "linear": 0,
    "smoothstep": 1,
    "last_nearest": 2,
    "all_nearest": 3,
}


class _grid_encode(Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        inputs,
        embeddings,
        offsets,
        per_level_scale,
        base_resolution,
        calc_grad_inputs=False,
        gridtype=0,
        align_corners=False,
        interpolation=0,
        grid_mask=None,
    ):
        # inputs: [B, D], float in [0, 1]
        # embeddings: [sO, C], float
        # offsets: [L + 1], int
        # RETURN: [B, F], float

        inputs = inputs.contiguous()

        B, D = inputs.shape  # batch size, coord dim
        L = offsets.shape[0] - 1  # level
        C = embeddings.shape[1]  # embedding dim for each level
        assert isinstance(per_level_scale, float) or isinstance(per_level_scale, torch.Tensor)
        is_rect = isinstance(per_level_scale, torch.Tensor)
        if not is_rect:
            S = np.log2(per_level_scale)  # resolution multiplier at each level, apply log2 for later CUDA exp2f
        else:
            S = torch.log2(per_level_scale)

        H = base_resolution  # base resolution

        # manually handle autocast (only use half precision embeddings, inputs must be float for enough precision)
        # if C % 2 != 0, force float, since half for atomicAdd is very slow.
        if torch.is_autocast_enabled() and C % 2 == 0:
            embeddings = embeddings.to(torch.half)

        # L first, optimize cache for cuda kernel, but needs an extra permute later
        outputs = torch.empty(L, B, C, device=inputs.device, dtype=embeddings.dtype)

        if calc_grad_inputs:
            dy_dx = torch.empty(B, L * D * C, device=inputs.device, dtype=embeddings.dtype)
        else:
            dy_dx = None

        if not is_rect:
            _backend.grid_encode_forward(
                inputs, embeddings, offsets, outputs, B, D, C, L, S, H, dy_dx, gridtype, align_corners, interpolation
            )
        else:
            _backend.rect_grid_encode_forward(
                inputs, embeddings, offsets, outputs, B, D, C, L, S, H, dy_dx, gridtype, align_corners, interpolation
            )

        # permute back to [B, L * C]
        outputs = outputs.permute(1, 0, 2).reshape(B, L * C)

        ctx.save_for_backward(inputs, embeddings, offsets, dy_dx, grid_mask)
        ctx.dims = [B, D, C, L, gridtype, interpolation]
        ctx.align_corners = align_corners
        ctx.H = H
        ctx.S = S
        ctx.is_rect = is_rect

        return outputs

    @staticmethod
    # @once_differentiable
    @custom_bwd
    def backward(ctx, grad):
        inputs, embeddings, offsets, dy_dx, grid_mask = ctx.saved_tensors
        B, D, C, L, gridtype, interpolation = ctx.dims
        align_corners = ctx.align_corners
        S = ctx.S
        H = ctx.H
        is_rect = ctx.is_rect

        # grad: [B, L * C] --> [L, B, C]
        grad = grad.view(B, L, C).permute(1, 0, 2).contiguous()

        grad_embeddings = torch.zeros_like(embeddings)

        if dy_dx is not None:
            grad_inputs = torch.zeros_like(inputs, dtype=embeddings.dtype)
        else:
            grad_inputs = None

        if not is_rect:
            _backend.grid_encode_backward(
                grad,
                inputs,
                embeddings,
                offsets,
                grad_embeddings,
                B,
                D,
                C,
                L,
                S,
                H,
                dy_dx,
                grad_inputs,
                gridtype,
                align_corners,
                interpolation,
            )
        else:
            _backend.rect_grid_encode_backward(
                grad,
                inputs,
                embeddings,
                offsets,
                grad_embeddings,
                B,
                D,
                C,
                L,
                S,
                H,
                dy_dx,
                grad_inputs,
                gridtype,
                align_corners,
                interpolation,
            )

        if dy_dx is not None:
            grad_inputs = grad_inputs.to(inputs.dtype)

        if grid_mask is not None:
            grad_embeddings = grad_embeddings * grid_mask.unsqueeze(dim=-1).to(grad_embeddings)

        return grad_inputs, grad_embeddings, None, None, None, None, None, None, None, None


grid_encode = _grid_encode.apply
grid_encode_hash_reinitialize = _backend.grid_encode_hash_reinitialize
grid_encode_set_static = _backend.grid_encode_set_static


class GridEncoder(nn.Module):
    def __init__(
        self,
        input_dim=3,
        num_levels=16,
        level_dim=2,
        per_level_scale=2,
        base_resolution=16,
        log2_hashmap_size=19,
        desired_resolution=None,
        gridtype="hash",
        align_corners=False,
        interpolation="linear",
        specific_resolution_for_each_level=None,
        std=1e-4,
        mean=0.0,
        customized_t_reso=None,
    ):
        super().__init__()

        print("=" * 100)
        print(interpolation)
        # the finest resolution desired at the last level, if provided, overridee per_level_scale
        is_rect = isinstance(base_resolution, (list, tuple))
        if is_rect:
            per_level_scale = np.array(per_level_scale)
            base_resolution = np.array(base_resolution)
        if desired_resolution is not None:
            if is_rect:
                desired_resolution = np.array(desired_resolution)
            per_level_scale = np.exp2(np.log2(desired_resolution / base_resolution) / (num_levels - 1))

        self.input_dim = input_dim  # coord dims, 2 or 3
        self.num_levels = num_levels  # num levels, each level multiply resolution by 2
        self.level_dim = level_dim  # encode channels per level
        # self.per_level_scale = per_level_scale # multiply resolution by this scale at each level.
        self.log2_hashmap_size = log2_hashmap_size
        # self.base_resolution = base_resolution
        self.output_dim = num_levels * level_dim
        self.gridtype = gridtype
        self.gridtype_id = _gridtype_to_id[gridtype]  # "tiled" or "hash"
        self.interpolation = interpolation
        self.interp_id = _interp_to_id[interpolation]  # "linear" or "smoothstep"
        self.align_corners = align_corners

        # allocate parameters
        offsets = []
        offset = 0
        self.max_params = 2**log2_hashmap_size
        for i in range(num_levels):
            if is_rect:
                resolution = np.ceil(base_resolution * per_level_scale**i).astype(int)
                if customized_t_reso is not None:
                    resolution[3] = int(customized_t_reso(i))
                    print("t reso overrided !")
                print(resolution, f"collide: {np.prod(resolution)/np.exp2(self.log2_hashmap_size):.2g}")
                if not align_corners:
                    resolution += 1
                params_in_level = min(self.max_params, np.cumprod(resolution)[-1])
                params_in_level = (np.ceil(params_in_level / 8) * 8).astype(int)  # make divisible
            else:
                resolution = int(np.ceil(base_resolution * per_level_scale**i))
                if not align_corners:
                    resolution += 1
                params_in_level = min(self.max_params, resolution**input_dim)  # limit max number
                params_in_level = int(np.ceil(params_in_level / 8) * 8)  # make divisible

            offsets.append(offset)
            offset += params_in_level

        offsets.append(offset)
        offsets = torch.from_numpy(np.array(offsets, dtype=np.int32))
        self.register_buffer("offsets", offsets)

        self.n_params = offsets[-1] * level_dim

        # parameters
        self.embeddings = nn.Parameter(torch.empty(offset, level_dim))
        self.grid_mask = None

        if is_rect:
            self.register_buffer("per_level_scale", torch.from_numpy(per_level_scale).to(torch.float))
            self.register_buffer("base_resolution", torch.from_numpy(base_resolution).to(torch.int32))
        else:
            self.per_level_scale = per_level_scale  # multiply resolution by this scale at each level.
            self.base_resolution = base_resolution

        self.is_rect = is_rect
        self.std = std
        self.mean = mean
        self.reset_parameters()

    def get_mask_loss(self):
        return (1 - self.embeddings.sigmoid()).mean()

    @torch.no_grad()
    def upsample(self, resolution=None):
        assert self.num_levels == 1, "Only support upsampling voxels with a single level"
        assert self.gridtype == "tiled", "Only support upsampling voxels for tiled grid"
        params_in_level = (resolution if self.align_corners else resolution + 1) ** self.input_dim
        params_in_level = int(np.ceil(params_in_level / 8) * 8)  # make divisible
        offsets = [0, params_in_level]
        offsets = torch.from_numpy(np.array(offsets, dtype=np.int32)).to(self.offsets)
        self.register_buffer("offsets", offsets)
        print(self.offsets)
        # self.offsets = offsets
        self.n_params = offsets[-1] * self.level_dim

        real_base_reso = self.base_resolution if self.align_corners else (self.base_resolution + 1)
        embeddings = self.embeddings[: real_base_reso**self.input_dim, :]
        embeddings = embeddings.reshape((real_base_reso, real_base_reso, real_base_reso, self.level_dim))
        embeddings = rearrange(embeddings, "x y z c -> c x y z").unsqueeze(dim=0)

        upsampled_embeddings = F.interpolate(
            input=embeddings.data,
            size=(
                resolution if self.align_corners else resolution + 1,
                resolution if self.align_corners else resolution + 1,
                resolution if self.align_corners else resolution + 1,
            ),
            mode="trilinear",
        )
        upsampled_embeddings = rearrange(upsampled_embeddings, "b c x y z -> (b x y z) c")
        new_embeddings = torch.zeros(offsets[-1], self.level_dim).to(upsampled_embeddings)
        new_embeddings[: upsampled_embeddings.shape[0], :] = upsampled_embeddings

        self.embeddings = nn.Parameter(new_embeddings)
        print(self.embeddings.requires_grad)
        self.base_resolution = resolution

    def reset_parameters(self):
        # std = 1e-4
        self.embeddings.data.uniform_(-self.std + self.mean, self.std + self.mean)

    def __repr__(self):
        return f"GridEncoder: input_dim={self.input_dim} num_levels={self.num_levels} level_dim={self.level_dim} resolution={self.base_resolution} -> {int(round(self.base_resolution * self.per_level_scale ** (self.num_levels - 1)))} per_level_scale={self.per_level_scale:.4f} params={tuple(self.embeddings.shape)} gridtype={self.gridtype} align_corners={self.align_corners} interpolation={self.interpolation}"

    def forward(self, inputs, bound=1):
        # inputs: [..., input_dim], normalized real world positions in [-bound, bound]
        # return: [..., num_levels * level_dim]

        # inputs = (inputs + bound) / (2 * bound) # map to [0, 1]

        # print('inputs', inputs.shape, inputs.dtype, inputs.min().item(), inputs.max().item())

        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.view(-1, self.input_dim)

        outputs = grid_encode(
            inputs,
            self.embeddings,
            self.offsets,
            self.per_level_scale,
            self.base_resolution,
            inputs.requires_grad,
            self.gridtype_id,
            self.align_corners,
            self.interp_id,
            self.grid_mask,
        )
        outputs = outputs.view(prefix_shape + [self.output_dim])

        # print('outputs', outputs.shape, outputs.dtype, outputs.min().item(), outputs.max().item())

        return outputs

    def hash_reinitialize(self, inputs, std=1e-4):
        # inputs = (inputs + bound) / (2 * bound) # map to [0, 1]
        assert self.grid_mask is not None, "hash reinitialize should be excuted after generate grid mask"

        # if self.grid_mask is None:
        # self.grid_mask = torch.ones(self.offsets[-1]).bool().cuda()

        inputs = inputs.view(-1, self.input_dim)
        B = inputs.shape[0]
        grid_encode_hash_reinitialize(
            inputs,
            self.embeddings,
            self.offsets,
            B,
            self.input_dim,
            self.embeddings.shape[1],
            self.offsets.shape[0] - 1,
            np.log2(self.per_level_scale),
            self.base_resolution,
            self.gridtype_id,
            self.align_corners,
            0,
            std,
            self.grid_mask,
        )

    def generate_grid_mask(self, inputs, bound=1):
        # inputs = (inputs + bound) / (2 * bound) # map to [0, 1]
        if self.grid_mask is None:
            self.grid_mask = torch.ones(self.offsets[-1]).bool().cuda()

        inputs = inputs.view(-1, self.input_dim)
        B = inputs.shape[0]
        # print(inputs)
        grid_encode_set_static(
            inputs,
            self.grid_mask,
            self.offsets,
            B,
            self.input_dim,
            self.embeddings.shape[1],
            self.offsets.shape[0] - 1,
            np.log2(self.per_level_scale),
            self.base_resolution,
            self.gridtype_id,
            self.align_corners,
        )

        # print(f"setting stat: {(self.grid_mask==0).sum()}")

    # always run in float precision!
    @torch.cuda.amp.autocast(enabled=False)
    def grad_total_variation(self, weight=1e-7, inputs=None, bound=1, B=1000000):
        # inputs: [..., input_dim], float in [-b, b], location to calculate TV loss.

        D = self.input_dim
        C = self.embeddings.shape[1]  # embedding dim for each level
        L = self.offsets.shape[0] - 1  # level
        if self.is_rect:
            S = torch.log2(self.per_level_scale)  # resolution multiplier at each level, apply log2 for later CUDA exp2f
        else:
            S = np.log2(self.per_level_scale)  # resolution multiplier at each level, apply log2 for later CUDA exp2f
        H = self.base_resolution  # base resolution

        if inputs is None:
            # randomized in [0, 1]
            inputs = torch.rand(B, self.input_dim, device=self.embeddings.device)
        else:
            # inputs = (inputs + bound) / (2 * bound) # map to [0, 1]
            inputs = inputs.view(-1, self.input_dim)
            B = inputs.shape[0]

        if self.embeddings.grad is None:
            raise ValueError("grad is None, should be called after loss.backward() and before optimizer.step()!")

        if self.is_rect:
            _backend.rect_grad_total_variation(
                inputs,
                self.embeddings,
                self.embeddings.grad,
                self.offsets,
                weight,
                B,
                D,
                C,
                L,
                S,
                H,
                self.gridtype_id,
                self.align_corners,
            )
        else:
            _backend.grad_total_variation(
                inputs,
                self.embeddings,
                self.embeddings.grad,
                self.offsets,
                weight,
                B,
                D,
                C,
                L,
                S,
                H,
                self.gridtype_id,
                self.align_corners,
            )
