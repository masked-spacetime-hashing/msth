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
        sembeddings,
        tembeddings,
        membeddings,
        soffsets,
        toffsets,
        per_level_scale,
        base_resolution,
        mask_resolution,
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
        L = toffsets.shape[0] - 1  # level
        C = tembeddings.shape[1]  # embedding dim for each level
        assert isinstance(per_level_scale, float) or isinstance(per_level_scale, torch.Tensor)
        is_rect = isinstance(per_level_scale, torch.Tensor)
        if not is_rect:
            S = np.log2(per_level_scale)  # resolution multiplier at each level, apply log2 for later CUDA exp2f
        else:
            S = torch.log2(per_level_scale)

        H = base_resolution  # base resolution
        M = mask_resolution

        # manually handle autocast (only use half precision embeddings, inputs must be float for enough precision)
        # if C % 2 != 0, force float, since half for atomicAdd is very slow.
        if torch.is_autocast_enabled() and C % 2 == 0:
            sembeddings = sembeddings.to(torch.half)
            tembeddings = tembeddings.to(torch.half)
            membeddings = membeddings.to(torch.half)

        # L first, optimize cache for cuda kernel, but needs an extra permute later
        outputs = torch.empty(L, B, C, device=inputs.device, dtype=sembeddings.dtype)
        tout = torch.empty(L, B, C, device=inputs.device, dtype=sembeddings.dtype)
        mout = torch.empty(B, device=inputs.device, dtype=sembeddings.dtype)

        if calc_grad_inputs:
            dy_dx = torch.empty(B, L * D * C, device=inputs.device, dtype=sembeddings.dtype)
        else:
            dy_dx = None

        _backend.stgrid_encode_forward(
            inputs,
            sembeddings,
            tembeddings,
            membeddings,
            soffsets,
            toffsets,
            outputs,
            tout,
            mout,
            B,
            D,
            C,
            L,
            S,
            H,
            M,
            dy_dx,
            gridtype,
            align_corners,
            interpolation,
        )

        # permute back to [B, L * C]
        outputs = outputs.permute(1, 0, 2).reshape(B, L * C)
        tout = tout.permute(1, 0, 2).reshape(B, L * C)
        print(tout / (1 - mout.unsqueeze(-1)))
        print(mout)

        ctx.save_for_backward(
            inputs, sembeddings, tembeddings, membeddings, soffsets, toffsets, tout, mout, dy_dx, grid_mask
        )
        ctx.dims = [B, D, C, L, gridtype, interpolation]
        ctx.align_corners = align_corners
        ctx.H = H
        ctx.M = M
        ctx.S = S

        return outputs

    @staticmethod
    # @once_differentiable
    @custom_bwd
    def backward(ctx, grad):
        (
            inputs,
            sembeddings,
            tembeddings,
            membeddings,
            soffsets,
            toffsets,
            tout,
            mout,
            dy_dx,
            grid_mask,
        ) = ctx.saved_tensors
        B, D, C, L, gridtype, interpolation = ctx.dims
        align_corners = ctx.align_corners
        S = ctx.S
        H = ctx.H
        M = ctx.M

        # grad: [B, L * C] --> [L, B, C]
        grad = grad.view(B, L, C).permute(1, 0, 2).contiguous()

        grad_sembeddings = torch.zeros_like(sembeddings)
        grad_tembeddings = torch.zeros_like(tembeddings)
        grad_membeddings = torch.zeros_like(membeddings)

        if dy_dx is not None:
            grad_inputs = torch.zeros_like(inputs, dtype=embeddings.dtype)
        else:
            grad_inputs = None

        _backend.stgrid_encode_backward(
            grad,
            inputs,
            tout,
            mout,
            soffsets,
            toffsets,
            grad_sembeddings,
            grad_tembeddings,
            grad_membeddings,
            B,
            D,
            C,
            L,
            S,
            H,
            M,
            dy_dx,
            grad_inputs,
            gridtype,
            align_corners,
            interpolation,
        )

        if dy_dx is not None:
            grad_inputs = grad_inputs.to(inputs.dtype)

        # if grid_mask is not None:
        # grad_embeddings = grad_embeddings * grid_mask.unsqueeze(dim=-1).to(grad_embeddings)

        return (
            grad_inputs,
            grad_sembeddings,
            grad_tembeddings,
            grad_membeddings,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


grid_encode = _grid_encode.apply


class SpatialTemporalGridEncoder(nn.Module):
    def __init__(
        self,
        input_dim=4,
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
        mask_resolution=(128, 128, 128),
        std=1e-4,
    ):
        super().__init__()
        # the finest resolution desired at the last level, if provided, overridee per_level_scale

        per_level_scale = np.array(per_level_scale)
        base_resolution = np.array(base_resolution)
        mask_resolution = np.array(mask_resolution)

        if desired_resolution is not None:
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
        soffsets = []
        soffset = 0
        self.max_params = 2**log2_hashmap_size
        for i in range(num_levels):
            sresolution = np.ceil(base_resolution[:3] * per_level_scale[:3] ** i).astype(int)
            if not align_corners:
                sresolution += 1
            params_in_level = min(self.max_params, np.cumprod(sresolution)[-1])
            params_in_level = (np.ceil(params_in_level / 8) * 8).astype(int)  # make divisible
            soffsets.append(soffset)
            soffset += params_in_level

        soffsets.append(soffset)
        soffsets = torch.from_numpy(np.array(soffsets, dtype=np.int32))
        self.register_buffer("soffsets", soffsets)

        self.n_params_s = soffsets[-1] * level_dim
        # parameters
        self.sembeddings = nn.Parameter(torch.empty(soffset, level_dim))

        toffsets = []
        toffset = 0
        # self.max_params = 2**log2_hashmap_size
        for i in range(num_levels):
            tresolution = np.ceil(base_resolution * per_level_scale**i).astype(int)
            if not align_corners:
                tresolution += 1
            params_in_level = min(self.max_params, np.cumprod(tresolution)[-1])
            params_in_level = (np.ceil(params_in_level / 8) * 8).astype(int)  # make divisible
            toffsets.append(toffset)
            toffset += params_in_level

        toffsets.append(toffset)
        toffsets = torch.from_numpy(np.array(toffsets, dtype=np.int32))
        self.register_buffer("toffsets", toffsets)

        self.n_params_t = toffsets[-1] * level_dim
        # parameters
        self.tembeddings = nn.Parameter(torch.empty(toffset, level_dim))

        self.membeddings = nn.Parameter(torch.empty(np.cumprod(mask_resolution)[-1]))

        self.grid_mask = None

        self.register_buffer("per_level_scale", torch.from_numpy(per_level_scale).to(torch.float))
        self.register_buffer("base_resolution", torch.from_numpy(base_resolution).to(torch.int32))
        self.register_buffer("mask_resolution", torch.from_numpy(mask_resolution).to(torch.int32))

        self.std = std

        self.reset_parameters()

    def reset_parameters(self):
        # std = 1e-4
        self.sembeddings.data.uniform_(-self.std, self.std)
        self.tembeddings.data.uniform_(-self.std, self.std)
        self.membeddings.data.uniform_(-self.std, self.std)

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
            self.sembeddings,
            self.tembeddings,
            self.membeddings,
            self.soffsets,
            self.toffsets,
            self.per_level_scale,
            self.base_resolution,
            self.mask_resolution,
            inputs.requires_grad,
            self.gridtype_id,
            self.align_corners,
            self.interp_id,
            self.grid_mask,
        )

        outputs = outputs.view(prefix_shape + [self.output_dim])

        # print('outputs', outputs.shape, outputs.dtype, outputs.min().item(), outputs.max().item())

        return outputs
