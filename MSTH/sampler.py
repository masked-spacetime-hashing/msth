from typing import Dict, Optional, Union
import numpy as np
import gc
import torch
from torchtyping import TensorType
from MSTH.dataset import VideoDatasetAllCached, VideoDatasetAllCachedUint8
from nerfstudio.data.pixel_samplers import PixelSampler
import torch.nn.functional as F
from einops import repeat, rearrange

from rich.console import Console

CONSOLE = Console(width=120)

# class CompletePixelSampler:
#     def __init__(self, num_rays_per_batch: int, ensure_all_sampled: bool, drop_last: bool = False) -> None:
#         self.num_rays_per_batch = num_rays_per_batch
#         self.sample_ptr = -1
#         self.ensure_all_sampled = ensure_all_sampled

#     def _reset(self):
#         assert self.all_indices.size(0) > 0
#         self.all_indices = self.all_indices[torch.randperm(self.all_indices.size(0))]
#         self.sample_ptr = 0

#     def set_image_batch(self, batch):
#         """set images to be sampled"""
#         images = batch["image"]
#         masks = batch["mask"]
#         self.images = images
#         self.masks = masks
#         self.all_indices = torch.nonzero(masks[..., 0], as_tuple=False)  # [num_unmasked_pixels, 3]
#         self._reset()

#     def sample_indices(self) -> TensorType["num_rays_per_batch", 3]:
#         if self.ensure_all_sampled:
#             if self.sample_ptr + self.num_rays_per_batch > self.all_indices.size(0):
#                 self._reset()
#                 ret = self.all_indices[self.sample_ptr : self.sample_ptr + self.num_rays_per_batch]
#                 self.sample_ptr += self.num_rays_per_batch

#                 return ret

#         else:
#             self._reset()
#             return self.all_indices[self.sample_ptr : self.sample_ptr + self.num_rays_per_batch]

#     def collate_image_dataset_batch(self, batch: Dict, num_rays_per_batch: int):
#         indices = self.sample_indices()
#         c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
#         # collated_batch = {
#         #     key: value[c, y, x] for key, value in batch.items() if key != "image_idx" and value is not None
#         # }
#         collated_batch = {
#             "image": self.images[c, y, x],
#             "mask": self.masks[c, y, x],
#         }

#         assert collated_batch["image"].shape == (num_rays_per_batch, 3), collated_batch["image"].shape

#         # Needed to correct the random indices to their actual camera idx locations.
#         # indices[:, 0] = batch["image_idx"][c]
#         collated_batch["indices"] = indices  # with the abs camera indices

#         return collated_batch

#     def sample(self, image_batch: Dict):
#         if image_batch["image"] is not self.images:
#             self.set_image_batch(image_batch)
#         return self.collate_image_dataset_batch(image_batch, self.num_rays_per_batch)


class CompletePixelSamplerIter:
    def __init__(self, num_rays_per_batch: int, image_batch: Dict, drop_last: bool = False) -> None:
        self.num_rays_per_batch = num_rays_per_batch
        self.images = image_batch["image"]
        self.masks = image_batch["mask"]

        self.sample_ptr = 0
        self.all_indices = torch.nonzero(self.masks[..., 0], as_tuple=False)
        self.drop_last = drop_last

    def set_batch(self, batch):
        """update images and masks and corresponding non-zero indices"""
        self.images = batch["image"]
        self.masks = batch["mask"]
        self.sample_ptr = 0
        self.all_indices = torch.nonzero(self.masks[..., 0], as_tuple=False)

    def get_batch(self, indices: TensorType["num_rays_per_batch", 3]):
        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        # collated_batch = {
        #     key: value[c, y, x] for key, value in batch.items() if key != "image_idx" and value is not None
        # }
        collated_batch = {
            "image": self.images[c, y, x],
            "mask": self.masks[c, y, x],
        }

        # assert collated_batch["image"].shape == (self.num_rays_per_batch, 3), collated_batch["image"].shape

        # Needed to correct the random indices to their actual camera idx locations.
        # TODO: check this, this is correct since all the camera are loader in the same order
        # indices[:, 0] = batch["image_idx"][c]
        collated_batch["indices"] = c  # with the abs camera indices

        return collated_batch

    def __iter__(self):
        self.all_indices = self.all_indices[torch.randperm(self.all_indices.size(0))]
        self.sample_ptr = 0
        return self

    def __next__(self):
        if self.sample_ptr >= self.all_indices.size(0):
            raise StopIteration
        if self.sample_ptr + self.num_rays_per_batch > self.all_indices.size(0) and self.drop_last:
            raise StopIteration

        sample_ptr_end = min(self.sample_ptr + self.num_rays_per_batch, self.all_indices.size(0))

        indices = self.all_indices[self.sample_ptr : sample_ptr_end]

        self.sample_ptr += self.num_rays_per_batch

        return self.get_batch(indices)


class CompletePixelSampler:
    def __init__(
        self, num_rays_per_batch: int, image_batch: Dict, drop_last: bool = False, use_mask: bool = True
    ) -> None:
        self.num_rays_per_batch = num_rays_per_batch
        self.images = image_batch["image"]
        self.masks = image_batch["mask"]

        self.sample_ptr = 0
        self.sample_ptr_invert = 0
        self.use_mask = use_mask
        if self.use_mask:
            self.all_indices = torch.nonzero(self.masks[..., 0], as_tuple=False)
            self.all_indices = self.all_indices[torch.randperm(self.all_indices.size(0))]

            self.all_indices_inverted = torch.nonzero(1 - self.masks[..., 0], as_tuple=False)
            self.all_indices_inverted = self.all_indices_inverted[torch.randperm(self.all_indices_inverted.size(0))]

        self.drop_last = drop_last
        self.use_mask = use_mask

    def set_batch(self, batch):
        """update images and masks and corresponding non-zero indices"""
        self.images = batch["image"]
        self.masks = batch["mask"]
        self.sample_ptr = 0
        self.sample_ptr_invert = 0
        if self.use_mask:
            self.all_indices = torch.nonzero(self.masks[..., 0], as_tuple=False)
            self.all_indices_inverted = torch.nonzero(1 - self.masks[..., 0], as_tuple=False)

    def get_batch(self, indices: TensorType["num_rays_per_batch", 3]):
        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        # collated_batch = {
        #     key: value[c, y, x] for key, value in batch.items() if key != "image_idx" and value is not None
        # }
        try:
            collated_batch = {
                "image": self.images[c, y, x],
                "mask": self.masks[c, y, x],
            }
        except:
            print(c, y, x)
            exit()

        # Needed to correct the random indices to their actual camera idx locations.
        # TODO: check this, this is correct since all the camera are loader in the same order
        # indices[:, 0] = batch["image_idx"][c]
        collated_batch["indices"] = indices  # with the abs camera indices

        return collated_batch

    # def __iter__(self):
    #     self.all_indices = self.all_indices[torch.randperm(self.all_indices.size(0))]
    #     self.sample_ptr = 0
    #     return self

    # def __next__(self):
    #     if self.sample_ptr >= self.all_indices.size(0):
    #         raise StopIteration
    #     if self.sample_ptr + self.num_rays_per_batch > self.all_indices.size(0) and self.drop_last:
    #         raise StopIteration

    #     sample_ptr_end = min(self.sample_ptr + self.num_rays_per_batch, self.all_indices.size(0))

    #     indices = self.all_indices[self.sample_ptr : sample_ptr_end]

    #     self.sample_ptr += self.num_rays_per_batch

    #     return self.get_batch(indices)

    def _reset(self):
        self.all_indices = self.all_indices[torch.randperm(self.all_indices.size(0))]
        self.sample_ptr = 0

    def _reset_inverse(self):
        self.all_indices_inverted = self.all_indices_inverted[torch.randperm(self.all_indices_inverted.size(0))]
        self.sample_ptr_invert = 0

    def sample(self):
        if self.use_mask:
            if self.sample_ptr + self.num_rays_per_batch > self.all_indices.size(0):
                self._reset()

            indices = self.all_indices[self.sample_ptr : self.sample_ptr + self.num_rays_per_batch]

            self.sample_ptr += self.num_rays_per_batch
        else:
            indices = torch.floor(
                torch.rand((self.num_rays_per_batch, 3), device=self.masks.device)
                * torch.tensor(self.images.shape[:-1], device=self.masks.device)
            ).long()

        ret = self.get_batch(indices)
        return ret

    def sample_inverse(self):
        assert self.use_mask
        if self.sample_ptr_invert + self.num_rays_per_batch > self.all_indices_inverted.size(0):
            self._reset_inverse()

        indices = self.all_indices_inverted[self.sample_ptr_invert : self.sample_ptr_invert + self.num_rays_per_batch]

        self.sample_ptr_invert += self.num_rays_per_batch

        ret = self.get_batch(indices)
        return ret


class PixelTimeUniformSampler_origin:
    def __init__(
        self, dataset: VideoDatasetAllCached, num_rays_per_batch: int, drop_last: bool = False, use_mask: bool = True
    ) -> None:
        CONSOLE.log("Using all uniform sampler")
        self.num_rays_per_batch = num_rays_per_batch
        self.drop_last = drop_last
        self.dataset = dataset

    def get_batch(self, indices):
        c, t, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))

        images = self.dataset.frames[c, t, y, x]
        if images.dtype != torch.float32:
            images = images.to(torch.float32) / 255.0

        batch = {"image": images}
        batch["indices"] = indices[..., [0, 2, 3]]
        batch["time"] = t.to(torch.float32) / self.dataset.num_frames

        return batch

    def sample(self):
        indices = torch.floor(
            torch.rand((self.num_rays_per_batch, 4), device=self.dataset.frames.device)
            * torch.tensor(self.dataset.frames.shape[:-1], device=self.dataset.frames.device)
        ).long()

        return self.get_batch(indices)


class PixelTimeUniformSampler:
    def __init__(
        self, dataset: VideoDatasetAllCached, num_rays_per_batch: int, drop_last: bool = False, use_mask: bool = True
    ) -> None:
        self.num_rays_per_batch = num_rays_per_batch
        self.drop_last = drop_last
        self.dataset = dataset
        CONSOLE.print("Using Uniform Sampler")

    def get_batch(self, indices):
        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        T = self.dataset.frames.shape[1]

        _repeat = T // 30
        t_group = torch.floor(torch.rand(1) * 30).long()
        t = torch.arange(T).reshape(_repeat, -1)[:, t_group[0]]
        c = c.repeat(_repeat)
        y = y.repeat(_repeat)
        x = x.repeat(_repeat)
        t = t.unsqueeze(dim=1).repeat(1, self.num_rays_per_batch // _repeat).flatten().long()

        images = self.dataset.frames[c, t, y, x]
        if images.dtype != torch.float32:
            images = images.to(torch.float32) / 255.0

        batch = {"image": images}
        # batch["indices"] = indices[..., [0, 2, 3]]
        # batch["indices"] = indices[..., [0, 1, 2]]
        batch["indices"] = indices.repeat(_repeat, 1)
        batch["time"] = t.to(torch.float32) / self.dataset.num_frames
        # print(batch["time"].shape)
        # print(batch["image"].shape)
        # print(batch["indices"].shape)
        return batch

    def sample(self):
        C, T, H, W = self.dataset.frames.shape[:-1]
        _repeat = T // 30
        indices = torch.floor(
            torch.rand((self.num_rays_per_batch // _repeat, 3), device=self.dataset.frames.device)
            * torch.tensor((C, H, W), device=self.dataset.frames.device)
        ).long()

        return self.get_batch(indices)

    def set_num_rays_per_batch(self, num_rays_per_batch):
        raise NotImplementedError


class PixelTimeSampler:
    def __init__(
        self,
        dataset: Union[VideoDatasetAllCached, VideoDatasetAllCachedUint8],
        num_rays_per_batch: int,
        static_dynamic_ratio,
        drop_last: bool = False,
        static_dynamic_ratio_end=None,
        total_steps=None,
    ) -> None:
        self.num_rays_per_batch = num_rays_per_batch
        self.step = 0
        self.total_steps = total_steps
        assert self.total_steps is not None
        self.static_dynamic_ratio = static_dynamic_ratio
        self.static_dynamic_ratio_end = static_dynamic_ratio_end
        self.set_static_dynamic_ratio()
        # self.num_static_rays_per_batch = int(num_rays_per_batch * static_dynamic_ratio / (1 + static_dynamic_ratio))
        # self.num_dynamic_rays_per_batch = num_rays_per_batch - self.num_static_rays_per_batch
        self.static_sample_ptr = 0
        self.dynamic_sample_ptr = 0
        self.drop_last = drop_last
        self.dataset = dataset
        self.static_indices = torch.nonzero(~self.dataset.masks[..., 0], as_tuple=False)
        self.dynamic_indices = torch.nonzero(self.dataset.masks[..., 0], as_tuple=False)
        # [n_cams, h, w]
        self._reset_static()
        self._reset_dynamic()

    def set_static_dynamic_ratio(self):
        if self.static_dynamic_ratio_end is None:
            self.num_static_rays_per_batch = int(
                self.num_rays_per_batch * self.static_dynamic_ratio / (1 + self.static_dynamic_ratio)
            )
            self.num_dynamic_rays_per_batch = self.num_rays_per_batch - self.num_static_rays_per_batch
        else:
            current_ratio = self.static_dynamic_ratio + (
                self.static_dynamic_ratio_end - self.static_dynamic_ratio
            ) * np.clip(self.step / self.total_steps, 0, 1)
            self.num_static_rays_per_batch = int(self.num_rays_per_batch * current_ratio / (1 + current_ratio))
            self.num_dynamic_rays_per_batch = self.num_rays_per_batch - self.num_static_rays_per_batch
        self.step += 1

    def _reset_static(self):
        self.static_indices = self.static_indices[torch.randperm(self.static_indices.size(0))]
        self.static_sample_ptr = 0

    def _reset_dynamic(self):
        self.dynamic_indices = self.dynamic_indices[torch.randperm(self.dynamic_indices.size(0))]
        self.dynamic_sample_ptr = 0

    def static_sample_indices(self):
        sampled = self.static_indices[self.static_sample_ptr : self.static_sample_ptr + self.num_static_rays_per_batch]
        self.static_sample_ptr += self.num_rays_per_batch
        return sampled

    def dynamic_sample_indices(self):
        sampled = self.dynamic_indices[
            self.dynamic_sample_ptr : self.dynamic_sample_ptr + self.num_dynamic_rays_per_batch
        ]
        self.dynamic_sample_ptr += self.num_rays_per_batch
        return sampled

    def get_batch(self, indices):
        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        t = torch.floor(torch.rand(*c.shape) * self.dataset.num_frames).to(c)
        images = self.dataset.frames[c, t, y, x]
        if images.dtype != torch.float32:
            images = images.to(torch.float32) / 255.0

        batch = {"image": images}
        # batch["indices"] = indices[..., [0, 2, 3]]
        batch["indices"] = indices
        # batch["time"] = t
        batch["time"] = t.to(torch.float32) / self.dataset.num_frames

        return batch

    def sample(self):
        if self.static_sample_ptr + self.num_rays_per_batch > self.static_indices.size(0):
            self._reset_static()
        if self.dynamic_sample_ptr + self.num_rays_per_batch > self.dynamic_indices.size(0):
            self._reset_dynamic()

        _static_indices = self.static_sample_indices()
        _dynamic_indices = self.dynamic_sample_indices()
        indices = torch.cat([_static_indices, _dynamic_indices], dim=0)
        is_static = torch.cat([torch.ones(_static_indices.shape[0]), torch.zeros(_dynamic_indices.shape[0])])
        self.set_static_dynamic_ratio()

        _perm = torch.randperm(indices.size(0))
        indices = indices[_perm]
        is_static = is_static[_perm]

        batch = self.get_batch(indices)
        batch["is_static"] = is_static.bool()
        return batch

    def set_num_rays_per_batch(self, num_rays_per_batch):
        self.num_rays_per_batch = num_rays_per_batch
        self.num_static_rays_per_batch = int(
            self.num_rays_per_batch
            * self.num_static_rays_per_batch
            / (self.num_static_rays_per_batch + self.num_dynamic_rays_per_batch)
        )
        self.num_dynamic_rays_per_batch = self.num_rays_per_batch - self.num_static_rays_per_batch
        assert self.num_static_rays_per_batch >= 0
        assert self.num_dynamic_rays_per_batch >= 0

    # def set_static_dynamic_ratio(self, ratio):
    #     self.static_dynamic_ratio = ratio
    #     self.num_static_rays_per_batch = int(self.num_rays_per_batch * self.static_dynamic_ratio / (1 + self.static_dynamic_ratio))
    #     self.num_dynamic_rays_per_batch = self.num_rays_per_batch - self.num_static_rays_per_batch


class SpatioTemporalSampler:
    def __init__(
        self,
        dataset: Union[VideoDatasetAllCached, VideoDatasetAllCachedUint8],
        num_rays_per_batch: int,
        static_dynamic_ratio,
        drop_last: bool = False,
        static_dynamic_ratio_end=None,
        total_steps=None,
        n_time_for_dynamic=lambda x: 1,
        use_temporal_weight="none",
    ) -> None:
        self.use_temporal_weight = use_temporal_weight
        self.n_time_for_dynamic = n_time_for_dynamic
        self.num_rays_per_batch = num_rays_per_batch
        self.step = 0
        self.total_steps = total_steps
        assert self.total_steps is not None
        self.static_dynamic_ratio = static_dynamic_ratio
        self.static_dynamic_ratio_end = static_dynamic_ratio_end
        self.set_static_dynamic_ratio()
        # self.num_static_rays_per_batch = int(num_rays_per_batch * static_dynamic_ratio / (1 + static_dynamic_ratio))
        # self.num_dynamic_rays_per_batch = num_rays_per_batch - self.num_static_rays_per_batch
        self.static_sample_ptr = 0
        self.dynamic_sample_ptr = 0
        self.drop_last = drop_last
        self.dataset = dataset
        self.static_indices = torch.nonzero(~self.dataset.masks[..., 0], as_tuple=False)
        self.dynamic_indices = torch.nonzero(self.dataset.masks[..., 0], as_tuple=False)
        self.masks = self.dataset.masks
        self.masked_ratio = torch.count_nonzero(self.masks).item() / self.masks.numel()
        # [n_cams, h, w]
        self._reset_static()
        self._reset_dynamic()

    def set_static_dynamic_ratio(self):
        if self.static_dynamic_ratio_end is None:
            self.num_static_rays_per_batch = int(
                self.num_rays_per_batch * self.static_dynamic_ratio / (1 + self.static_dynamic_ratio)
            )
            self.num_dynamic_rays_per_batch = self.num_rays_per_batch - self.num_static_rays_per_batch
        else:
            current_ratio = self.static_dynamic_ratio + (
                self.static_dynamic_ratio_end - self.static_dynamic_ratio
            ) * np.clip(self.step / self.total_steps, 0, 1)
            self.num_static_rays_per_batch = int(self.num_rays_per_batch * current_ratio / (1 + current_ratio))
            self.num_dynamic_rays_per_batch = self.num_rays_per_batch - self.num_static_rays_per_batch
        self.step += 1

    def _reset_static(self):
        self.static_indices = self.static_indices[torch.randperm(self.static_indices.size(0))]
        self.static_sample_ptr = 0

    def _reset_dynamic(self):
        self.dynamic_indices = self.dynamic_indices[torch.randperm(self.dynamic_indices.size(0))]
        self.dynamic_sample_ptr = 0

    def static_sample_indices(self):
        sampled = self.static_indices[self.static_sample_ptr : self.static_sample_ptr + self.num_static_rays_per_batch]
        self.static_sample_ptr += self.num_rays_per_batch
        return sampled

    def dynamic_sample_indices(self):
        sampled = self.dynamic_indices[
            self.dynamic_sample_ptr : self.dynamic_sample_ptr + self.num_dynamic_rays_per_batch
        ]
        self.dynamic_sample_ptr += self.num_rays_per_batch
        return sampled

    def get_batch(self, indices, is_static):
        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        batch_size = c.shape[0]
        c_static = c[is_static]
        y_static = y[is_static]
        x_static = x[is_static]
        t_static = torch.floor(torch.rand(*c_static.shape) * self.dataset.num_frames).to(c_static)

        c_dynamic = c[~is_static]
        y_dynamic = y[~is_static]
        x_dynamic = x[~is_static]
        c_dynamic = repeat(c_dynamic, "b -> (n b)", n=int(self.n_time_for_dynamic(self.step)))
        y_dynamic = repeat(y_dynamic, "b -> (n b)", n=int(self.n_time_for_dynamic(self.step)))
        x_dynamic = repeat(x_dynamic, "b -> (n b)", n=int(self.n_time_for_dynamic(self.step)))

        if self.use_temporal_weight == "none":
            t_dynamic = torch.floor(torch.rand(*c_dynamic.shape) * self.dataset.num_frames).to(c_dynamic)
        elif self.use_temporal_weight == "median":
            _tweight = self.dataset.frames[c[~is_static], :, y[~is_static], x[~is_static]].float() / 255.0
            _tweight = (_tweight - _tweight.median(dim=1, keepdim=True)[0]).abs().mean(dim=-1)  # should be B x T
            t_dynamic = torch.multinomial(_tweight, int(self.n_time_for_dynamic(self.step)), replacement=False)
            t_dynamic = t_dynamic.transpose(0, 1).flatten()
        elif self.use_temporal_weight == "diff":
            _tweight = self.dataset.frames[c[~is_static], :, y[~is_static], x[~is_static]].float() / 255.0
            diff = _tweight[:, 1:] - _tweight[:, :-1]
            _tweight = torch.cat((_tweight[:, 0:1], diff), dim=1).abs().mean(dim=-1)
            t_dynamic = torch.multinomial(_tweight, int(self.n_time_for_dynamic(self.step)), replacement=False)
            t_dynamic = t_dynamic.transpose(0, 1).flatten()
        else:
            raise NotImplementedError

        c = torch.cat([c_static, c_dynamic], dim=0)
        x = torch.cat([x_static, x_dynamic], dim=0)
        y = torch.cat([y_static, y_dynamic], dim=0)
        t = torch.cat([t_static, t_dynamic], dim=0)

        _perm = torch.randperm(c.size(0))
        c = c[_perm][:batch_size]
        x = x[_perm][:batch_size]
        y = y[_perm][:batch_size]
        t = t[_perm][:batch_size]

        indices = torch.stack([c, y, x], dim=-1)

        images = self.dataset.frames[c, t, y, x]
        if images.dtype != torch.float32:
            images = images.to(torch.float32) / 255.0

        batch = {"image": images}
        # batch["indices"] = indices[..., [0, 2, 3]]
        batch["indices"] = indices
        # batch["time"] = t
        batch["time"] = t.to(torch.float32) / self.dataset.num_frames

        return batch

    def sample(self):
        if self.static_sample_ptr + self.num_rays_per_batch > self.static_indices.size(0):
            self._reset_static()
        if self.dynamic_sample_ptr + self.num_rays_per_batch > self.dynamic_indices.size(0):
            self._reset_dynamic()

        _static_indices = self.static_sample_indices()
        _dynamic_indices = self.dynamic_sample_indices()
        indices = torch.cat([_static_indices, _dynamic_indices], dim=0)
        is_static = torch.cat([torch.ones(_static_indices.shape[0]), torch.zeros(_dynamic_indices.shape[0])])
        self.set_static_dynamic_ratio()

        _perm = torch.randperm(indices.size(0))
        indices = indices[_perm]
        is_static = is_static[_perm].bool()

        batch = self.get_batch(indices, is_static)
        batch["is_static"] = is_static.bool()
        return batch


class ISGSampler:
    """ISG Sampler proposed in DyNeRF, weight each pixel by its departure with time median"""

    def __init__(
        self, dataset, num_rays_per_batch, num_rays_per_pixel, gamma, alpha_fn, use_spatial_uniform, sd_ratio_fn
    ) -> None:
        """
        sd_ratio for static dynamic ratio in spatial sampling
        alpha stands for exponential of time weight, 0 for uniform and 1 for performing like original ISG sampler
        """
        self.num_rays_per_batch = num_rays_per_batch
        self.num_rays_per_pixel = num_rays_per_pixel
        self.num_pixel_per_batch = num_rays_per_batch // num_rays_per_pixel
        self.gamma = gamma
        self.dataset = dataset
        self.medians = dataset.medians  # [n_cams, h, w, 3]
        self.num_cams, self.h, self.w = self.medians.shape[:-1]
        self.use_spatial_uniform = use_spatial_uniform
        self.sd_ratio_fn = sd_ratio_fn
        self.alpha_fn = alpha_fn
        self.step = 0
        self.static_indices = torch.nonzero(~self.dataset.masks[..., 0], as_tuple=False)
        self.dynamic_indices = torch.nonzero(self.dataset.masks[..., 0], as_tuple=False)
        self._update()

    def _update(self):
        self.sd_ratio = self.sd_ratio_fn(self.step)
        assert self.sd_ratio >= 0 and self.sd_ratio <= 1
        self.num_static_rays_per_batch = int(self.num_rays_per_batch * self.sd_ratio / (1 + self.sd_ratio))
        self.num_dynamic_rays_per_batch = self.num_rays_per_batch
        self.alpha = self.alpha_fn(self.step)

    def _reset_static(self):
        self.static_indices = self.static_indices[torch.randperm(self.static_indices.size(0))]
        self.static_sample_ptr = 0

    def _reset_dynamic(self):
        self.dynamic_indices = self.dynamic_indices[torch.randperm(self.dynamic_indices.size(0))]
        self.dynamic_sample_ptr = 0

    def static_sample_indices(self):
        sampled = self.static_indices[self.static_sample_ptr : self.static_sample_ptr + self.num_static_rays_per_batch]
        self.static_sample_ptr += self.num_rays_per_batch
        return sampled

    def dynamic_sample_indices(self):
        sampled = self.dynamic_indices[
            self.dynamic_sample_ptr : self.dynamic_sample_ptr + self.num_dynamic_rays_per_batch
        ]
        self.dynamic_sample_ptr += self.num_rays_per_batch
        return sampled

    def sample(self):
        # if not use_spatial_uniform:
        #     indices = torch.floor(
        #         torch.rand(self.num_pixel_per_batch, 3) * torch.tensor([self.num_cams, self.h, self.w])
        #     ).long()
        # else:
        #     indices =

        _static_indices = self.static_sample_indices()
        _dynamic_indices = self.dynamic_sample_indices()
        indices = torch.cat([_static_indices, _dynamic_indices], dim=0)
        is_static = torch.cat([torch.ones(_static_indices.shape[0]), torch.zeros(_dynamic_indices.shape[0])])

        _perm = torch.randperm(indices.size(0))
        indices = indices[_perm]
        is_static = is_static[_perm]
        is_static = is_static.unsqueeze(-1).repeat(1, self.num_rays_per_pixel).flatten()

        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        del indices
        pixel_seq = self.dataset.frames[c, :, y, x]  # [bs, n_frames, 3]
        if pixel_seq.dtype is not torch.float32:
            pixel_seq = pixel_seq.to(torch.float32) / 255.0
        diff = pixel_seq - self.medians[c, y, x].unsqueeze(1).repeat(1, self.dataset.frames.size(1), 1)  # sanity check
        # print(self.medians.max())

        weights = (
            torch.norm(diff.square() / (diff.square() + self.gamma * self.gamma), p="fro", dim=-1) + 1e-3
        )  # [bs, n_frames]

        weights = F.normalize(weights, p=1, dim=-1)
        # print(weights.sum(dim=-1).min())
        t = torch.multinomial(weights, self.num_rays_per_pixel, False).view(-1)  # [bs, rays_per_pixel]
        del pixel_seq
        del diff
        del weights

        c = c.unsqueeze(-1).repeat(1, self.num_rays_per_pixel).view(-1)
        y = y.unsqueeze(-1).repeat(1, self.num_rays_per_pixel).view(-1)

        x = x.unsqueeze(-1).repeat(1, self.num_rays_per_pixel).view(-1)
        pixels = self.dataset.frames[c, t, y, x]
        if pixels.dtype is not torch.float32:
            pixels = pixels.to(torch.float32) / 255.0

        batch = {
            "image": pixels,
            "indices": torch.stack([c, y, x], dim=-1),
            "time": t.to(torch.float32) / self.dataset.num_frames,
            "is_static": is_static,
        }

        return batch


class ISTSampler:
    """IST Sampler"""


class VarTemporalSampler:
    def __init__(self) -> None:
        pass


class ErrorMapSampler:
    def __init__(
        self,
        dataset,
        num_rays_per_batch,
        sreso,
        treso,
        ema_coeff,
        stratified_on_cams,
        static_init,
        replacement,
        device,
    ) -> None:
        self.dataset = dataset
        self.num_rays_per_batch = num_rays_per_batch
        self.ema_coeff = ema_coeff
        self.sreso = sreso
        self.treso = treso
        n_cams = self.dataset.frames.shape[0]
        self.n_cams = n_cams
        self.num_rays_per_cam = num_rays_per_batch // self.n_cams
        self.error_map = torch.ones([n_cams, treso * sreso * sreso], dtype=torch.float32, device=device)
        self.device = device
        # TODO: add stratified on cams == False
        self.stratified_on_cams = stratified_on_cams

        self.t, self.h, self.w = self.dataset.frames.shape[1:-1]
        pad_h = (sreso - self.h % sreso) % sreso
        pad_w = (sreso - self.w % sreso) % sreso
        dyna_masks = F.pad(self.dataset.masks.squeeze(), (0, pad_w, 0, pad_h, 0, 0), "constant", 0)

        patch_h, patch_w = dyna_masks.shape[1] // sreso, dyna_masks.shape[2] // sreso
        static_masks = rearrange(~dyna_masks, "c (h p1) (w p2) -> c h w (p1 p2)", h=sreso, w=sreso).sum(dim=-1) / (
            patch_h * patch_w
        )

        static_masks = static_masks.unsqueeze(1).repeat(1, treso, 1, 1).view(static_masks.shape[0], -1)
        self.error_map += static_init * static_masks.to(self.error_map)

        self.masks = self.dataset.masks
        self.replacement = replacement

    @torch.no_grad()
    def update(self, batch, error):
        # NOTE: error here means ray level loss
        # cam_indices = torch.arange(self.n_cams).long()
        # inds_coarse = batch["inds_coarse"].reshape(self.n_cams, -1)
        # print(inds_coarse.shape)

        # error_map = self.error_map[cam_indices]
        # error = error.to(self.device)
        # a = self.ema_coeff * error_map.gather(1, inds_coarse)
        # b = (1.0 - self.ema_coeff) * error
        # print(a.shape)
        # print(b.shape)
        # ema_error = self.ema_coeff * error_map.gather(1, inds_coarse) + (1.0 - self.ema_coeff) * error
        # error_map.scatter_(1, inds_coarse, ema_error)
        # self.error_map[cam_indices] = error_map

        cam_indices = batch["indices"][..., 0]
        # print(cam_indices.shape)
        inds_coarse = batch["inds_coarse"]
        # print(inds_coarse.shape)
        # print(error.shape)
        # print(error.max())
        # print(error.min())
        self.error_map[cam_indices, inds_coarse] = (
            self.ema_coeff * self.error_map[cam_indices, inds_coarse] + (1.0 - self.ema_coeff) * error
        )

    def get_batch(self, indices, extras=None):
        # extras means
        if isinstance(indices, torch.Tensor):
            c, t, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        else:
            c, t, y, x = indices
        batch = {
            "image": self.dataset.frames[c, t, y, x],
            # "mask": self.dataset.masks
        }

        if batch["image"].dtype != torch.float32:
            batch["image"] = batch["image"].to(torch.float32) / 255.0

        batch["indices"] = torch.stack([c, y, x], dim=-1)
        batch["time"] = t.to(torch.float32) / self.dataset.num_frames

        if extras is not None:
            batch.update(extras)

        return batch

    def sample(self):
        # from torch-ngp
        inds_coarse = torch.multinomial(self.error_map, self.num_rays_per_cam, replacement=self.replacement)
        inds_t = inds_coarse // (self.sreso * self.sreso)
        inds_xy = inds_coarse % (self.sreso * self.sreso)
        inds_x, inds_y = inds_xy // self.sreso, inds_xy % self.sreso
        st, sx, sy = self.t / self.treso, self.w / self.sreso, self.h / self.sreso
        # TODO: add clamp
        inds_t = (
            ((inds_t * st + torch.rand(self.n_cams, self.num_rays_per_cam, device=self.device) * st))
            .long()
            .clamp(max=self.t - 1)
            .flatten()
        )
        inds_x = (
            (inds_x * sx + sx * torch.rand(self.n_cams, self.num_rays_per_cam, device=self.device))
            .long()
            .clamp(max=self.w - 1)
            .flatten()
        )
        inds_y = (
            (inds_y * sy + sy * torch.rand(self.n_cams, self.num_rays_per_cam, device=self.device))
            .long()
            .clamp(max=self.h - 1)
            .flatten()
        )
        inds_cam = (
            torch.arange(self.n_cams, device=self.device, dtype=torch.long)
            .unsqueeze(-1)
            .repeat(1, self.num_rays_per_cam)
        ).flatten()

        _perm = torch.randperm(inds_cam.size(0))

        indices = torch.stack([inds_cam, inds_t, inds_y, inds_x], dim=-1).reshape(-1, 4).long()[_perm]

        is_static = ~self.masks[inds_cam, inds_y, inds_x].flatten().bool()

        extras = {"inds_coarse": inds_coarse.reshape(-1)[_perm], "is_static": is_static}

        return self.get_batch(indices, extras)


spacetime_samplers = dict(
    uniform=PixelTimeUniformSampler,
    stratified=PixelTimeSampler,
    isg=ISGSampler,
    ist=ISTSampler,
    spatio=SpatioTemporalSampler,
    error_map=ErrorMapSampler,
)

spacetime_samplers_default_args = {
    "error_map": {
        "sreso": 128,
        "treso": 30,
        "ema_coeff": 0.1,
        "stratified_on_cams": True,
        # "dynamic_area_thre": 10000000,
        "static_init": 0,
        "replacement": False,
    },
    "spatio": {},
}
