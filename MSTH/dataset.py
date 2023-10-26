import gc
import concurrent
import concurrent.futures
import os
from copy import deepcopy
from time import time
from typing import Dict, Union
from tqdm import tqdm, trange

import cv2
import numpy as np
import numpy.typing as npt
import torch
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchtyping import TensorType

from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.utils.misc import get_dict_to_torch
from MSTH.dataparser import VideoDataParserOutputs
from MSTH.utils import Timer
from MSTH.ibrnet.feature_extractor import ResUNet
from pathlib import Path
from rich.console import Console

CONSOLE = Console(width=120)


def get_mask_single_image(mask):
    threshold = 0.05
    mask = gaussian_filter(mask, sigma=5)
    return torch.where(torch.from_numpy(mask) > threshold, 1.0, 0.0)


def extend_mask(mask, radius=5):
    mask = gaussian_filter(mask, sigma=5, radius=[radius, radius])
    mask[np.where(mask > 0.0)] = 1.0
    # print("shit")
    return torch.from_numpy(mask)


class VideoDataset(Dataset):
    def __init__(
        self,
        dataparser_outputs: VideoDataParserOutputs,
        scale_factor: float = 1.0,
        mask_extend_radius: int = 5,
        next_n_frames: int = 1,
    ) -> None:
        super().__init__()
        self._dataparser_outputs = dataparser_outputs
        self.scale_factor = scale_factor

        self.scene_box = deepcopy(dataparser_outputs.scene_box)
        self.metadata = deepcopy(dataparser_outputs.metadata)
        self.cameras = deepcopy(dataparser_outputs.cameras)
        self.cameras.rescale_output_resolution(scaling_factor=scale_factor)
        self.vcs = []
        # TODO: maybe add h and w to dataparseroutputs ?
        self.h = self.cameras.height[0][0].item()
        assert isinstance(self.h, int), "support only all the inputs share same size"
        self.w = self.cameras.width[0][0].item()
        self._prepare_video_captures()
        self.next_n_frames = next_n_frames
        if next_n_frames > 1:
            self.load_first_n_frames()
        else:
            self.load_first_frame()
        self.mask_extend_radius = mask_extend_radius
        # TODO: add support on starting from a specific frame for resuming training

    def _prepare_video_captures(self):
        """loading video captures"""
        print("loading video captures ...")
        for video_filename in self._dataparser_outputs.video_filenames:
            self.vcs.append(cv2.VideoCapture(str(video_filename)))
        self.num_cams = len(self.vcs)
        self.num_frames = self._dataparser_outputs.num_frames
        self.start_frame = self._dataparser_outputs.start_frame

        # TODO: check image format
        self.cur_frame_buffer = np.zeros([self.num_cams, self.h, self.w, 3], dtype=np.float32)
        self.prev_frame_buffer = np.zeros([self.num_cams, self.h, self.w, 3], dtype=np.float32)
        self.next_frame_buffer = np.zeros([self.num_cams, self.h, self.w, 3], dtype=np.float32)

        self.next_frame_loading_results = None
        self.cur_frame = 0

    # depretade
    # def _calc_mask(self):
    #     """assuming cur_frame and last_frame is set correctly"""
    #     # TODO: add how to calc diff here
    #     print(self.cur_frame_buffer.shape)
    #     with Timer("calc norm"):
    #         diff = torch.from_numpy(np.linalg.norm(self.cur_frame_buffer - self.prev_frame_buffer, ord=2, axis=-1))
    #     # TODO: add threshold here
    #     mask_threshold = torch.mean(diff)
    #     self.mask = torch.where(diff > mask_threshold, 1, 0)

    def __len__(self):
        return len(self._dataparser_outputs.video_filenames)

    def load_first_frame(self):
        print("loading first image ...")
        for idx in range(self.num_cams):
            success, self.cur_frame_buffer[idx] = self.vcs[idx].read()
            assert success
            # self.cur_frame_buffer[idx] = cv2.cvtColor(self.cur_frame_buffer[idx], cv2.COLOR_BGR2RGB)
            self.cur_frame_buffer[idx] = self.cur_frame_buffer[idx][..., [2, 1, 0]]
        self.cur_frame = 1
        self.cur_frame_buffer /= 255.0
        self.mask = torch.ones([self.num_cams, self.h, self.w, 1])
        self.next_mask = torch.zeros([self.num_cams, self.h, self.w, 1])
        self.load_next_frame()
        assert self.next_frame_loading_results is not None

    def tick(self):
        if self.next_frame_loading_results is not None:
            concurrent.futures.wait(self.next_frame_loading_results)
        self.set_next_frame()
        self.load_next_frame()
        self.cur_frame += 1

    def set_next_frame(self):
        tmp = self.prev_frame_buffer
        self.prev_frame_buffer = self.cur_frame_buffer
        self.cur_frame_buffer = self.next_frame_buffer
        self.next_frame_buffer = tmp
        self.mask, self.next_mask = self.next_mask, self.mask

    def load_next_frame_worker(self, idx: int):
        """worker function for setting next frame with idx-th camera"""
        success, self.next_frame_buffer[idx] = self.vcs[idx].read()
        assert success
        # self.next_frame_buffer[idx] = cv2.cvtColor(self.next_frame_buffer[idx], cv2.COLOR_BGR2RGB)
        self.next_frame_buffer[idx] = self.next_frame_buffer[idx][..., [2, 1, 0]]
        self.next_frame_buffer[idx] /= 255.0
        new_mask = np.linalg.norm(self.next_frame_buffer[idx] - self.cur_frame_buffer[idx], ord=np.inf, axis=-1)
        ## determine how to set threshold
        # mask_threshold = np.mean(new_mask) * 5.0
        # self.next_mask[idx] = torch.where(torch.from_numpy(new_mask) > mask_threshold, 1.0, 0.0)
        new_mask = get_mask_single_image(new_mask)
        # print("here")
        # try:
        self.next_mask[idx] = extend_mask(new_mask, self.mask_extend_radius).unsqueeze(-1)

    def load_next_frame(self):
        self.next_frame_loading_results = []
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        for i in range(self.num_cams):
            self.next_frame_loading_results.append(executor.submit(self.load_next_frame_worker, i))

        # assert len(self.next_frame_loading_results) == self.num_cams

    def get_numpy_image(self, image_idx: int) -> npt.NDArray[np.uint8]:
        return self.cur_frame_buffer[image_idx]

    def get_image(self, image_idx: int) -> TensorType["image_height", "image_width", "num_channels"]:
        image = torch.from_numpy(self.get_numpy_image(image_idx))
        assert image.size(-1) == 3

        return image

    def get_mask(self, image_idx: int) -> TensorType["image_height", "image_width"]:
        return self.mask[image_idx]

    def load_next_n_frames_worker(self, next_n_frames, idx: int):
        for _ in range(next_n_frames - 1):
            success, _ = self.vcs[idx].read()
            assert success, "load next frame failed !"
        success, self.next_frame_buffer[idx] = self.vcs[idx].read()
        assert success
        # self.next_frame_buffer[idx] = cv2.cvtColor(self.next_frame_buffer[idx], cv2.COLOR_BGR2RGB)
        self.next_frame_buffer[idx] = self.next_frame_buffer[idx][..., [2, 1, 0]]
        self.next_frame_buffer[idx] /= 255.0
        new_mask = np.linalg.norm(self.next_frame_buffer[idx] - self.cur_frame_buffer[idx], ord=np.inf, axis=-1)
        ## determine how to set threshold

        new_mask = get_mask_single_image(new_mask)

        self.next_mask[idx] = extend_mask(new_mask, self.mask_extend_radius).unsqueeze(-1)

    def load_next_n_frames(self, next_n_frame: int):
        self.next_frame_loading_results = []
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() - 1)
        for i in range(self.num_cams):
            self.next_frame_loading_results.append(executor.submit(self.load_next_n_frames_worker, next_n_frame, i))

    def load_first_n_frames(self):
        print("loading first image ...")
        for idx in range(self.num_cams):
            success, self.cur_frame_buffer[idx] = self.vcs[idx].read()
            assert success
            # self.cur_frame_buffer[idx] = cv2.cvtColor(self.cur_frame_buffer[idx], cv2.COLOR_BGR2RGB)
            self.cur_frame_buffer[idx] = self.cur_frame_buffer[idx][..., [2, 1, 0]]
        self.cur_frame = 1
        self.cur_frame_buffer /= 255.0
        self.mask = torch.ones([self.num_cams, self.h, self.w, 1])
        self.next_mask = torch.zeros([self.num_cams, self.h, self.w, 1])
        self.load_next_n_frames(self.next_n_frames)
        assert self.next_frame_loading_results is not None

    def tick_n_frames(self):
        if self.next_frame_loading_results is not None:
            concurrent.futures.wait(self.next_frame_loading_results)
        self.set_next_frame()
        self.load_next_n_frames(self.next_n_frames)
        self.cur_frame += 1

    def __del__(self):
        for vc in self.vcs:
            vc.release()

    def get_all_data(self, device: Union[torch.device, str] = "cpu") -> Dict[str, TensorType]:
        return get_dict_to_torch(
            {"image": torch.from_numpy(self.cur_frame_buffer), "mask": self.mask}, device, exclude=["image"]
        )

    def __getitem__(self, index) -> Dict:
        return {"image": self.get_image(index), "mask": self.get_mask(index)}


class VideoDataLoader(DataLoader):
    def __init__(
        self, dataset: VideoDataset, device: Union[torch.device, str] = "cpu", collate_fn=nerfstudio_collate, **kwargs
    ):
        self.dataset = dataset
        super().__init__(dataset=dataset, **kwargs)
        self.collate_fn = collate_fn
        self.device = device
        self.num_workers = kwargs.get("num_works", 0)


# class EvalVideoDataset(Dataset):
#     def __init__(self, dataparser_outputs: VideoDataParserOutputs, scale_factor: float = 1.0) -> None:
#         super().__init__()
#         self._dataparser_outputs = dataparser_outputs
#         self.scale_factor = scale_factor

#         self.scene_box = deepcopy(dataparser_outputs.scene_box)
#         self.metadata = deepcopy(dataparser_outputs.metadata)
#         self.cameras = deepcopy(dataparser_outputs.cameras)
#         self.cameras.rescale_output_resolution(scaling_factor=scale_factor)
#         self.vcs = []
#         # TODO: maybe add h and w to dataparseroutputs ?
#         self.h = self.cameras.height[0][0].item()
#         assert isinstance(self.h, int), "support only all the inputs share same size"
#         self.w = self.cameras.width[0][0].item()
#         self._prepare_video_captures()
#         self.load_first_frame()

#     def _prepare_video_captures(self):
#         """loading video captures"""
#         print("loading video captures ...")
#         for video_filename in self._dataparser_outputs.video_filenames:
#             self.vcs.append(cv2.VideoCapture(str(video_filename)))
#         self.num_cams = len(self.vcs)
#         self.num_frames = self._dataparser_outputs.num_frames
#         self.start_frame = self._dataparser_outputs.start_frame

#         # TODO: check image format
#         self.cur_frame_buffer = np.zeros([self.num_cams, self.h, self.w, 3], dtype=np.float32)
#         self.prev_frame_buffer = np.zeros([self.num_cams, self.h, self.w, 3], dtype=np.float32)
#         self.next_frame_buffer = np.zeros([self.num_cams, self.h, self.w, 3], dtype=np.float32)

#         self.next_frame_loading_results = None
#         self.cur_frame = 0

#     def __len__(self):
#         return len(self._dataparser_outputs.video_filenames)

#     def load_first_frame(self):
#         print("loading first image ...")
#         for idx in range(self.num_cams):
#             success, self.cur_frame_buffer[idx] = self.vcs[idx].read()
#             assert success
#         self.cur_frame = 1
#         self.cur_frame_buffer /= 255.0
#         self.load_next_frame()
#         assert self.next_frame_loading_results is not None

#     def tick(self):
#         if self.next_frame_loading_results is not None:
#             concurrent.futures.wait(self.next_frame_loading_results)
#         self.set_next_frame()
#         self.load_next_frame()

#     def set_next_frame(self):
#         tmp = self.prev_frame_buffer
#         self.prev_frame_buffer = self.cur_frame_buffer
#         self.cur_frame_buffer = self.next_frame_buffer
#         self.next_frame_buffer = tmp

#     def load_next_frame_worker(self, idx: int):
#         """worker function for setting next frame with idx-th camera"""
#         success, self.next_frame_buffer[idx] = self.vcs[idx].read()
#         assert success
#         self.next_frame_buffer[idx] /= 255.0

#     def load_next_frame(self):
#         self.next_frame_loading_results = []
#         executor = concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() - 1)
#         for i in range(self.num_cams):
#             self.next_frame_loading_results.append(executor.submit(self.load_next_frame_worker, i))

#     def __del__(self):
#         for vc in self.vcs:
#             vc.release()

#     def get_all_data(self, device: Union[torch.device, str] = "cpu") -> Dict[str, TensorType]:
#         return get_dict_to_torch(
#             {
#                 "image": torch.from_numpy(self.cur_frame_buffer),
#             },
#             device,
#             exclude=["image"],
#         )

#     def load_next_n_frame_worker(self, next_n_frames, idx: int):
#         for _ in range(next_n_frames - 1):
#             success, _ = self.vcs[idx].read()
#             assert success, "load next frame failed !"
#         success, self.next_frame_buffer[idx] = self.vcs[idx].read()
#         assert success
#         # self.next_frame_buffer[idx] = cv2.cvtColor(self.next_frame_buffer[idx], cv2.COLOR_BGR2RGB)
#         self.next_frame_buffer[idx] = self.next_frame_buffer[idx][..., [2, 1, 0]]
#         self.next_frame_buffer[idx] /= 255.0
#         new_mask = np.linalg.norm(self.next_frame_buffer[idx] - self.cur_frame_buffer[idx], ord=np.inf, axis=-1)
#         ## determine how to set threshold

#         new_mask = get_mask_single_image(new_mask)

#         self.next_mask[idx] = extend_mask(new_mask, self.mask_extend_radius).unsqueeze(-1)

#     def load_next_n_frame(self, next_n_frame: int):
#         self.next_frame_loading_results = []
#         executor = concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() - 1)
#         for i in range(self.num_cams):
#             self.next_frame_loading_results.append(executor.submit(self.load_next_n_frame_worker, next_n_frame, i))

#     def tick_n_frames(self, next_n_frames: int = None):
#         if next_n_frames is None:
#             next_n_frame = self.next_n_frames
#         if self.next_frame_loading_results is not None:
#             concurrent.futures.wait(self.next_frame_loading_results)
#         self.set_next_frame()
#         self.load_next_n_frame(next_n_frames)
#         self.cur_frame += 1


class VideoDatasetWithFeature(Dataset):
    def __init__(
        self,
        dataparser_outputs: VideoDataParserOutputs,
        scale_factor: float = 1.0,
        mask_extend_radius: int = 5,
        next_n_frames: int = 1,
        pretrained_path: Union[str, Path] = "",
        fe_device: Union[str, torch.device] = "cuda",
    ) -> None:
        super().__init__()
        self._dataparser_outputs = dataparser_outputs
        self.scale_factor = scale_factor

        self.scene_box = deepcopy(dataparser_outputs.scene_box)
        self.metadata = deepcopy(dataparser_outputs.metadata)
        self.cameras = deepcopy(dataparser_outputs.cameras)
        self.cameras.rescale_output_resolution(scaling_factor=scale_factor)
        self.fe_device = fe_device
        self.feature_extractor = ResUNet.load_from_pretrained(pretrained_path)
        self.feature_extractor.to(self.fe_device)
        self.vcs = []
        # TODO: maybe add h and w to dataparseroutputs ?
        self.h = self.cameras.height[0][0].item()
        assert isinstance(self.h, int), "support only all the inputs share same size"
        self.w = self.cameras.width[0][0].item()
        self._prepare_video_captures()
        self.next_n_frames = next_n_frames
        self.load_first_frame()
        # self._build_feats()
        self.mask_extend_radius = mask_extend_radius
        # TODO: add support on starting from a specific frame for resuming training

    def _prepare_video_captures(self):
        """loading video captures"""
        print("loading video captures ...")
        for video_filename in self._dataparser_outputs.video_filenames:
            self.vcs.append(cv2.VideoCapture(str(video_filename)))
        self.num_cams = len(self.vcs)
        self.num_frames = self._dataparser_outputs.num_frames
        self.start_frame = self._dataparser_outputs.start_frame

        # TODO: check image format
        self.cur_frame_buffer = np.zeros([self.num_cams, self.h, self.w, 3], dtype=np.float32)
        self.prev_frame_buffer = np.zeros([self.num_cams, self.h, self.w, 3], dtype=np.float32)
        self.next_frame_buffer = np.zeros([self.num_cams, self.h, self.w, 3], dtype=np.float32)

        self.next_frame_loading_results = None
        self.cur_frame = 0

    def __len__(self):
        return len(self._dataparser_outputs.video_filenames)

    def load_first_frame(self):
        print("loading first image ...")
        for idx in range(self.num_cams):
            success, self.cur_frame_buffer[idx] = self.vcs[idx].read()
            assert success
            # self.cur_frame_buffer[idx] = cv2.cvtColor(self.cur_frame_buffer[idx], cv2.COLOR_BGR2RGB)
            self.cur_frame_buffer[idx] = self.cur_frame_buffer[idx][..., [2, 1, 0]]
        self.cur_frame = 1
        self.cur_frame_buffer /= 255.0
        self.mask = torch.ones([self.num_cams, self.h, self.w, 1])
        self.next_mask = torch.zeros([self.num_cams, self.h, self.w, 1])
        self.load_next_frame()
        self.extract_cur_frame_feature()
        assert self.next_frame_loading_results is not None

    def tick(self):
        if self.next_frame_loading_results is not None:
            concurrent.futures.wait(self.next_frame_loading_results)
        self.set_next_frame()
        self.load_next_frame()
        self.cur_frame += 1

    def set_next_frame(self):
        tmp = self.prev_frame_buffer
        self.prev_frame_buffer = self.cur_frame_buffer
        self.cur_frame_buffer = self.next_frame_buffer
        self.next_frame_buffer = tmp
        self.mask, self.next_mask = self.next_mask, self.mask

    def load_next_frame_worker(self, idx: int):
        """worker function for setting next frame with idx-th camera"""
        success, self.next_frame_buffer[idx] = self.vcs[idx].read()
        assert success
        # self.next_frame_buffer[idx] = cv2.cvtColor(self.next_frame_buffer[idx], cv2.COLOR_BGR2RGB)
        self.next_frame_buffer[idx] = self.next_frame_buffer[idx][..., [2, 1, 0]]
        self.next_frame_buffer[idx] /= 255.0
        new_mask = np.linalg.norm(self.next_frame_buffer[idx] - self.cur_frame_buffer[idx], ord=np.inf, axis=-1)
        ## determine how to set threshold
        # mask_threshold = np.mean(new_mask) * 5.0
        # self.next_mask[idx] = torch.where(torch.from_numpy(new_mask) > mask_threshold, 1.0, 0.0)
        new_mask = get_mask_single_image(new_mask)
        # print("here")
        # try:
        self.next_mask[idx] = extend_mask(new_mask, self.mask_extend_radius).unsqueeze(-1)

        # print("hello")

    def load_next_frame(self):
        self.next_frame_loading_results = []
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        for i in range(self.num_cams):
            self.next_frame_loading_results.append(executor.submit(self.load_next_frame_worker, i))

        # assert len(self.next_frame_loading_results) == self.num_cams

    def get_numpy_image(self, image_idx: int) -> npt.NDArray[np.uint8]:
        return self.cur_frame_buffer[image_idx]

    def get_image(self, image_idx: int) -> TensorType["image_height", "image_width", "num_channels"]:
        image = torch.from_numpy(self.get_numpy_image(image_idx))
        assert image.size(-1) == 3

        return image

    def get_mask(self, image_idx: int) -> TensorType["image_height", "image_width"]:
        return self.mask[image_idx]

    def load_next_n_frames_worker(self, next_n_frames, idx: int):
        for _ in range(next_n_frames - 1):
            success, _ = self.vcs[idx].read()
            assert success, "load next frame failed !"
        success, self.next_frame_buffer[idx] = self.vcs[idx].read()
        assert success
        # self.next_frame_buffer[idx] = cv2.cvtColor(self.next_frame_buffer[idx], cv2.COLOR_BGR2RGB)
        self.next_frame_buffer[idx] = self.next_frame_buffer[idx][..., [2, 1, 0]]
        self.next_frame_buffer[idx] /= 255.0
        new_mask = np.linalg.norm(self.next_frame_buffer[idx] - self.cur_frame_buffer[idx], ord=np.inf, axis=-1)
        ## determine how to set threshold

        new_mask = get_mask_single_image(new_mask)

        self.next_mask[idx] = extend_mask(new_mask, self.mask_extend_radius).unsqueeze(-1)

    def extract_cur_frame_feature(self):
        self.device_cur_frame = torch.from_numpy(self.cur_frame_buffer).to(self.fe_device)

        num_images = self.cur_frame_buffer.shape[0]
        self.cur_frame_feats_buffer = torch.zeros(self.device_cur_frame.shape[:-1] + (32,))
        mini_bs = 2
        for start in range(0, num_images, mini_bs):
            end = min(num_images, start + mini_bs)
            self.cur_frame_feats_buffer[start:end] = self.feature_extractor(self.device_cur_frame[start:end])[1].to(
                "cpu"
            )

        # set to zero array for testing
        # self.cur_frame_feats_buffer = torch.zeros(self.cur_frame_buffer.shape[:-1] + (32,))

        print("feature shape: ", self.cur_frame_feats_buffer.shape)

    def _build_feats(self):
        # workaround for test
        self.cur_frame_feats_buffer = torch.zeros(self.cur_frame_buffer.shape[:-1] + (32,))

    def get_all_data(self, device: Union[torch.device, str] = "cpu") -> Dict[str, TensorType]:
        return get_dict_to_torch(
            {"image": torch.from_numpy(self.cur_frame_buffer), "mask": self.mask}, device, exclude=["image"]
        )

    def __getitem__(self, index) -> Dict:
        return {"image": self.get_image(index), "mask": self.get_mask(index)}


def get_varianece_mask(frames, threshold=0.03, device="cpu"):
    n_cams, _, h, w, _ = frames.shape
    masks = torch.zeros(n_cams, h, w, 1, dtype=torch.bool)
    # [n_cams, h, w]
    n_cams = frames.size(0)
    for i in trange(n_cams):
        if frames.dtype != torch.float32:
            tmp = frames[i].to(torch.float32) / 255.0
        else:
            tmp = frames[i]
        dynam = torch.var(tmp.to(device), dim=0)
        dynam = torch.mean(dynam, dim=-1, keepdim=True)  # [h, w, 1]
        masks[i] = (dynam > threshold).to("cpu")
    CONSOLE.print("number of dynamic pixels: {}/{}".format(torch.count_nonzero(masks).item(), masks.numel()))
    # torch.save(masks, "/data/machine/data/flame_salmon_videos_2/masks.pt")
    # torch.save(masks, "/data/machine/data/flame_salmon_videos_2/masks.pt")

    return masks


def get_median(frames, device="cpu"):
    n_cams, _, h, w, _ = frames.shape
    medians = torch.zeros(n_cams, h, w, 3, dtype=torch.float32)
    # [n_cams, h, w]
    n_cams = frames.size(0)
    for i in trange(n_cams):
        if frames.dtype != torch.float32:
            tmp = frames[i].to(torch.float32) / 255.0
        else:
            tmp = frames[i]
        dynam = torch.median(tmp.to(device), dim=0).values
        medians[i] = dynam.to(medians)

    return medians


class VideoDatasetAllCached(Dataset):
    def __init__(
        self,
        dataparser_outputs: VideoDataParserOutputs,
        scale_factor: float = 1.0,
        mask_extend_radius: int = 5,
        use_mask: bool = False,
    ) -> None:
        super().__init__()
        self._dataparser_outputs = dataparser_outputs
        self.scale_factor = scale_factor
        self.mask_extend_radius = mask_extend_radius

        self.scene_box = deepcopy(dataparser_outputs.scene_box)
        self.metadata = deepcopy(dataparser_outputs.metadata)
        self.cameras = deepcopy(dataparser_outputs.cameras)
        self.cameras.rescale_output_resolution(scaling_factor=scale_factor)
        self.vcs = []
        # TODO: maybe add h and w to dataparseroutputs ?
        self.h = self.cameras.height[0][0].item()
        assert isinstance(self.h, int), "support only all the inputs share same size"
        self.w = self.cameras.width[0][0].item()
        self.num_cams = len(self._dataparser_outputs.video_filenames)
        self.num_frames = self._dataparser_outputs.num_frames
        self.use_mask = use_mask
        self.cache_all_frames()

    def cache_all_frames(self):
        self.frames = torch.zeros([self.num_cams, self._dataparser_outputs.num_frames, self.h, self.w, 3])

        if self.use_mask:
            self.masks = torch.zeros(
                [self.num_cams, self._dataparser_outputs.num_frames, self.h, self.w, 1], dtype=torch.bool
            )

        for i in trange(self.num_cams):
            vc = cv2.VideoCapture(str(self._dataparser_outputs.video_filenames[i]))
            for j in range(self.num_frames):
                suc, frame = vc.read()
                assert suc
                self.frames[i, j] = torch.from_numpy(frame[..., [2, 1, 0]]) / 255.0
                if self.use_mask:
                    if j == 0:
                        self.masks[i, j].fill_(False)
                    else:
                        self.masks[i, j] = torch.norm(self.frames[i, j] - self.frames[i, j - 1], p=np.inf, dim=-1)[
                            ..., None
                        ]
                        self.masks[i, j] = get_mask_single_image(self.masks[i, j]).to(torch.bool)
                        self.masks[i, j] = extend_mask(self.masks[i, j, ..., 0], self.mask_extend_radius)[..., None].to(
                            torch.bool
                        )

    def __len__(self):
        return len(self._dataparser_outputs.video_filenames)

    # def __getitem__(self, index):
    #     return self.frames[index]


class VideoDatasetAllCachedUint8(Dataset):
    def __init__(
        self,
        dataparser_outputs: VideoDataParserOutputs,
        scale_factor: float = 1.0,
        mask_extend_radius: int = 5,
        use_mask: bool = True,
        use_median=False,
    ) -> None:
        super().__init__()
        self._dataparser_outputs = dataparser_outputs
        self.scale_factor = scale_factor
        self.use_mask = use_mask
        self.use_median = use_median

        self.scene_box = deepcopy(dataparser_outputs.scene_box)
        self.metadata = deepcopy(dataparser_outputs.metadata)
        self.cameras = deepcopy(dataparser_outputs.cameras)
        self.cameras.rescale_output_resolution(scaling_factor=scale_factor)
        # self.fe_device = fe_device
        # self.feature_extractor = ResUNet.load_from_pretrained(pretrained_path)
        # self.feature_extractor.to(self.fe_device)
        # self.vcs = []
        # TODO: maybe add h and w to dataparseroutputs ?
        self.h = self.cameras.height[0][0].item()
        assert isinstance(self.h, int), "support only all the inputs share same size"
        self.w = self.cameras.width[0][0].item()
        self.mask_extend_radius = mask_extend_radius
        self.num_cams = len(self._dataparser_outputs.video_filenames)
        self.num_frames = self._dataparser_outputs.num_frames
        self.cache_all_frames()

    def cache_all_frames(self):
        self.frames = torch.zeros(
            [self.num_cams, self._dataparser_outputs.num_frames, self.h, self.w, 3], dtype=torch.uint8
        )

        self.masks = torch.zeros(
            [self.num_cams, self._dataparser_outputs.num_frames, self.h, self.w, 1], dtype=torch.bool
        )

        for i in trange(self.num_cams):
            vc = cv2.VideoCapture(str(self._dataparser_outputs.video_filenames[i]))
            for j in range(self.num_frames):
                suc, frame = vc.read()
                assert suc
                self.frames[i, j] = torch.from_numpy(frame[..., [2, 1, 0]])

        if self.use_mask:
            masks_path = self._dataparser_outputs.data_dir / "masks.pt"
            if masks_path.exists():
                self.masks = torch.load(masks_path, map_location="cpu")
                CONSOLE.log("load precomputed mask from {}".format(str(masks_path)))
            else:
                self.masks = get_varianece_mask(self.frames, device="cuda")
                torch.save(self.masks, masks_path)
                CONSOLE.log("Masks calced and saved since no cached masks found")

            num_mask_pixels = torch.count_nonzero(self.masks).item()
            CONSOLE.log(
                f"masked: {num_mask_pixels}/{self.masks.numel()}, rate: {num_mask_pixels/self.masks.numel():.2f}"
            )

        if self.use_median:
            medians_path = self._dataparser_outputs.data_dir / "medians.pt"
            if medians_path.exists():
                self.medians = torch.load(medians_path, map_location="cpu")
                CONSOLE.log("load precomputed median from {}".format(str(medians_path)))
            else:
                self.medians = get_median(self.frames, device="cuda")
                torch.save(self.medians, medians_path)
                CONSOLE.log("Medians calced and saved since no cached medians found")

    def __getitem__(self, index):
        images = self.frames[index, 0].to(torch.float32) / 255.0
        return {"image": images}

    def __len__(self):
        return len(self._dataparser_outputs.video_filenames)
