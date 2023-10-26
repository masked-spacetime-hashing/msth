import time
from collections import defaultdict
from pprint import pprint
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.cameras.cameras import Cameras
import torch


class Timer:
    recorder = defaultdict(list)

    def __init__(self, des="", verbose=True, record=False) -> None:
        self.des = des
        self.verbose = verbose
        self.record = record

    def __enter__(self):
        return self
        self.start = time.time()
        self.start_cuda = torch.cuda.Event(enable_timing=True)
        self.end_cuda = torch.cuda.Event(enable_timing=True)
        self.start_cuda.record()
        return self

    def __exit__(self, *args):
        return
        self.end = time.time()
        self.end_cuda.record()
        self.interval = self.end - self.start
        if self.verbose:
            torch.cuda.synchronize()
            print(f"[cudasync]{self.des} consuming {self.start_cuda.elapsed_time(self.end_cuda)/1000.:.8f}")

            print(f"{self.des} consuming {self.interval:.8f}")
        if self.record:
            Timer.recorder[self.des].append(self.interval)

    @staticmethod
    def show_recorder():
        pprint(Timer.recorder)


def get_ndc_coeffs_from_cameras(cameras: Cameras):
    fx = cameras.fx[0][0].item()
    fy = cameras.fy[0][0].item()
    h = float(cameras.height[0][0].item())
    w = float(cameras.width[0][0].item())

    return (2 * fx / w, 2 * fy / h)


def gen_ndc_rays(ray_bundle: RayBundle, ndc_coeffs, near, normalize_dir=False):
    origins = ray_bundle.origins
    directions = ray_bundle.directions
    # Shift ray origins to near plane, not sure if needed
    t = (near - origins[Ellipsis, 0]) / directions[Ellipsis, 0]
    origins = origins + t[Ellipsis, None] * directions

    dx, dy, dz = directions.unbind(-1)
    ox, oy, oz = origins.unbind(-1)
    # print("ox", ox)

    # Projection
    # o0 = ndc_coeffs[0] * (ox / oz)
    # o1 = ndc_coeffs[1] * (oy / oz)
    # o2 = 1 - 2 * near / oz

    # d0 = ndc_coeffs[0] * (dx / dz - ox / oz)
    # d1 = ndc_coeffs[1] * (dy / dz - oy / oz)
    # d2 = 2 * near / oz

    o0 = 1 - 2 * near / ox
    o1 = -ndc_coeffs[0] * (oy / ox)
    o2 = ndc_coeffs[1] * (oz / ox)

    d0 = 2 * near / ox
    d1 = ndc_coeffs[0] * (-dy / dx + oy / ox)
    d2 = ndc_coeffs[1] * (dz / dx - oz / ox)

    origins = torch.stack([o0, o1, o2], -1).to(ray_bundle.origins)
    directions = torch.stack([d0, d1, d2], -1).to(ray_bundle.origins)
    if normalize_dir:
        directions = torch.nn.functional.normalize(directions, dim=-1)

    if not normalize_dir:
        ray_bundle.nears = torch.zeros(origins.shape[:-1] + (1,), device=ray_bundle.origins.device)
        ray_bundle.fars = torch.ones(origins.shape[:-1] + (1,), device=ray_bundle.origins.device)
    else:
        ray_bundle.nears = torch.zeros(origins.shape[:-1] + (1,), device=ray_bundle.origins.device)
        ray_bundle.fars = None
    ray_bundle.origins = origins
    ray_bundle.directions = directions

    return origins, directions


def gen_ndc_rays_v2(ray_bundle: RayBundle, ndc_coeffs, near, normalize_dir=False):
    origins = ray_bundle.origins
    directions = ray_bundle.directions
    # Shift ray origins to near plane, not sure if needed
    t = (near - origins[Ellipsis, 0]) / directions[Ellipsis, 0]
    origins = origins + t[Ellipsis, None] * directions

    dx, dy, dz = directions.unbind(-1)
    ox, oy, oz = origins.unbind(-1)
    # print("ox", ox)

    # Projection
    # o0 = ndc_coeffs[0] * (ox / oz)
    # o1 = ndc_coeffs[1] * (oy / oz)
    # o2 = 1 - 2 * near / oz

    # d0 = ndc_coeffs[0] * (dx / dz - ox / oz)
    # d1 = ndc_coeffs[1] * (dy / dz - oy / oz)
    # d2 = 2 * near / oz

    o0 = 1 + 2 * near / ox
    o1 = -ndc_coeffs[0] * (oy / ox)
    o2 = ndc_coeffs[1] * (oz / ox)

    d0 = -2 * near / ox
    d1 = ndc_coeffs[0] * (-dy / dx + oy / ox)
    d2 = ndc_coeffs[1] * (dz / dx - oz / ox)

    origins = torch.stack([o0, o1, o2], -1).to(ray_bundle.origins)
    directions = torch.stack([d0, d1, d2], -1).to(ray_bundle.origins)
    if normalize_dir:
        directions = torch.nn.functional.normalize(directions, dim=-1)

    if not normalize_dir:
        ray_bundle.nears = torch.zeros(origins.shape[:-1] + (1,), device=ray_bundle.origins.device)
        ray_bundle.fars = torch.ones(origins.shape[:-1] + (1,), device=ray_bundle.origins.device)
    else:
        ray_bundle.nears = torch.zeros(origins.shape[:-1] + (1,), device=ray_bundle.origins.device)
        ray_bundle.fars = None
    ray_bundle.origins = origins
    ray_bundle.directions = directions

    return origins, directions


def print_value_range(positions):
    print("==================")
    print(positions[..., 0].max())
    print(positions[..., 0].min())
    print(positions[..., 1].max())
    print(positions[..., 1].min())
    print(positions[..., 2].max())
    print(positions[..., 2].min())
    print("==================")


import matplotlib.pyplot as plt


def get_chart(x, y):
    fig, ax = plt.subplots()
    #
    ax.plot(x, y)
    return ax


def sparse_loss(pred, func="logoneplus"):
    # pred should be any shape
    sparse_funcs = {
        "logoneplus": lambda x: (2 * (x**2) + 1.0).log().mean(),
        "l1": lambda x: x.abs().mean(),
    }
    return sparse_funcs[func](pred)


def gen_ndc_rays_looking_at_world_z(ray_bundle: RayBundle, ndc_coeffs, near, normalize_dir):
    origins = ray_bundle.origins
    directions = ray_bundle.directions

    t = (near - origins[Ellipsis, 2]) / directions[Ellipsis, 2]
    # print("tmax:", t.max().item())
    origins = origins + t[Ellipsis, None] * directions

    dx, dy, dz = directions.unbind(-1)
    ox, oy, oz = origins.unbind(-1)

    # Projection
    o0 = ndc_coeffs[0] * (ox / oz)
    o1 = ndc_coeffs[1] * (oy / oz)
    o2 = 1 - 2 * near / oz

    d0 = ndc_coeffs[0] * (dx / dz - ox / oz)
    d1 = ndc_coeffs[1] * (dy / dz - oy / oz)
    d2 = 2 * near / oz

    origins = torch.stack([o0, o1, o2], -1)
    directions = torch.stack([d0, d1, d2], -1)

    if not normalize_dir:
        ray_bundle.nears = torch.zeros(origins.shape[:-1] + (1,), device=ray_bundle.origins.device)
        ray_bundle.fars = torch.ones(origins.shape[:-1] + (1,), device=ray_bundle.origins.device)
    else:
        ray_bundle.nears = torch.zeros(origins.shape[:-1] + (1,), device=ray_bundle.origins.device)
        ray_bundle.fars = None
    ray_bundle.origins = origins
    ray_bundle.directions = directions

    return origins, directions
