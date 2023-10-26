from abc import abstractmethod
from typing import Callable, List, Optional, Tuple

import nerfacc
import torch

# from nerfacc import OccupancyGrid
from torch import nn
from torchtyping import TensorType
from MSTH.utils import Timer

from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples
from nerfstudio.model_components.ray_samplers import (
    UniformSampler,
    PDFSampler,
    UniformLinDispPiecewiseSampler,
    ProposalNetworkSampler,
    Sampler,
    SpacedSampler,
)

if "line_profiler" not in dir() and "profile" not in dir():

    def profile(func):
        return func


def spacetime(ray_samples: RaySamples):
    positions = ray_samples.frustums.get_positions()
    assert ray_samples.times is not None, "ray samples should contain time information"
    times = ray_samples.times.unsqueeze(1).repeat(1, positions.size(1), 1).to(positions)
    # [num_rays, num_samples, 4]
    return torch.cat([positions, times], dim=-1)


# def spacetime_2(positions, times):
def spacetime_concat(positions, times):
    # print(positions.shape)
    # print(times.shape)
    if times.dim() < positions.dim():
        times = times.unsqueeze(1).repeat(1, positions.size(1), 1)
    return torch.cat([positions, times.to(positions)], dim=-1)


class UniformSamplerSpatial(UniformSampler):
    def generate_ray_samples(
        self, ray_bundle: Optional[RayBundle] = None, num_samples: Optional[int] = None
    ) -> RaySamples:
        ray_samples = super().generate_ray_samples(ray_bundle, num_samples)
        assert ray_bundle.times is not None, "ray_bundle should contain time information"
        ray_samples.times = ray_bundle.times

        return ray_samples


# class UniformLinDispPiecewiseSamplerSpatial(UniformLinDispPiecewiseSampler):
#     def generate_ray_samples(
#         self, ray_bundle: Optional[RayBundle] = None, num_samples: Optional[int] = None
#     ) -> RaySamples:
#         ray_samples = super(UniformLinDispPiecewiseSampler, self).generate_ray_samples(ray_bundle, num_samples)
#         assert ray_bundle.times is not None, "ray_bundle should contain time information"
#         ray_samples.times = ray_bundle.times

#         return ray_samples


class UniformLinDispPiecewiseSamplerSpatial(SpacedSampler):
    """Piecewise sampler along a ray that allocates the first half of the samples uniformly and the second half
    using linearly in disparity spacing.


    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defaults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        train_stratified=True,
        single_jitter=False,
        middle_distance=1,
    ) -> None:
        k = middle_distance
        super().__init__(
            num_samples=num_samples,
            spacing_fn=lambda x: torch.where(x < k, x / (2 * k), 1 - k / (2 * x)),
            spacing_fn_inv=lambda x: torch.where(x < 0.5, 2 * k * x, k / (2 - 2 * x)),
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )

    def generate_ray_samples(
        self, ray_bundle: Optional[RayBundle] = None, num_samples: Optional[int] = None
    ) -> RaySamples:
        ray_samples = super().generate_ray_samples(ray_bundle, num_samples)
        assert ray_bundle.times is not None, "ray_bundle should contain time information"
        ray_samples.times = ray_bundle.times

        return ray_samples


class UniformLinDispSamplerSpatial(SpacedSampler):
    """Piecewise sampler along a ray that allocates the first half of the samples uniformly and the second half
    using linearly in disparity spacing.


    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defaults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        train_stratified=True,
        single_jitter=False,
    ) -> None:
        super().__init__(
            num_samples=num_samples,
            spacing_fn=lambda x: 1 - 1.0 / (x + 1),
            spacing_fn_inv=lambda x: 1 / (1 - x) - 1,
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )

    def generate_ray_samples(
        self, ray_bundle: Optional[RayBundle] = None, num_samples: Optional[int] = None
    ) -> RaySamples:
        ray_samples = super().generate_ray_samples(ray_bundle, num_samples)
        assert ray_bundle.times is not None, "ray_bundle should contain time information"
        ray_samples.times = ray_bundle.times

        return ray_samples


class PDFSamplerSpatial(PDFSampler):
    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        ray_samples: Optional[RaySamples] = None,
        weights: TensorType[..., "num_samples", 1] = None,
        num_samples: Optional[int] = None,
        eps: float = 0.00001,
    ) -> RaySamples:
        ret = super().generate_ray_samples(ray_bundle, ray_samples, weights, num_samples, eps)
        assert ray_bundle.times is not None, "ray_bundle should contain time information"
        ret.times = ray_bundle.times
        # print(ray_samples.shape)

        return ret


class ProposalNetworkSamplerSpatial(ProposalNetworkSampler):
    def __init__(
        self,
        num_proposal_samples_per_ray: Tuple[int] = ...,
        num_nerf_samples_per_ray: int = 32,
        num_proposal_network_iterations: int = 2,
        single_jitter: bool = True,
        update_sched: Callable = ...,
        initial_sampler: Optional[Sampler] = None,
        middle_distance: int = 1,
    ) -> None:
        super(Sampler, self).__init__()
        self.num_proposal_samples_per_ray = num_proposal_samples_per_ray
        self.num_nerf_samples_per_ray = num_nerf_samples_per_ray
        self.num_proposal_network_iterations = num_proposal_network_iterations
        self.update_sched = update_sched
        if self.num_proposal_network_iterations < 1:
            raise ValueError("num_proposal_network_iterations must be >= 1")

        # samplers
        if initial_sampler is None:
            self.initial_sampler = UniformLinDispPiecewiseSamplerSpatial(
                single_jitter=single_jitter, middle_distance=middle_distance
            )
        else:
            self.initial_sampler = initial_sampler
        self.pdf_sampler = PDFSamplerSpatial(include_original=False, single_jitter=single_jitter)

        self._anneal = 1.0
        self._steps_since_update = 0
        self._step = 0

    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        density_fns: Optional[List[Callable]] = None,
    ) -> Tuple[RaySamples, List, List, List]:
        with Timer("total_ps"):
            with Timer("prep"):
                assert ray_bundle is not None
                assert density_fns is not None

                weights_list = []
                # weights_list_static = []
                ray_samples_list = []

                n = self.num_proposal_network_iterations
                weights = None
                weights_static = None
                ray_samples = None
                updated = self._steps_since_update > self.update_sched(self._step) or self._step < 10
            # print(n)
            for i_level in range(n + 1):
                with Timer("level"):
                    is_prop = i_level < n
                    num_samples = (
                        self.num_proposal_samples_per_ray[i_level] if is_prop else self.num_nerf_samples_per_ray
                    )
                # print("num_samples", num_samples)
                if i_level == 0:
                    # Uniform sampling because we need to start with some samples
                    with Timer("initial"):
                        ray_samples = self.initial_sampler(ray_bundle, num_samples=num_samples)
                else:
                    # PDF sampling based on the last samples and their weights
                    # Perform annealing to the weights. This will be a no-op if self._anneal is 1.0.
                    assert weights is not None
                    with Timer("anneal"):
                        annealed_weights = torch.pow(weights, self._anneal)
                        self.last_weight = annealed_weights
                    with Timer("pdf"):
                        ray_samples = self.pdf_sampler(
                            ray_bundle, ray_samples, annealed_weights, num_samples=num_samples
                        )
                    # print("ray_samples.shape", ray_samples.shape)
                if is_prop:
                    if updated:
                        # always update on the first step or the inf check in grad scaling crashes
                        # density = density_fns[i_level](ray_samples.frustums.get_positions())
                        with Timer("updated density_fn query"):
                            density, density_static = density_fns[i_level](spacetime(ray_samples))
                        # print(density.max())
                    else:
                        with Timer("no_updated density_fn query"):
                            with torch.no_grad():
                                # density = density_fns[i_level](ray_samples.frustums.get_positions())
                                density, density_static = density_fns[i_level](spacetime(ray_samples))
                    # print("ray_samples.shape", ray_samples.shape)
                    # print("density.shape", density.shape)
                    with Timer("get weights"):
                        weights = ray_samples.get_weights(density)
                    with Timer("append"):
                        weights_list.append(weights)  # (num_rays, num_samples)
                        # weights_static = ray_samples.get_weights(density_static)
                        # weights_list_static.append(weights_static)  # (num_rays, num_samples)
                        ray_samples_list.append(ray_samples)
            if updated:
                self._steps_since_update = 0

        assert ray_samples is not None
        return ray_samples, weights_list, ray_samples_list  # , weights_list_static
