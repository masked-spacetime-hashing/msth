import wandb
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

# import pytest

# import pytest
import torch
from MSTH.dataparser import VideoDataParser, VideoDataParserConfig
from MSTH.dataset import VideoDataset
from MSTH.sampler import CompletePixelSampler, CompletePixelSamplerIter
from MSTH.utils import Timer
import argparse


def test_video_dataparaser():
    parser = VideoDataParserConfig().setup()
    outputs = parser._generate_dataparser_outputs("train")

    print(outputs)


def test_video_dataset_basics():
    parser = VideoDataParserConfig().setup()
    outputs = parser._generate_dataparser_outputs("train")
    print(len(outputs.video_filenames))
    vd = VideoDataset(outputs, 1)

    frame0 = vd[0]["image"]
    plt.imshow(frame0)
    # plt.show()
    vd.tick()
    frame1 = vd[0]["image"]
    plt.imshow(frame1)
    # plt.show()
    mask = vd[0]["mask"]

    plt.imshow(mask)
    plt.show()

    mask_gt = np.linalg.norm(frame1.numpy() - frame0.numpy(), ord=2, axis=-1)
    print(np.mean(mask_gt))
    plt.imshow(mask_gt)
    plt.show()

    del vd


def test_video_dataset_stress():
    parser = VideoDataParserConfig().setup()
    outputs = parser._generate_dataparser_outputs("train")
    print(f"Number of videos loaded: {len(outputs.video_filenames)}")
    vd = VideoDataset(outputs, 1.0)

    for i in range(50):
        with Timer(f"{i}-th tick", record=True):
            vd.tick()

    Timer.show_recorder()


def test_complete_pixel_sampler():
    parser = VideoDataParserConfig().setup()
    outputs = parser._generate_dataparser_outputs("train")
    print(f"Number of videos loaded: {len(outputs.video_filenames)}")
    vd = VideoDataset(outputs, 1.0)

    vd.tick()
    masks = vd.mask
    indices = torch.nonzero(masks[..., 0])
    print(indices.size())

    data = vd.get_all_data()

    # test iter version of sampler
    sampler = CompletePixelSamplerIter(1024, data, False)

    num = 0

    for sample in sampler:
        num += sample["image"].shape[0]

    print(num)
    assert num == indices.size(0)

    # test original version of sampler
    sampler = CompletePixelSampler(1024, True)


def test_space_time_video_dataset():
    from pathlib import Path
    from MSTH.dataset import VideoDatasetAllCached
    from MSTH.sampler import PixelTimeSampler, PixelTimeUniformSampler

    parser = VideoDataParserConfig(data=Path("/data/machine/data/flame_salmon_videos_2"), downscale_factor=2).setup()
    outputs = parser._generate_dataparser_outputs("train")
    with Timer("caching all video frames"):
        vd = VideoDatasetAllCached(outputs)
    sampler = PixelTimeUniformSampler(vd, 64, True)
    sample = sampler.sample()
    print(sample["time"].shape)
    print(sample["time"].max())


def test_ray_samplers():
    from pathlib import Path
    from MSTH.dataset import VideoDatasetAllCached
    from MSTH.sampler import PixelTimeSampler, PixelTimeUniformSampler
    from nerfstudio.model_components.ray_generators import RayGenerator
    from MSTH.datamanager import SpaceTimeDataManagerConfig
    from pprint import pprint

    parser = VideoDataParserConfig(data=Path("/data/machine/data/flame_salmon_videos_test"), downscale_factor=2)
    # outputs = parser._generate_dataparser_outputs("train")
    # with Timer("caching all video frames"):
    #     vd = VideoDatasetAllCached(outputs)
    # sampler = PixelTimeUniformSampler(vd, 64, True)
    # sample = sampler.sample()
    dm = SpaceTimeDataManagerConfig(dataparser=parser).setup()
    ray_bundle, batch = dm.next_train(0)
    print(ray_bundle)
    print(batch.keys())
    print(batch["time"])

    ## test model
    from MSTH.SpaceTimeHashing.model import SpaceTimeHashingModelConfig

    model = SpaceTimeHashingModelConfig().setup(
        scene_box=dm.train_dataset.scene_box, num_train_data=18, metadata=dm.train_dataset.metadata
    )
    model_outputs = model(ray_bundle)

# simulate training
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset

    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})

    # print(weights_list[0].max())
    # print(weights_list[1].max())
    loss = torch.sum(weights_list[0]) + torch.sum(weights_list[1])
    loss.backward()

    print(model.get_param_groups()["proposal_networks"][0].grad.max())
    print(model.get_param_groups()["proposal_networks"][0].grad.min())
    print(model.get_param_groups()["proposal_networks"][1].grad.max())
    print(model.get_param_groups()["proposal_networks"][1].grad.min())

    # inputs = torch.randn([100, 256, 4]).clamp(0, 1)
    # wl = model.density_fns[0](inputs)
    # wl = rs.get_weights(wl)
    # # print(wl.shape)
    # # loss = torch.sum(wl[0]) + torch.sum(wl[1])
    # loss = torch.sum(wl)
    # loss.backward()
    # print(model.get_param_groups()["proposal_networks"][0].grad.max())


def test_pdf_samples():
    from MSTH.SpaceTimeHashing.ray_samplers import PDFSamplerSpatial, UniformSamplerSpatial
    from pathlib import Path
    from MSTH.dataset import VideoDatasetAllCached
    from MSTH.sampler import PixelTimeSampler, PixelTimeUniformSampler
    from nerfstudio.model_components.ray_generators import RayGenerator
    from MSTH.datamanager import SpaceTimeDataManagerConfig

    parser = VideoDataParserConfig(data=Path("/data/machine/data/flame_salmon_videos_test"), downscale_factor=2)
    # outputs = parser._generate_dataparser_outputs("train")
    # with Timer("caching all video frames"):
    #     vd = VideoDatasetAllCached(outputs)
    # sampler = PixelTimeUniformSampler(vd, 64, True)
    # sample = sampler.sample()
    dm = SpaceTimeDataManagerConfig(dataparser=parser).setup()
    uni_sampler = UniformSamplerSpatial(256)
    ray_bundle, batch = dm.next_train(0)
    ray_bundle.nears = torch.zeros(ray_bundle.origins.shape[0], 1) + 5e-3
    ray_bundle.fars = torch.zeros(ray_bundle.origins.shape[0], 1) + 1e2
    ray_samples = uni_sampler(ray_bundle, 256)
    weights = torch.randn(*ray_samples.shape[:-1], 1).clamp(0)
    sampler = PDFSamplerSpatial()
    samples = sampler(ray_bundle, ray_samples, weights, num_samples=48)
    print(samples.shape)


def test_video_cached_dataset_uint8():
    from MSTH.dataset import VideoDatasetAllCachedUint8
    from pathlib import Path

    parser = VideoDataParserConfig(data=Path("/data/machine/data/flame_salmon_videos_2"), downscale_factor=2).setup()
    outputs = parser._generate_dataparser_outputs("train")
    with Timer("caching all video frames"):
        vd = VideoDatasetAllCachedUint8(outputs)


def test_use_mask():
    from MSTH.dataset import VideoDatasetAllCachedUint8
    from pathlib import Path

    parser = VideoDataParserConfig(data=Path("/data/machine/data/flame_salmon_videos_2"), downscale_factor=2).setup()
    outputs = parser._generate_dataparser_outputs("train")
    with Timer("caching all video frames"):
        vd = VideoDatasetAllCachedUint8(outputs, use_mask=True, use_precomputed_mask=False)


def test_pixel_time_sampler():
    from MSTH.dataset import VideoDatasetAllCachedUint8
    from pathlib import Path

    parser = VideoDataParserConfig(data=Path("/data/machine/data/flame_salmon_videos_2"), downscale_factor=2).setup()
    outputs = parser._generate_dataparser_outputs("train")
    with Timer("caching all video frames"):
        vd = VideoDatasetAllCachedUint8(outputs, use_mask=True, use_precomputed_mask=True)
    from MSTH.sampler import PixelTimeSampler

    sampler = PixelTimeSampler(vd, 1024, 1.0)
    samples = sampler.sample()
    indices = samples["indices"]
    c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
    masks = vd.masks[c, y, x]
    print("mask is one: {} / {}".format(torch.count_nonzero(masks), masks.numel()))


def test_rect_model():
    from MSTH.SpaceTimeHashing.rect_model import RectSpaceTimeHashingModelConfig
    from pathlib import Path
    from MSTH.dataset import VideoDatasetAllCached
    from MSTH.sampler import PixelTimeSampler, PixelTimeUniformSampler
    from nerfstudio.model_components.ray_generators import RayGenerator
    from MSTH.datamanager import SpaceTimeDataManagerConfig
    from pprint import pprint

    parser = VideoDataParserConfig(data=Path("/data/machine/data/flame_salmon_videos_test"), downscale_factor=2)

    # torch.set_default_tensor_type(torch.cuda.FloatTensor)
    dm = SpaceTimeDataManagerConfig(dataparser=parser).setup()
    ray_bundle, batch = dm.next_train(0)
    print(ray_bundle)
    print(batch.keys())
    print(batch["time"])

    ## test model
    model = (
        RectSpaceTimeHashingModelConfig()
        .setup(scene_box=dm.train_dataset.scene_box, num_train_data=18, metadata=dm.train_dataset.metadata)
        .to("cuda")
    )
    model_outputs = model(ray_bundle.to("cuda"))
    print(model_outputs)

    # ray_samples, weights_list, ray_samples_list = model.proposal_sampler(ray_bundle, density_fns=model.density_fns)

    # model = RectSpaceTimeHashingModelConfig().setup()


def test_ndc():
    from MSTH.SpaceTimeHashing.rect_model import RectSpaceTimeHashingModelConfig
    from pathlib import Path
    from MSTH.dataset import VideoDatasetAllCached
    from MSTH.sampler import PixelTimeSampler, PixelTimeUniformSampler
    from nerfstudio.model_components.ray_generators import RayGenerator
    from MSTH.datamanager import SpaceTimeDataManagerConfig
    from pprint import pprint

    parser = VideoDataParserConfig(data=Path("/data/machine/data/flame_salmon_videos_test"), downscale_factor=2)
    from MSTH.utils import convert_to_ndc

    # torch.set_default_tensor_type(torch.cuda.FloatTensor)
    dm = SpaceTimeDataManagerConfig(dataparser=parser).setup()
    ray_bundle, batch = dm.next_train(0)
    print(ray_bundle)
    cameras = dm.train_dataparser_outputs.cameras
    convert_to_ndc(ray_bundle, dm.train_dataset.metadata["ndc_coeffs"])
    print(ray_bundle)
    # print(batch.keys())
    # print(batch["time"])

    # ## test model
    # model = RectSpaceTimeHashingModelConfig(use_ndc=True).setup(scene_box=dm.train_dataset.scene_box, num_train_data=18, metadata=dm.train_dataset.metadata).to("cuda")
    # model_outputs = model(ray_bundle.to("cuda"))


def test_dataset_get_median():
    from MSTH.dataset import VideoDatasetAllCachedUint8
    from MSTH.dataset import VideoDatasetAllCachedUint8
    from pathlib import Path

    parser = VideoDataParserConfig(data=Path("/data/machine/data/flame_salmon_videos_2"), downscale_factor=2).setup()
    outputs = parser._generate_dataparser_outputs("train")
    with Timer("caching all video frames"):
        vd = VideoDatasetAllCachedUint8(outputs, use_median=True)


def test_isg_sampler():
    from MSTH.sampler import ISGSampler
    from MSTH.dataset import VideoDatasetAllCachedUint8
    from pathlib import Path

    parser = VideoDataParserConfig(data=Path("/data/machine/data/flame_salmon_videos_2"), downscale_factor=2).setup()
    outputs = parser._generate_dataparser_outputs("train")
    with Timer("caching all video frames"):
        vd = VideoDatasetAllCachedUint8(outputs, use_median=True)
    sampler = ISGSampler(vd, 128, 32, 1e-3)
    with Timer("sampler once from ISG sampler"):
        sampler.sample()


def test_torchvision_video():
    from torchvision.io import write_video

    frames = torch.zeros([100, 512, 512, 3], dtype=torch.uint8)
    frames[::10] = 255
    write_video("/data/czl/tmp/test_video.mp4", frames, 30)


def test_render_spiral(path, save_path):
    from MSTH.video_pipeline import VideoPipeline
    from MSTH.SpaceTimeHashing.trainer import SpaceTimeHashingTrainerConfig
    from MSTH.configs.method_configs import method_configs
    from nerfstudio.utils.eval_utils import eval_setup
    from pathlib import Path

    # trainer = method_configs["dsth_with_base"].setup(local_rank=0, world_size=1)
    # trainer.setup()
    # pipeline = trainer.pipeline
    # config = torch.load("/data/czl/nerf/MSTH_new/MSTH/tmp/dsth_with_base/Spatial_Time_Hashing_With_Base/2023-04-06_013551/nerfstudio_models/step-000029999.ckpt")
    # pipeline.load_pipeline(config["pipeline"], 29999)
    # # pipeline.render_from_cameras(1.0, 5.0, save_path="/data/czl/tmp/test.mp4", fps=5, num_frames=5)
    # trainer.mock_eval()
    config_path = Path(path)
    _, pipeline, _ = eval_setup(config_path)
    print(pipeline.config)
    pipeline.mock_eval()
    pipeline.render_from_cameras(1.0, 5.0, save_path=save_path, fps=30, num_frames=300)


def test_load():
    from MSTH.SpaceTimeHashing.trainer import SpaceTimeHashingTrainer
    from MSTH.configs.method_configs import method_configs
    from MSTH.video_pipeline import VideoPipeline
    from MSTH.SpaceTimeHashing.trainer import SpaceTimeHashingTrainerConfig
    from MSTH.configs.method_configs import method_configs
    from nerfstudio.utils.eval_utils import eval_setup
    from pathlib import Path

    config = method_configs["dsth_with_base"]
    config.save_only_latest_checkpoint = False
    config.max_num_iterations = 500
    config.viewer.quit_on_train_completion = True
    trainer = config.setup()
    config.save_config()
    trainer.setup()
    trainer.train()

    ## save ckpt
    trainer.save_checkpoint(0)

    ckpt_dir = trainer.checkpoint_dir

    config_path = ckpt_dir.parent.absolute() / "config.yml"

    _, pipeline, _ = eval_setup(config_path)

    trainer.mock_eval()
    cpipeline = trainer.pipeline
    pipeline.mock_eval()

    cf = cpipeline.get_param_groups()["fields"]
    f = pipeline.get_param_groups()["fields"]
    with torch.no_grad():
        for p, q in zip(cf, f):
            print(torch.abs(p - q).max())

    print("hello")

def test_mask_occupancy():
    from MSTH.video_pipeline import VideoPipeline
    from MSTH.SpaceTimeHashing.trainer import SpaceTimeHashingTrainerConfig 
    from MSTH.configs.method_configs import method_configs
    from nerfstudio.utils.eval_utils import eval_setup
    from pathlib import Path
    from MSTH.SpaceTimeHashing.viz import gen_spatial_grid, viz_histograms, viz_distribution
    from tqdm import trange
    from varname import nameof
    

    config_path = Path("/data/czl/nerf/MSTH_new/tmp/czl_1/Spatial_Time_Hashing_With_Base/2023-04-07_202608/config.yml")
    _, pipeline, _ = eval_setup(config_path)
    pipeline.mock_eval()
    model = pipeline.model
    grid = gen_spatial_grid().to(model.device)
    print(grid.max())
    print(grid.min())
    chunk_size = 1 << 17
    tot_size = grid.size(0)
    prop1_vals = torch.zeros([tot_size, 1])
    prop2_vals = torch.zeros([tot_size, 1])
    nerf_vals = torch.zeros([tot_size, 1])
    
    @torch.no_grad()
    def get_val(grid, net, ret):
        for start in trange(0, tot_size, chunk_size):
            end = min(start+chunk_size, tot_size)
            ret[start:end] = net(grid[start:end])[..., 0:1].to(ret)
    
    get_val(grid, model.proposal_networks[0].mlp_base.temporal_prod_net, prop1_vals)
    get_val(grid, model.proposal_networks[1].mlp_base.temporal_prod_net, prop2_vals)
    get_val(grid, model.field.mlp_base.temporal_prod_net, nerf_vals)

    viz_histograms([prop1_vals, prop2_vals, nerf_vals], names=["prop1", "prop2", "nerf"])

def test_draw_dist():
    from MSTH.video_pipeline import VideoPipeline
    from MSTH.SpaceTimeHashing.trainer import SpaceTimeHashingTrainerConfig 
    from MSTH.configs.method_configs import method_configs
    from nerfstudio.utils.eval_utils import eval_setup
    from pathlib import Path
    from MSTH.SpaceTimeHashing.viz import gen_spatial_grid, viz_histograms, viz_distribution
    from tqdm import trange
    from varname import nameof
    

    config_path = Path("/data/czl/nerf/MSTH_new/tmp/czl_1/Spatial_Time_Hashing_With_Base/2023-04-07_202608/config.yml")
    _, pipeline, _ = eval_setup(config_path)
    pipeline.mock_eval()
    model = pipeline.model
    _, ray_bundle, _ = pipeline.datamanager.next_eval_image(0)
    start_idx = 500
    end_idx = 510
    ray_bundle = ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
    ray_bundle = model.collider(ray_bundle)
    samples, _, _ = model.proposal_sampler(ray_bundle, model.density_fns)
    ts = samples.frustums.starts
    print(ts.shape)
    viz_distribution(ts)

def test_3d_mask():
    import plotly.express as px

    from MSTH.video_pipeline import VideoPipeline
    from MSTH.SpaceTimeHashing.trainer import SpaceTimeHashingTrainerConfig 
    from MSTH.configs.method_configs import method_configs
    from nerfstudio.utils.eval_utils import eval_setup
    from pathlib import Path
    from MSTH.SpaceTimeHashing.viz import gen_spatial_grid, viz_histograms, viz_distribution
    from tqdm import trange
    from varname import nameof
    import plotly.graph_objects as go
    

    config_path = Path("/data/czl/nerf/MSTH_new/tmp/czl_1/Spatial_Time_Hashing_With_Base/2023-04-07_202608/config.yml")
    _, pipeline, _ = eval_setup(config_path)

    trainer.mock_eval()
    cpipeline = trainer.pipeline
    pipeline.mock_eval()
    model = pipeline.model
    grid = gen_spatial_grid().to(model.device)
    print(grid.max())
    print(grid.min())
    chunk_size = 1 << 17
    tot_size = grid.size(0)
    prop1_vals = torch.zeros([tot_size, 1])
    prop2_vals = torch.zeros([tot_size, 1])
    nerf_vals = torch.zeros([tot_size, 1])
    
    @torch.no_grad()
    def get_val(grid, net, ret):
        for start in trange(0, tot_size, chunk_size):
            end = min(start+chunk_size, tot_size)
            ret[start:end] = net(grid[start:end])[..., 0:1].to(ret)
    
    # get_val(grid, model.proposal_networks[0].mlp_base.temporal_prod_net, prop1_vals)
    # get_val(grid, model.proposal_networks[1].mlp_base.temporal_prod_net, prop2_vals)
    get_val(grid, model.field.mlp_base.temporal_prod_net, nerf_vals)
    
    nerf_vals = nerf_vals.reshape(256, 256, 256)
    x, y, z = torch.nonzero(nerf_vals > 0.8, as_tuple=True)
    selected = nerf_vals[x, y, z].cpu().numpy()
    x = x.float()
    y = y.float()
    z = z.float()
    x /= 256
    y /= 256
    z /= 256
    
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=1/300, color=selected, colorscale='Viridis'))])

    fig.show()

def test_larger_near():
    from MSTH.video_pipeline import VideoPipeline
    from MSTH.SpaceTimeHashing.trainer import SpaceTimeHashingTrainerConfig 
    from MSTH.configs.method_configs import method_configs
    from nerfstudio.utils.eval_utils import eval_setup
    from pathlib import Path
    from MSTH.SpaceTimeHashing.viz import gen_spatial_grid, viz_histograms, viz_distribution
    from tqdm import trange
    from varname import nameof
    

    config_path = Path("/data/czl/nerf/MSTH_new/tmp/imm-2/Spatial_Time_Hashing_With_Base/2023-04-08_003729/config.yml")
    _, pipeline, _ = eval_setup(config_path)
    pipeline.mock_eval()
    pipeline.render_from_cameras(1.0, 5.0, save_path="./test_larger_near.mp4", fps=3, num_frames=30)
    
    

if __name__ == "__main__":
    # test_ray_samplers()
    # test_video_cached_dataset_uint8()
    # test_pdf_samples()
    # test_use_mask()
    # test_pixel_time_sampler()
    # test_use_mask()
    # test_rect_model()
    # test_ray_samplers()
    # test_ndc()
    # test_dataset_get_median()
    # test_isg_sampler()
    # test_torchvision_video()
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--path",
    #     type=str,
    #     default="/data/czl/nerf/MSTH_new/MSTH/tmp/dsth_with_base/Spatial_Time_Hashing_With_Base/2023-04-06_013551/config.yml",
    # )
    # parser.add_argument(
    #     "--save_path",
    #     type=str,
    #     default="/data/czl/tmp/test_v1.2_rawvideo.avi",
    # )
    # args = parser.parse_args()
    # test_render_spiral(args.path, args.save_path)
    # test_load()
    # test_mask_occupancy()
    # test_draw_dist()
    # test_3d_mask()
    test_larger_near()
