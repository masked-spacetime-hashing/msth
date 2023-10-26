from MSTH.configs.method_import import *
import numpy as np
import itertools
import random

method_configs: Dict[str, Union[TrainerConfig, VideoTrainerConfig]] = {}


method_configs["rubik1"] = SpaceTimeHashingTrainerConfig(
    method_name="Spatial_Time_Hashing_With_Base",
    steps_per_eval_batch=1000,
    steps_per_save=50000,
    max_num_iterations=10001,
    mixed_precision=True,
    log_gradients=True,
    pipeline=SpaceTimePipelineConfig(
        datamanager=SpaceTimeDataManagerConfig(
            dataparser=VideoDataParserConfig(
                data=Path("/data/machine/data/rubik/videos_2/"),
                # data=Path("/data/machine/data/fit/videos_2"),
                downscale_factor=2,
                # scale_factor=1.0 / 2.0,
                scale_factor=0.5,
            ),
            train_num_rays_per_batch=16384,
            # eval_num_rays_per_batch=32768,
            camera_optimizer=CameraOptimizerConfig(mode="off"),
            use_uint8=True,
            # use_stratified_pixel_sampler=True,
            spatial_temporal_sampler="st",
            # n_time_for_dynamic=3,
            n_time_for_dynamic=lambda x: 1 if x < 1000 else 1 + 5 * np.sin((x - 1000) * np.pi / (2 * (10000 - 1000))),
            static_dynamic_sampling_ratio=50,
            static_dynamic_sampling_ratio_end=30,
            static_ratio_decay_total_steps=10000,
        ),
        model=DSpaceTimeHashingModelConfig(
            # distortion_loss_mult=0.0,
            sparse_loss_mult=1e-6,
            sparse_loss_mult_end=1e-6,
            max_res=(2048, 2048, 2048, 300),
            base_res=(16, 16, 16, 15),
            # num_proposal_samples_per_ray=(256, 96),
            num_proposal_samples_per_ray=(128,),
            num_nerf_samples_per_ray=48,
            proposal_weights_anneal_max_num_iters=5000,
            # proposal_weights_anneal_slope = 10.0,
            log2_hashmap_size_spatial=19,
            log2_hashmap_size_temporal=19,
            eval_num_rays_per_chunk=32768,
            mask_loss_mult=0.1,
            mask_loss_for_proposal=False,
            mst_mode="mst",
            mask_reso=(128, 128, 128),
            mask_log2_hash_size=21,
            mask_type="global",
            st_mlp_mode="shared",
            num_proposal_iterations=1,
            use_loss_static=False,
            render_static=False,
            interp="linear",
            mask_init_mean=-1,
            hidden_dim=128,
            hidden_dim_color=128,
            proposal_net_args_list=[
                {
                    "hidden_dim": 16,
                    "log2_hashmap_size_spatial": 17,
                    "log2_hashmap_size_temporal": 17,
                    "num_levels": 5,
                    "max_res": (128, 128, 128, 150),
                    "base_res": (16, 16, 16, 15),
                    "use_linear": False,
                    "mode": "mst",
                    "mask_reso": (64, 64, 64),
                    "mask_log2_hash_size": 18,
                    "mask_type": "global",
                    "st_mlp_mode": "shared",
                    "interp": "linear",
                    "mask_init_mean": -1,
                },
            ],
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": CosineDecaySchedulerConfig(lr_final=2e-5, max_steps=10000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": CosineDecaySchedulerConfig(lr_final=2e-5, max_steps=10000),
        },
    },
)
