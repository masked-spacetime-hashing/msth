from MSTH.configs.method_import import *
from copy import deepcopy

method_configs: Dict[str, Union[TrainerConfig, VideoTrainerConfig]] = {}

method_configs["czl_1"] = SpaceTimeHashingTrainerConfig(
    method_name="Spatial_Time_Hashing_With_Base",
    steps_per_eval_batch=1000,
    steps_per_save=2000000000000,
    max_num_iterations=30000,
    mixed_precision=True,
    log_gradients=True,
    pipeline=SpaceTimePipelineConfig(
        datamanager=SpaceTimeDataManagerConfig(
            dataparser=VideoDataParserConfig(
                data=Path("/data/machine/data/flame_salmon_videos_2"),
                downscale_factor=2,
                # scale_factor=1.0 / 2.0,
                scale_factor=0.5,
            ),
            train_num_rays_per_batch=16384,
            eval_num_rays_per_batch=16384,
            camera_optimizer=CameraOptimizerConfig(mode="off"),
            use_uint8=True,
            use_stratified_pixel_sampler=True,
            static_dynamic_sampling_ratio=50.0,
            static_dynamic_sampling_ratio_end=1.0,
            static_ratio_decay_total_steps=20000,
        ),
        model=DSpaceTimeHashingModelConfig(
            # distortion_loss_mult=0.0,
            max_res=(2048, 2048, 2048, 300),
            base_res=(16, 16, 16, 15),
            num_proposal_samples_per_ray=(256, 96),
            num_nerf_samples_per_ray=48,
            proposal_weights_anneal_max_num_iters=5000,
            # proposal_weights_anneal_slope = 10.0,
            log2_hashmap_size_spatial=19,
            log2_hashmap_size_temporal=19,
            # far_plane=500,
            # proposal_initial_sampler="uniform",
            # sparse_loss_mult_h=0.01,
            # sparse_loss_mult_f=0.01,
            mask_loss_mult=0.01,
            mst_mode="mt",
            mask_reso=(256, 256, 256),
            mask_log2_hash_size=24,
            mask_type="hierarchical",
            use_loss_static=True,
            render_static=True,
            proposal_net_args_list=[
                {
                    "hidden_dim": 16,
                    "log2_hashmap_size_spatial": 17,
                    "log2_hashmap_size_temporal": 17,
                    "num_levels": 5,
                    "max_res": (128, 128, 128, 150),
                    "base_res": (16, 16, 16, 15),
                    "use_linear": False,
                    "mode": "mt",
                    "mask_reso": (256, 256, 256),
                    "mask_log2_hash_size": 24,
                    "mask_type": "hierarchical",
                },
                {
                    "hidden_dim": 16,
                    "log2_hashmap_size_spatial": 17,
                    "log2_hashmap_size_temporal": 17,
                    "num_levels": 5,
                    "max_res": (256, 256, 256, 150),
                    "base_res": (16, 16, 16, 15),
                    "use_linear": False,
                    "mode": "mt",
                    "mask_reso": (256, 256, 256),
                    "mask_log2_hash_size": 24,
                    "mask_type": "hierarchical",
                },
            ],
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-4, max_steps=30000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-4, max_steps=30000),
        },
    },
)

method_configs["2_lower_ratio"] = deepcopy(method_configs["czl_1"])
