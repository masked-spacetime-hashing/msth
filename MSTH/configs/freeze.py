from MSTH.configs.method_import import *

method_configs: Dict[str, Union[TrainerConfig, VideoTrainerConfig]] = {}

method_configs["freeze_mask"] = SpaceTimeHashingTrainerConfig(
    method_name="stmodel with mask freeze",
    steps_per_eval_batch=1000,
    steps_per_save=20000,
    max_num_iterations=30000,
    mixed_precision=True,
    log_gradients=True,
    pipeline=SpaceTimePipelineConfig(
        datamanager=SpaceTimeDataManagerConfig(
            dataparser=VideoDataParserConfig(
                # data=Path("/data/machine/data/flame_salmon_videos_2"),
                data=Path("/data/machine/data/flame_salmon_videos_2"),
                # data=Path("/data/machine/data/flame_salmon_videos_test"),
                downscale_factor=2,
                scale_factor=1 / 2.0,
                # scene_scale=8,
            ),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(mode="off"),
            use_uint8=True,
            use_stratified_pixel_sampler=True,
            static_dynamic_sampling_ratio=50.0,
            static_dynamic_sampling_ratio_end=10.0,
            static_ratio_decay_total_steps=20000,
        ),
        model=DSpaceTimeHashingModelConfig(
            freeze_mask=True,
            freeze_mask_step=7000,
            max_res=(2048, 2048, 2048, 300),
            base_res=(16, 16, 16, 30),
            proposal_weights_anneal_max_num_iters=5000,
            # proposal_weights_anneal_slope = 10.0,
            log2_hashmap_size_spatial=19,
            log2_hashmap_size_temporal=21,
            proposal_net_args_list=[
                {
                    "hidden_dim": 16,
                    "log2_hashmap_size_spatial": 17,
                    "log2_hashmap_size_temporal": 17,
                    "num_levels": 5,
                    "max_res": (128, 128, 128, 150),
                    "base_res": (16, 16, 16, 30),
                    "use_linear": False,
                },
                {
                    "hidden_dim": 16,
                    "log2_hashmap_size_spatial": 17,
                    "log2_hashmap_size_temporal": 17,
                    "num_levels": 5,
                    "max_res": (256, 256, 256, 300),
                    "base_res": (16, 16, 16, 30),
                    "use_linear": False,
                },
            ],
            # use_field_with_base=True,
            # use_sampler_with_base=True,
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=15000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=15000),
        },
    },
)