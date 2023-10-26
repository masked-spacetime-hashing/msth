"""
Put all the method implementations in one location.
"""

from MSTH.configs.method_import import *

method_configs: Dict[str, Union[TrainerConfig, VideoTrainerConfig]] = {}
descriptions = {
    "nerfacto": "Recommended real-time model tuned for real captures. This model will be continually updated.",
    "depth-nerfacto": "Nerfacto with depth supervision.",
    "volinga": "Real-time rendering model from Volinga. Directly exportable to NVOL format at https://volinga.ai/",
    "instant-ngp": "Implementation of Instant-NGP. Recommended real-time model for unbounded scenes.",
    "instant-ngp-bounded": "Implementation of Instant-NGP. Recommended for bounded real and synthetic scenes",
    "mipnerf": "High quality model for bounded scenes. (slow)",
    "semantic-nerfw": "Predicts semantic segmentations and filters out transient objects.",
    "vanilla-nerf": "Original NeRF model. (slow)",
    "tensorf": "tensorf",
    "dnerf": "Dynamic-NeRF model. (slow)",
    "phototourism": "Uses the Phototourism data.",
    "nerfplayer-nerfacto": "NeRFPlayer with nerfacto backbone.",
    "nerfplayer-ngp": "NeRFPlayer with InstantNGP backbone.",
}


# method_configs["nerfacto_split"] = TrainerConfig(
#     method_name="nerfacto_split",
#     steps_per_eval_batch=500,
#     steps_per_save=2000,
#     max_num_iterations=30000,
#     mixed_precision=True,
#     pipeline=VanillaPipelineConfig(
#         datamanager=VanillaDataManagerConfig(
#             dataparser=NerfstudioDataParserConfig(scale_factor=1.0, train_val_json_split=True),
#             train_num_rays_per_batch=4096,
#             eval_num_rays_per_batch=4096,
#             camera_optimizer=CameraOptimizerConfig(
#                 mode="off",
#             ),
#         ),
#         model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15, use_appearance_embedding=True),
#     ),
#     optimizers={
#         "proposal_networks": {
#             "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=30000),
#         },
#         "fields": {
#             "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=30000),
#         },
#     },
#     viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
#     vis="viewer",
# )

# method_configs["nerfacto_split_deferred"] = TrainerConfig(
#     method_name="nerfacto_split_deferred",
#     steps_per_eval_batch=500,
#     steps_per_save=2000,
#     max_num_iterations=10000,
#     mixed_precision=True,
#     pipeline=VanillaPipelineConfig(
#         datamanager=VanillaDataManagerConfig(
#             dataparser=NerfstudioDataParserConfig(scale_factor=1.0, train_val_json_split=True),
#             train_num_rays_per_batch=16384,
#             eval_num_rays_per_batch=16384,
#             camera_optimizer=CameraOptimizerConfig(
#                 mode="off",
#             ),
#         ),
#         model=DeferredNerfactoModelConfig(eval_num_rays_per_chunk=1 << 15, use_appearance_embedding=False),
#     ),
#     optimizers={
#         "proposal_networks": {
#             "optimizer": AdamOptimizerConfig(lr=5e-3, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=30000),
#         },
#         "fields": {
#             "optimizer": AdamOptimizerConfig(lr=5e-3, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=30000),
#         },
#     },
#     viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
#     vis="viewer",
# )

# method_configs["nerfacto_split_streamable"] = TrainerConfig(
#     method_name="nerfacto_split_streamable",
#     steps_per_eval_batch=500,
#     steps_per_save=2000,
#     max_num_iterations=30000,
#     mixed_precision=True,
#     pipeline=VanillaPipelineConfig(
#         datamanager=VanillaDataManagerConfig(
#             dataparser=NerfstudioDataParserConfig(scale_factor=1.0, train_val_json_split=True),
#             train_num_rays_per_batch=4096,
#             eval_num_rays_per_batch=4096,
#             camera_optimizer=CameraOptimizerConfig(
#                 mode="off",
#             ),
#         ),
#         model=StreamableNerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
#     ),
#     optimizers={
#         "proposal_networks": {
#             "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=30000),
#         },
#         "fields": {
#             "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=30000),
#         },
#     },
#     viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
#     vis="viewer",
# )

# # method_configs["nerfacto2"] = method_configs["nerfacto"]
# # method_configs["nerfacto2"].pipeline.datamanager.dataparser.scale_factor = 1 / 16.0
# # method_configs["nerfacto2"].pipeline.datamanager.dataparser.scene_scale = 1.0
# # method_configs["nerfacto2"].pipeline.model.distortion_loss_mult = 0.0
# # method_configs["nerfacto2"].pipeline.model.contraction_type = ContractionType.AABB
# # method_configs["nerfacto2"].optimizers["fields"]["scheduler"] = ExponentialDecaySchedulerConfig(
# #     lr_final=0.0001, max_steps=30000
# # )
# # method_configs["nerfacto2"].optimizers["fields"]["optimizer"].lr = 3e-2


# from MSTH.datamanager import VideoDataManagerConfig
# from MSTH.dataparser import VideoDataParserConfig
# from MSTH.video_pipeline import VideoPipelineConfig

# method_configs["video-tensorf-baseline"] = VideoTrainerConfig(
#     method_name="video-tensorf-baseline",
#     steps_per_eval_batch=500,
#     steps_per_save=2000,
#     num_static_iterations=30000,
#     num_dynamic_frames=1,
#     num_dynamic_iterations=10000,
#     mixed_precision=False,
#     pipeline=VideoPipelineConfig(
#         datamanager=VideoDataManagerConfig(
#             dataparser=VideoDataParserConfig(),
#         ),
#         model=TensoRFModelConfig(),
#     ),
#     optimizers={
#         "fields": {
#             "optimizer": AdamOptimizerConfig(lr=0.001),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=30000),
#         },
#         "encodings": {
#             "optimizer": AdamOptimizerConfig(lr=0.02),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.002, max_steps=30000),
#         },
#     },
#     vis="wandb",
# )

# method_configs["video-nerfacto-baseline"] = VideoTrainerConfig(
#     method_name="video-nerfacto-baseline",
#     skip_static=True,
#     static_model_path="/opt/czl/nerf/MSTH/MSTH/tmp/unnamed/video-nerfacto-baseline/2023-03-19_110928/nerfstudio_models/static-step-000029999.ckpt",
#     steps_per_eval_batch=500,
#     steps_per_save=2000,
#     num_static_iterations=30000,
#     num_dynamic_frames=1,
#     num_dynamic_iterations=10000,
#     mixed_precision=True,
#     pipeline=VideoPipelineConfig(
#         datamanager=VideoDataManagerConfig(
#             dataparser=VideoDataParserConfig(scale_factor=1 / 4.0, scene_scale=2),
#             train_num_rays_per_batch=4096,
#             eval_num_rays_per_batch=4096,
#             camera_optimizer=CameraOptimizerConfig(mode="off"),
#         ),
#         model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15, use_appearance_embedding=False),
#     ),
#     optimizers={
#         "proposal_networks": {
#             "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=30000),
#         },
#         "fields": {
#             "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=30000),
#         },
#     },
#     vis="wandb",
# )

# method_configs["stream-nerfacto-baseline"] = VideoTrainerConfig(
#     method_name="stream-nerfacto-baseline",
#     steps_per_eval_batch=500,
#     steps_per_save=2000,
#     num_static_iterations=30000,
#     skip_static=True,
#     static_model_path="/opt/czl/nerf/MSTH/MSTH/tmp/streamable-nerfacto/stream-nerfacto-baseline/2023-03-19_155918/nerfstudio_models/dynamic-step-000000000.ckpt",
#     num_dynamic_frames=1,
#     num_dynamic_iterations=10000,
#     mixed_precision=True,
#     pipeline=VideoPipelineConfig(
#         datamanager=VideoDataManagerConfig(
#             dataparser=VideoDataParserConfig(scale_factor=1),
#             train_num_rays_per_batch=16384 * 2,
#             eval_num_rays_per_batch=16384,
#             camera_optimizer=CameraOptimizerConfig(mode="off"),
#         ),
#         model=StreamableNerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
#     ),
#     optimizers={
#         "proposal_networks": {
#             "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=30000),
#         },
#         "fields": {
#             "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=30000),
#         },
#     },
#     vis="wandb+viewer",
# )

method_configs["instant-ngp"] = TrainerConfig(
    method_name="instant-ngp",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=DynamicBatchPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(train_val_json_split=True),
            train_num_rays_per_batch=8192,
        ),
        model=InstantNGPModelConfig(eval_num_rays_per_chunk=8192),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        }
    },
    viewer=ViewerConfig(num_rays_per_chunk=64000),
    vis="viewer",
)

method_configs["nerfacto"] = VideoTrainerConfig(
    method_name="nerfacto",
    # skip_static=True,
    # static_model_path="/opt/czl/nerf/MSTH/MSTH/tmp/unnamed/video-nerfacto-baseline/2023-03-19_110928/nerfstudio_models/static-step-000029999.ckpt",
    steps_per_eval_batch=1000,
    steps_per_save=2000,
    mixed_precision=True,
    max_num_iterations=10000,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(mode="off"),
        ),
        model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=30000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=30000),
        },
    },
    vis="wandb+viewer",
)
# method_configs["stream-nerfacto-baseline-short"] = VideoTrainerConfig(
#     method_name="stream-nerfacto-baseline",
#     # skip_static=True,
#     # static_model_path="/opt/czl/nerf/MSTH/MSTH/tmp/unnamed/video-nerfacto-baseline/2023-03-19_110928/nerfstudio_models/static-step-000029999.ckpt",
#     steps_per_eval_batch=1000,
#     steps_per_save=2000,
#     num_static_iterations=10000,
#     num_dynamic_frames=0,
#     num_dynamic_iterations=100,
#     mixed_precision=True,
#     pipeline=VideoPipelineConfig(
#         datamanager=VideoDataManagerConfig(
#             num_frames=300,
#             dataparser=VideoDataParserConfig(scale_factor=1),
#             train_num_rays_per_batch=4096,
#             eval_num_rays_per_batch=4096,
#             camera_optimizer=CameraOptimizerConfig(mode="off"),
#         ),
#         model=StreamableNerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
#     ),
#     optimizers={
#         "proposal_networks": {
#             "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=30000),
#         },
#         "fields": {
#             "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=30000),
#         },
#     },
#     vis="wandb+viewer",
# )

# method_configs["stream-nerfacto-baseline-short-test1"] = copy.deepcopy(method_configs["stream-nerfacto-baseline-short"])
# method_configs["stream-nerfacto-baseline-short-test1"].pipeline.model.max_res = 500

# method_configs["stream-nerfacto-baseline-hashlarge-short"] = VideoTrainerConfig(
#     method_name="stream-nerfacto-baseline",
#     # skip_static=True,
#     # static_model_path="/opt/czl/nerf/MSTH/MSTH/tmp/unnamed/video-nerfacto-baseline/2023-03-19_110928/nerfstudio_models/static-step-000029999.ckpt",
#     steps_per_eval_batch=1000,
#     steps_per_save=2000,
#     num_static_iterations=10000,
#     num_dynamic_frames=0,
#     num_dynamic_iterations=10000,
#     mixed_precision=True,
#     pipeline=VideoPipelineConfig(
#         datamanager=VideoDataManagerConfig(
#             dataparser=VideoDataParserConfig(scale_factor=1),
#             train_num_rays_per_batch=4096,
#             eval_num_rays_per_batch=4096,
#             camera_optimizer=CameraOptimizerConfig(mode="off"),
#         ),
#         model=StreamableNerfactoModelConfig(eval_num_rays_per_chunk=1 << 15, log2_hashmap_size=25),
#     ),
#     optimizers={
#         "proposal_networks": {
#             "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=30000),
#         },
#         "fields": {
#             "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=30000),
#         },
#     },
#     vis="wandb+viewer",
# )

# method_configs["stream-nerfacto-baseline-short-tiled"] = VideoTrainerConfig(
#     method_name="stream-nerfacto-baseline",
#     # skip_static=True,
#     # static_model_path="/opt/czl/nerf/MSTH/MSTH/tmp/unnamed/video-nerfacto-baseline/2023-03-19_110928/nerfstudio_models/static-step-000029999.ckpt",
#     steps_per_eval_batch=1000,
#     steps_per_save=2000,
#     num_static_iterations=10000,
#     num_dynamic_frames=0,
#     num_dynamic_iterations=10000,
#     mixed_precision=True,
#     pipeline=VideoPipelineConfig(
#         datamanager=VideoDataManagerConfig(
#             dataparser=VideoDataParserConfig(scale_factor=1),
#             train_num_rays_per_batch=4096,
#             eval_num_rays_per_batch=4096,
#             camera_optimizer=CameraOptimizerConfig(mode="off"),
#         ),
#         model=StreamableNerfactoModelConfig(
#             num_levels=1,
#             features_per_level=4,
#             base_res=320,
#             max_res=320,
#             log2_hashmap_size=28,
#             gridtype='tiled',
#             eval_num_rays_per_chunk=1 << 15
#         ),
#     ),
#     optimizers={
#         "proposal_networks": {
#             "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=30000),
#         },
#         "fields": {
#             "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=30000),
#         },
#     },
#     vis="wandb+viewer",
# )

# method_configs["stream-nerfacto-baseline-short-hash-nomlp"] = VideoTrainerConfig(
#     method_name="stream-nerfacto-baseline-nomlp",
#     # skip_static=True,
#     # static_model_path="/data/machine/nerfstudio/outputs/flame_salmon_videos/stream-nerfacto-baseline-nomlp/2023-03-20_184257/nerfstudio_models/step-000008000.ckpt",
#     steps_per_eval_batch=1000,
#     steps_per_save=2000,
#     num_static_iterations=10000,
#     # num_dynamic_frames=1,
#     num_dynamic_iterations=10000,
#     mixed_precision=True,
#     pipeline=VideoPipelineConfig(
#         datamanager=VideoDataManagerConfig(
#             dataparser=VideoDataParserConfig(scale_factor=1),
#             train_num_rays_per_batch=4096,
#             eval_num_rays_per_batch=4096,
#             camera_optimizer=CameraOptimizerConfig(mode="off"),
#         ),
#         model=DeferredNerfactoModelConfig(
#             num_levels=16,
#             features_per_level=2,
#             base_res=16,
#             max_res=2048,
#             log2_hashmap_size=22,
#             gridtype='hash',
#             eval_num_rays_per_chunk=1 << 15,
#             proposal_net_args_list = [
#                     {"log2_hashmap_size": 17, "num_levels": 5, "max_res": 128, "use_linear": False, "base_res": 16, "gridtype": "hash"},
#                     {"log2_hashmap_size": 17, "num_levels": 5, "max_res": 256, "use_linear": False, "base_res": 16, "gridtype": "hash"},
#             ]
#         ),
#     ),
#     optimizers={
#         "proposal_networks": {
#             "optimizer": AdamOptimizerConfig(lr=1e-1, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=10000),
#         },
#         "fields_grid": {
#             "optimizer": AdamOptimizerConfig(lr=1e-1, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=10000),
#         },
#         "fields_mlp": {
#             "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=10000),
#         },
#     },
#     vis="wandb+viewer",
# )


# method_configs["stream-nerfacto-baseline-short-tiled-nomlp"] = VideoTrainerConfig(
#     method_name="stream-nerfacto-baseline-nomlp",
#     # skip_static=True,
#     # static_model_path="/data/machine/nerfstudio/outputs/flame_salmon_videos/stream-nerfacto-baseline-nomlp/2023-03-20_184257/nerfstudio_models/step-000008000.ckpt",
#     steps_per_eval_batch=1000,
#     steps_per_save=2000,
#     num_static_iterations=10000,
#     # num_dynamic_frames=1,
#     num_dynamic_iterations=10000,
#     mixed_precision=True,
#     pipeline=VideoPipelineConfig(
#         datamanager=VideoDataManagerConfig(
#             dataparser=VideoDataParserConfig(scale_factor=1),
#             train_num_rays_per_batch=4096,
#             eval_num_rays_per_batch=4096,
#             camera_optimizer=CameraOptimizerConfig(mode="off"),
#         ),
#         model=DeferredNerfactoModelConfig(
#             num_levels=4,
#             features_per_level=2,
#             base_res=128,
#             max_res=1000,
#             log2_hashmap_size=28,
#             gridtype='tiled',
#             eval_num_rays_per_chunk=1 << 15,
#             proposal_net_args_list = [
#                     {"log2_hashmap_size": 28, "num_levels": 5, "max_res": 128, "use_linear": False, "base_res": 16, "gridtype": "tiled"},
#                     {"log2_hashmap_size": 28, "num_levels": 5, "max_res": 128, "use_linear": False, "base_res": 16, "gridtype": "tiled"},
#             ]
#         ),
#     ),
#     optimizers={
#         "proposal_networks": {
#             "optimizer": AdamOptimizerConfig(lr=1e-1, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=10000),
#         },
#         "fields_grid": {
#             "optimizer": AdamOptimizerConfig(lr=1e-1, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=10000),
#         },
#         "fields_mlp": {
#             "optimizer": AdamOptimizerConfig(lr=3e-2, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=10000),
#         },
#     },
#     vis="wandb+viewer",
# )

# method_configs["stream-nerfacto-baseline-short-tiled-nomlp-dynamic"] = VideoTrainerConfig(
#     method_name="stream-nerfacto-baseline-nomlp",
#     skip_static=True,
#     static_model_path="/data/machine/nerfstudio/outputs/flame_salmon_videos/stream-nerfacto-baseline-nomlp/2023-03-20_211639/nerfstudio_models/step-000008000.ckpt",
#     steps_per_eval_batch=1000,
#     steps_per_save=2000,
#     num_static_iterations=10000,
#     num_dynamic_frames=1,
#     num_dynamic_iterations=10000,
#     mixed_precision=True,
#     pipeline=VideoPipelineConfig(
#         datamanager=VideoDataManagerConfig(
#             dataparser=VideoDataParserConfig(scale_factor=1),
#             train_num_rays_per_batch=8192,
#             eval_num_rays_per_batch=8192,
#             camera_optimizer=CameraOptimizerConfig(mode="off"),
#         ),
#         model=DeferredNerfactoModelConfig(
#             num_levels=4,
#             features_per_level=2,
#             base_res=16,
#             max_res=1000,
#             log2_hashmap_size=28,
#             gridtype='tiled',
#             eval_num_rays_per_chunk=1 << 15,
#             proposal_net_args_list = [
#                     {"log2_hashmap_size": 28, "num_levels": 5, "max_res": 128, "use_linear": False, "base_res": 16, "gridtype": "tiled"},
#                     {"log2_hashmap_size": 28, "num_levels": 5, "max_res": 128, "use_linear": False, "base_res": 16, "gridtype": "tiled"},
#             ]
#         ),
#     ),
#     optimizers={
#         "proposal_networks": {
#             "optimizer": AdamOptimizerConfig(lr=1e-1, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=10000),
#         },
#         "fields_grid": {
#             "optimizer": AdamOptimizerConfig(lr=1e-1, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=10000),
#         },
#         # "fields_mlp": {
#             # "optimizer": AdamOptimizerConfig(lr=3e-2, eps=1e-15),
#             # "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=10000),
#         # },
#     },
#     vis="wandb+viewer",
# )

# method_configs["stream-nerfacto-baseline-short-tiled-nomlp-multilevel"] = VideoTrainerConfig(
#     method_name="stream-nerfacto-baseline-nomlp-onelevel",
#     # skip_static=True,
#     # static_model_path="/data/machine/nerfstudio/outputs/flame_salmon_videos/stream-nerfacto-baseline-nomlp/2023-03-20_184257/nerfstudio_models/step-000008000.ckpt",
#     steps_per_eval_batch=1000,
#     steps_per_save=2000,
#     num_static_iterations=10000,
#     num_dynamic_frames=0,
#     num_dynamic_iterations=10000,
#     mixed_precision=True,
#     upsample_enable=False,
#     upsample_steps=[1001, 1501, 2501, 4001],
#     resolution_list = [32, 64, 256, 512],
#     tv_loss_weight = 0.0,
#     pipeline=VideoPipelineConfig(
#         datamanager=VideoDataManagerConfig(
#             dataparser=VideoDataParserConfig(scale_factor=1),
#             train_num_rays_per_batch=4096,
#             eval_num_rays_per_batch=4096,
#             camera_optimizer=CameraOptimizerConfig(mode="off"),
#         ),
#         model=DeferredNerfactoModelConfig(
#             num_levels=4,
#             features_per_level=2,
#             base_res=16,
#             max_res=2000,
#             log2_hashmap_size=26,
#             gridtype='hash',
#             eval_num_rays_per_chunk=1 << 15,
#             proposal_net_args_list = [
#                     {"log2_hashmap_size": 28, "num_levels": 5, "max_res": 128, "use_linear": False, "base_res": 16, "gridtype": "tiled"},
#                     {"log2_hashmap_size": 28, "num_levels": 5, "max_res": 128, "use_linear": False, "base_res": 16, "gridtype": "tiled"},
#             ]
#         ),
#     ),

#     optimizers={
#         "proposal_networks": {
#             "optimizer": AdamOptimizerConfig(lr=3e-2, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.001, max_steps=30000),
#         },
#         "fields_grid": {
#             "optimizer": AdamOptimizerConfig(lr=3e-2, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.001, max_steps=30000),
#         },
#         "fields_mlp": {
#             "optimizer": AdamOptimizerConfig(lr=3e-2, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.001, max_steps=30000),
#         },
#     },
#     vis="wandb+viewer",
# )

# method_configs["stream-nerfacto-baseline-short-tiled-nomlp-multilevel-dynamic"] = VideoTrainerConfig(
#     method_name="stream-nerfacto-baseline-nomlp-onelevel",
#     skip_static=True,
#     static_model_path="/data/machine/nerfstudio/outputs/flame_salmon_videos/stream-nerfacto-baseline-nomlp-onelevel/2023-03-22_024552/nerfstudio_models/dynamic-step-000000000.ckpt",
#     steps_per_eval_batch=1000,
#     all_steps_per_eval_image=100,
#     steps_per_save=2000,
#     num_static_iterations=10000,
#     num_dynamic_frames=1,
#     num_dynamic_iterations=10000,
#     mixed_precision=True,
#     upsample_enable=False,
#     upsample_steps=[1001, 1501, 2501, 4001],
#     resolution_list = [32, 64, 256, 512],
#     tv_loss_weight = 0.0,
#     pipeline=VideoPipelineConfig(
#         datamanager=VideoDataManagerConfig(
#             dataparser=VideoDataParserConfig(scale_factor=1),
#             train_num_rays_per_batch=16384,
#             eval_num_rays_per_batch=16384,
#             camera_optimizer=CameraOptimizerConfig(mode="off"),
#         ),
#         model=DeferredNerfactoModelConfig(
#             num_levels=4,
#             features_per_level=2,
#             base_res=16,
#             max_res=2000,
#             log2_hashmap_size=26,
#             gridtype='hash',
#             eval_num_rays_per_chunk=1 << 15,
#             proposal_net_args_list = [
#                     {"log2_hashmap_size": 28, "num_levels": 5, "max_res": 128, "use_linear": False, "base_res": 16, "gridtype": "tiled"},
#                     {"log2_hashmap_size": 28, "num_levels": 5, "max_res": 128, "use_linear": False, "base_res": 16, "gridtype": "tiled"},
#             ]
#         ),
#     ),

#     optimizers={
#         "proposal_networks": {
#             "optimizer": AdamOptimizerConfig(lr=1e-1, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.001, max_steps=30000),
#         },
#         "fields_grid": {
#             "optimizer": AdamOptimizerConfig(lr=1e-1, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.001, max_steps=30000),
#         },
#         "fields_mlp": {
#             "optimizer": AdamOptimizerConfig(lr=3e-2, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.001, max_steps=30000),
#         },
#     },
#     vis="wandb+viewer",
# )


# method_configs["stream-nerfacto-baseline-short-tiled-nomlp-onelevel"] = VideoTrainerConfig(
#     method_name="stream-nerfacto-baseline-nomlp-onelevel",
#     # skip_static=True,
#     # static_model_path="/data/machine/nerfstudio/outputs/flame_salmon_videos/stream-nerfacto-baseline-nomlp/2023-03-20_184257/nerfstudio_models/step-000008000.ckpt",
#     steps_per_eval_batch=1000,
#     steps_per_save=2000,
#     num_static_iterations=10000,
#     num_dynamic_frames=0,
#     num_dynamic_iterations=10000,
#     mixed_precision=True,
#     upsample_enable=True,
#     upsample_steps=[1001, 1501, 2501, 4001],
#     resolution_list = [32, 64, 256, 512],
#     tv_loss_weight = 0.01,
#     pipeline=VideoPipelineConfig(
#         datamanager=VideoDataManagerConfig(
#             dataparser=VideoDataParserConfig(scale_factor=1),
#             train_num_rays_per_batch=4096,
#             eval_num_rays_per_batch=4096,
#             camera_optimizer=CameraOptimizerConfig(mode="off"),
#         ),
#         model=DeferredNerfactoModelConfig(
#             num_levels=1,
#             features_per_level=2,
#             base_res=16,
#             max_res=16,
#             log2_hashmap_size=28,
#             gridtype='tiled',
#             eval_num_rays_per_chunk=1 << 15,
#             proposal_net_args_list = [
#                     {"log2_hashmap_size": 28, "num_levels": 5, "max_res": 128, "use_linear": False, "base_res": 16, "gridtype": "tiled"},
#                     {"log2_hashmap_size": 28, "num_levels": 5, "max_res": 128, "use_linear": False, "base_res": 16, "gridtype": "tiled"},
#             ]
#         ),
#     ),

#     optimizers={
#         "proposal_networks": {
#             "optimizer": AdamOptimizerConfig(lr=3e-2, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.00001, max_steps=10000),
#         },
#         "fields_grid": {
#             "optimizer": AdamOptimizerConfig(lr=3e-2, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.00001, max_steps=10000),
#         },
#         "fields_mlp": {
#             "optimizer": AdamOptimizerConfig(lr=3e-2, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.00001, max_steps=10000),
#         },
#     },
#     vis="wandb+viewer",
# )

# #
# method_configs["stream-nerfacto-baseline-short-tiled-nomlp-onelevel-dynamic"] = VideoTrainerConfig(
#     method_name="stream-nerfacto-baseline-nomlp-onelevel-dynamic",
#     skip_static=True,
#     static_model_path="/data/machine/nerfstudio/outputs/flame_salmon_videos/stream-nerfacto-baseline-nomlp-onelevel/2023-03-22_001145/nerfstudio_models/dynamic-step-000000000.ckpt",
#     steps_per_eval_batch=1000,
#     all_steps_per_eval_image=100,
#     steps_per_save=2000,
#     num_static_iterations=10000,
#     num_dynamic_frames=1,
#     num_dynamic_iterations=10000,
#     mixed_precision=True,
#     upsample_enable=False,
#     upsample_steps=[1001, 1501, 2501, 3501],
#     resolution_list = [64, 128, 256, 512],
#     pipeline=VideoPipelineConfig(
#         datamanager=VideoDataManagerConfig(
#             dataparser=VideoDataParserConfig(scale_factor=1),
#             train_num_rays_per_batch=8192,
#             eval_num_rays_per_batch=8192,
#             camera_optimizer=CameraOptimizerConfig(mode="off"),
#         ),
#         model=DeferredNerfactoModelConfig(
#             num_levels=1,
#             features_per_level=2,
#             base_res=512,
#             max_res=512,
#             log2_hashmap_size=28,
#             gridtype='tiled',
#             eval_num_rays_per_chunk=1 << 15,
#             proposal_net_args_list = [
#                     {"log2_hashmap_size": 28, "num_levels": 5, "max_res": 128, "use_linear": False, "base_res": 16, "gridtype": "tiled"},
#                     {"log2_hashmap_size": 28, "num_levels": 5, "max_res": 128, "use_linear": False, "base_res": 16, "gridtype": "tiled"},
#             ]
#         ),
#     ),
#     optimizers={
#         "proposal_networks": {
#             "optimizer": AdamOptimizerConfig(lr=4e-2, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=10000),
#         },
#         "fields_grid": {
#             "optimizer": AdamOptimizerConfig(lr=4e-2, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=10000),
#         },
#         "fields_mlp": {
#             "optimizer": AdamOptimizerConfig(lr=3e-2, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=10000),
#         },
#     },
#     vis="wandb+viewer",
# )


# method_configs["stream-nerfacto-baseline-short-tiled2"] = copy.deepcopy(method_configs["stream-nerfacto-baseline-short-tiled"])
# method_configs["stream-nerfacto-baseline-short-tiled2"].pipeline.model.num_levels = 16
# method_configs["stream-nerfacto-baseline-short-tiled2"].pipeline.model.feature_per_level = 2
# method_configs["stream-nerfacto-baseline-short-tiled2"].pipeline.model.base_res = 8
# method_configs["stream-nerfacto-baseline-short-tiled2"].pipeline.model.max_res = 320
# # method_configs["stream-nerfacto-baseline-short-tiled2"].pipeline.model.proposal_net_args_list = field(
#         # default_factory=lambda: [
#             # {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 128, "use_linear": False},
#             # {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256, "use_linear": False},
#         # ]
#     # )
# method_configs["stream-nerfacto-baseline-short-tiled2"].pipeline.model.num_proposal_samples_per_ray = (512, 192)
# method_configs["stream-nerfacto-baseline-short-tiled2"].pipeline.model.num_nerf_samples_per_ray = 96

# method_configs["stream-nerfacto-baseline-short-tiled2-next"] = copy.deepcopy(method_configs["stream-nerfacto-baseline-short-tiled2"])
# method_configs["stream-nerfacto-baseline-short-tiled2-next"].skip_static = True
# method_configs["stream-nerfacto-baseline-short-tiled2-next"].num_dynamic_frames = 1
# method_configs["stream-nerfacto-baseline-short-tiled2-next"].static_model_path="/data/machine/nerfstudio/outputs/flame_salmon_videos/stream-nerfacto-baseline/2023-03-20_131459/nerfstudio_models/dynamic-step-000000000.ckpt"

# method_configs["stream-nerfacto-baseline-short-next"] = VideoTrainerConfig(
#     method_name="stream-nerfacto-baseline",
#     skip_static=True,
#     static_model_path="/data/machine/nerfstudio/outputs/flame_salmon_videos/stream-nerfacto-baseline/2023-03-20_104947/nerfstudio_models/dynamic-step-000000000.ckpt",
#     steps_per_eval_batch=1000,
#     steps_per_save=2000,
#     num_static_iterations=10000,
#     num_dynamic_frames=1,
#     num_dynamic_iterations=10000,
#     mixed_precision=True,
#     pipeline=VideoPipelineConfig(
#         datamanager=VideoDataManagerConfig(
#             dataparser=VideoDataParserConfig(scale_factor=1),
#             train_num_rays_per_batch=4096,
#             eval_num_rays_per_batch=4096,
#             camera_optimizer=CameraOptimizerConfig(mode="off"),
#         ),
#         model=StreamableNerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
#     ),
#     optimizers={
#         "proposal_networks": {
#             "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=30000),
#         },
#         "fields": {
#             "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=30000),
#         },
#     },
#     vis="wandb+viewer",
# )

method_configs["stream-nerfacto-baseline-short-tiled-nomlp-multilevel2"] = VideoTrainerConfig(
    method_name="stream-nerfacto-baseline-nomlp-onelevel",
    # skip_static=True,
    # static_model_path="/data/machine/nerfstudio/outputs/flame_salmon_videos/stream-nerfacto-baseline-nomlp/2023-03-20_184257/nerfstudio_models/step-000008000.ckpt",
    steps_per_eval_batch=1000,
    steps_per_save=2000,
    num_static_iterations=10000,
    num_dynamic_frames=0,
    num_dynamic_iterations=10000,
    mixed_precision=True,
    upsample_enable=False,
    upsample_steps=[1001, 1501, 2501, 4001],
    resolution_list=[32, 64, 256, 512],
    tv_loss_weight=0.00001,
    pipeline=VideoPipelineConfig(
        datamanager=VideoDataManagerConfig(
            dataparser=VideoDataParserConfig(scale_factor=1),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(mode="off"),
        ),
        model=DeferredNerfactoModelConfig(
            num_levels=4,
            features_per_level=2,
            base_res=(16, 16, 16),
            # max_res=(1200, 800, 360),
            max_res=(620, 620, 620),
            log2_hashmap_size=30,
            gridtype="tiled",
            sparsity_loss_weight=0.00001,
            eval_num_rays_per_chunk=1 << 15,
            num_proposal_iterations=2,
            num_proposal_samples_per_ray=(256, 96),
            num_nerf_samples_per_ray=48,
            proposal_net_args_list=[
                {
                    "log2_hashmap_size": 28,
                    "num_levels": 5,
                    "max_res": 128,
                    "use_linear": False,
                    "base_res": 16,
                    "gridtype": "tiled",
                },
                {
                    "log2_hashmap_size": 28,
                    "num_levels": 5,
                    "max_res": 128,
                    "use_linear": False,
                    "base_res": 16,
                    "gridtype": "tiled",
                },
            ],
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=3e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.001, max_steps=20000),
        },
        "fields_grid": {
            "optimizer": AdamOptimizerConfig(lr=3e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.001, max_steps=20000),
        },
        "fields_mlp": {
            "optimizer": AdamOptimizerConfig(lr=3e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.001, max_steps=20000),
        },
    },
    vis="wandb+viewer",
)

method_configs["stream-nerfacto-baseline-short-tiled-nomlp-multilevel2-dynamic"] = VideoTrainerConfig(
    method_name="stream-nerfacto-baseline-nomlp-onelevel",
    skip_static=True,
    static_model_path="/data/machine/nerfstudio/outputs/flame_salmon_videos/stream-nerfacto-baseline-nomlp-onelevel/2023-03-24_092737/nerfstudio_models/dynamic-step-000000000.ckpt",
    steps_per_eval_batch=1000,
    all_steps_per_eval_image=20,
    steps_per_save=2000,
    num_static_iterations=10000,
    num_dynamic_frames=1,
    num_dynamic_iterations=10000,
    mixed_precision=True,
    upsample_enable=False,
    upsample_steps=[1001, 1501, 2501, 4001],
    resolution_list=[32, 64, 256, 512],
    tv_loss_weight=0.0,
    step_scale=1,
    reset_proposal_samplers=False,
    hash_reinit_std=0.5,
    pipeline=VideoPipelineConfig(
        datamanager=VideoDataManagerConfig(
            dataparser=VideoDataParserConfig(scale_factor=1),
            train_num_rays_per_batch=16384,
            eval_num_rays_per_batch=65536,
            camera_optimizer=CameraOptimizerConfig(mode="off"),
            mask_extend_radius=50,
            next_n_frames=30,
        ),
        model=DeferredNerfactoModelConfig(
            num_levels=4,
            features_per_level=2,
            base_res=(16, 16, 16),
            max_res=(620, 620, 620),
            log2_hashmap_size=30,
            gridtype="tiled",
            # sparsity_loss_weight=0.0001,
            eval_num_rays_per_chunk=1 << 15,
            proposal_net_args_list=[
                {
                    "log2_hashmap_size": 28,
                    "num_levels": 5,
                    "max_res": 128,
                    "use_linear": False,
                    "base_res": 16,
                    "gridtype": "tiled",
                },
                {
                    "log2_hashmap_size": 28,
                    "num_levels": 5,
                    "max_res": 128,
                    "use_linear": False,
                    "base_res": 16,
                    "gridtype": "tiled",
                },
            ],
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=3e-1, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.001, max_steps=1000),
        },
        "fields_grid": {
            "optimizer": AdamOptimizerConfig(lr=3e-1, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.001, max_steps=1000),
        },
        "fields_mlp": {
            "optimizer": AdamOptimizerConfig(lr=3e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.001, max_steps=1000),
        },
    },
    vis="wandb+viewer",
)


# method_configs["ibr"] = VideoTrainerConfig(
#     method_name="ibr",
#     # skip_static=True,
#     # static_model_path="/data/machine/nerfstudio/outputs/flame_salmon_videos/stream-nerfacto-baseline-nomlp/2023-03-20_184257/nerfstudio_models/step-000008000.ckpt",
#     steps_per_eval_batch=1000,
#     steps_per_save=2000,
#     num_static_iterations=10000,
#     num_dynamic_frames=0,
#     num_dynamic_iterations=10000,
#     mixed_precision=False,
#     upsample_enable=False,
#     upsample_steps=[1001, 1501, 2501, 4001],
#     resolution_list=[32, 64, 256, 512],
#     tv_loss_weight=0.00001,
#     pipeline=VideoPipelineConfig(
#         datamanager=VideoFeatureDataManagerConfig(
#             dataparser=VideoDataParserConfig(scale_factor=1, data=Path("/data/machine/data/flame_salmon_videos")),
#             train_num_rays_per_batch=512,
#             eval_num_rays_per_batch=256,
#             camera_optimizer=CameraOptimizerConfig(mode="off"),
#         ),
#         model=DeferredNerfactoModelConfig(
#             eval_num_rays_per_chunk=512,
#             num_levels=4,
#             features_per_level=2,
#             base_res=(16, 16, 16),
#             # max_res=(1200, 800, 360),
#             max_res=(620, 620, 620),
#             log2_hashmap_size=30,
#             gridtype="tiled",
#             sparsity_loss_weight=0.00001,
#             num_proposal_iterations=2,
#             num_proposal_samples_per_ray=(256, 96),
#             num_nerf_samples_per_ray=48,
#             proposal_net_args_list=[
#                 {
#                     "log2_hashmap_size": 28,
#                     "num_levels": 5,
#                     "max_res": 128,
#                     "use_linear": False,
#                     "base_res": 16,
#                     "gridtype": "tiled",
#                 },
#                 {
#                     "log2_hashmap_size": 28,
#                     "num_levels": 5,
#                     "max_res": 128,
#                     "use_linear": False,
#                     "base_res": 16,
#                     "gridtype": "tiled",
#                 },
#             ],
#         ),
#     ),
#     optimizers={
#         "proposal_networks": {
#             "optimizer": AdamOptimizerConfig(lr=3e-2, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.001, max_steps=20000),
#         },
#         "fields_grid": {
#             "optimizer": AdamOptimizerConfig(lr=3e-2, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.001, max_steps=20000),
#         },
#         "fields_mlp": {
#             "optimizer": AdamOptimizerConfig(lr=3e-2, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.001, max_steps=20000),
#         },
#         "colorizer": {
#             "optimizer": AdamOptimizerConfig(lr=3e-3, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.001, max_steps=20000),
#         },
#     },
#     vis="wandb+viewer",
# )


method_configs["sth"] = SpaceTimeHashingTrainerConfig(
    method_name="Spatial_Time_Hashing",
    steps_per_eval_batch=1000,
    steps_per_save=2000000000,
    max_num_iterations=30000,
    mixed_precision=False,
    log_gradients=True,
    pipeline=SpaceTimePipelineConfig(
        datamanager=SpaceTimeDataManagerConfig(
            dataparser=VideoDataParserConfig(
                # data=Path("/data/machine/data/flame_salmon_videos_2"),
                data=Path("/data/machine/data/flame_salmon_videos_2"),
                downscale_factor=2,
                scale_factor=1 / 4.0,
                # scene_scale=8,
            ),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(mode="off"),
        ),
        model=SpaceTimeHashingModelConfig(
            max_res=1024,
            log2_hashmap_size=24,
            proposal_net_args_list=[
                {"hidden_dim": 16, "log2_hashmap_size": 18, "num_levels": 5, "max_res": 128, "use_linear": False},
                {"hidden_dim": 16, "log2_hashmap_size": 18, "num_levels": 5, "max_res": 256, "use_linear": False},
            ],
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=20000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=20000),
        },
    },
)

method_configs["dsth_with_base"] = SpaceTimeHashingTrainerConfig(
    method_name="Spatial_Time_Hashing_With_Base",
    steps_per_eval_batch=1000,
    steps_per_save=300,
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
            static_dynamic_sampling_ratio=1.0,
        ),
        model=DSpaceTimeHashingModelConfig(
            eval_num_rays_per_chunk=1 << 16,
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
    vis="tensorboard",
)

method_configs["dsth_with_base_lr1"] = SpaceTimeHashingTrainerConfig(
    method_name="Spatial_Time_Hashing_With_Base2",
    steps_per_eval_batch=1000,
    steps_per_save=20000,
    max_num_iterations=10000,
    mixed_precision=False,
    log_gradients=True,
    pipeline=SpaceTimePipelineConfig(
        datamanager=SpaceTimeDataManagerConfig(
            dataparser=VideoDataParserConfig(
                # data=Path("/data/machine/data/flame_salmon_videos_2"),
                data=Path("/data/machine/data/flame_salmon_videos_2"),
                # data=Path("/data/machine/data/flame_salmon_videos_test"),
                downscale_factor=2,
                scale_factor=1 / 16.0,
                # scene_scale=8,
            ),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(mode="off"),
            use_uint8=True,
            use_stratified_pixel_sampler=True,
            static_dynamic_sampling_ratio=1.0,
        ),
        model=DSpaceTimeHashingModelConfig(
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
            "optimizer": AdamOptimizerConfig(lr=5e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=15000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=15000),
        },
    },
)

method_configs["sth_profiling"] = SpaceTimeHashingTrainerConfig(
    method_name="Spatial_Time_Hashing_Profiling",
    steps_per_eval_batch=1000,
    steps_per_save=2000,
    max_num_iterations=1000,
    mixed_precision=False,
    pipeline=SpaceTimePipelineConfig(
        datamanager=SpaceTimeDataManagerConfig(
            dataparser=VideoDataParserConfig(
                # data=Path("/data/machine/data/flame_salmon_videos_2"),
                data=Path("/data/machine/data/flame_salmon_videos_2"),
                downscale_factor=2,
                scale_factor=1 / 4.0,
            ),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(mode="off"),
        ),
        model=SpaceTimeHashingModelConfig(max_res=1024, log2_hashmap_size=22),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
    },
)

# method_configs["dst"] = SpaceTimeHashingTrainerConfig(
#     method_name="DSpatial_Time_Hashing_With_Base",
#     steps_per_eval_batch=1000,
#     steps_per_save=2000,
#     max_num_iterations=30000,
#     mixed_precision=False,
#     log_gradients=True,
#     pipeline=SpaceTimePipelineConfig(
#         datamanager=SpaceTimeDataManagerConfig(
#             dataparser=VideoDataParserConfig(
#                 # data=Path("/data/machine/data/flame_salmon_videos_2"),
#                 data=Path("/data/machine/data/flame_salmon_videos_2"),
#                 downscale_factor=2,
#                 scale_factor=1 / 4.0,
#                 # scene_scale=8,
#             ),
#             train_num_rays_per_batch=4096,
#             eval_num_rays_per_batch=4096,
#             camera_optimizer=CameraOptimizerConfig(mode="off"),
#         ),
#         model=DSpaceTimeHashingModelConfig(
#             max_res=1024,
#             log2_hashmap_size=23,
#             proposal_net_args_list=[
#                 {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 128, "use_linear": False},
#                 {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256, "use_linear": False},
#             ],
#             use_field_with_base=True,
#             use_sampler_with_base=True,
#         ),
#     ),
#     optimizers={
#         "proposal_networks": {
#             "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=20000),
#         },
#         "fields": {
#             "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
#             "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=20000),
#         },
#     },
# )


method_configs["tp1"] = SpaceTimeHashingTrainerConfig(
    method_name="Spatial_Time_Hashing_With_Base",
    steps_per_eval_batch=1000,
    steps_per_save=20000,
    max_num_iterations=10000,
    mixed_precision=True,
    log_gradients=True,
    pipeline=SpaceTimePipelineConfig(
        datamanager=SpaceTimeDataManagerConfig(
            dataparser=VideoDataParserConfig(
                # data=Path("/data/machine/data/flame_salmon_videos_2"),
                data=Path("/data/machine/data/flame_salmon_videos_2"),
                # data=Path("/data/machine/data/flame_salmon_videos_test"),
                downscale_factor=2,
                scale_factor=1.0 / 2.0,
                # scale_factor=1.0 / 4.0,
                # scale_factor=1.0 / 8.0,
                # auto_scale_poses=False,
                # scale_factor=1 / 8.0,
                # scale_factor=1 / 1.0,
                # scene_scale=8,
            ),
            train_num_rays_per_batch=20000,
            eval_num_rays_per_batch=20000,
            camera_optimizer=CameraOptimizerConfig(mode="off"),
            use_uint8=True,
            use_stratified_pixel_sampler=False,
            static_dynamic_sampling_ratio=50,
            static_dynamic_sampling_ratio_end=1,
            static_ratio_decay_total_steps=6000,
        ),
        model=DSpaceTimeHashingModelConfig(
            # distortion_loss_mult=0.0,
            max_res=(800, 800, 800, 300),
            base_res=(16, 16, 16, 15),
            num_proposal_samples_per_ray=(256, 96),
            num_nerf_samples_per_ray=48,
            proposal_weights_anneal_max_num_iters=5000,
            # proposal_weights_anneal_slope = 10.0,
            log2_hashmap_size_spatial=19,
            log2_hashmap_size_temporal=19,
            far_plane=500,
            proposal_initial_sampler="uniform",
            # sparse_loss_mult_h=0.01,
            # sparse_loss_mult_f=0.01,
            proposal_net_args_list=[
                {
                    "hidden_dim": 16,
                    "log2_hashmap_size_spatial": 17,
                    "log2_hashmap_size_temporal": 17,
                    "num_levels": 5,
                    "max_res": (128, 128, 128, 150),
                    "base_res": (16, 16, 16, 15),
                    "use_linear": False,
                },
                {
                    "hidden_dim": 16,
                    "log2_hashmap_size_spatial": 17,
                    "log2_hashmap_size_temporal": 17,
                    "num_levels": 5,
                    "max_res": (256, 256, 256, 150),
                    "base_res": (16, 16, 16, 15),
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
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-4, max_steps=15000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-4, max_steps=15000),
        },
    },
)


method_configs["nerfacto_profiling"] = TrainerConfig(
    method_name="nerfacto_profiling",
    steps_per_eval_batch=1000,
    steps_per_save=2000,
    log_gradients=True,
    max_num_iterations=30000,
    mixed_precision=False,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(
                data=Path("/data/machine/data/flame_salmon_image"),
                scale_factor=1 / 4.0,
                train_val_json_split=True,
                scene_scale=4,
            ),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            # camera_optimizer=CameraOptimizerConfig(
            # mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            # ),
            camera_optimizer=CameraOptimizerConfig(
                mode="off",
            ),
        ),
        model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15, log2_hashmap_size=22),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="tensorboard",
)


import MSTH.configs.method_configs_tp as method_configs_tp
import MSTH.configs.method_configs_tpa as method_configs_tpa
import MSTH.configs.method_config_parameter_search as method_configs_ps
import MSTH.configs.method_config_ps2 as method_configs_ps2
import MSTH.configs.method_config_coffee as method_config_coffee
import MSTH.configs.method_config_cook_spinach as method_config_cook
import MSTH.configs.method_config_rubik as method_config_rubik
from MSTH.configs.freeze import method_configs as freeze_configs
from MSTH.configs.method_configs_czl import method_configs as mc_czl
from MSTH.configs.method_configs_imm import method_configs as mc_imm

method_configs.update(method_configs_tp.method_configs)
method_configs.update(method_configs_tpa.method_configs)
method_configs.update(method_configs_ps.method_configs)
method_configs.update(method_configs_ps2.method_configs)
method_configs.update(method_config_coffee.method_configs)
method_configs.update(method_config_cook.method_configs)
method_configs.update(method_config_rubik.method_configs)

method_configs.update(mc_czl)

method_configs.update(freeze_configs)

method_configs.update(mc_imm)

from MSTH.scripts import tunning_machine

# base_method = method_configs["base_it40000_st_4"]


# def setp(exps):
#     def setfunc(x, v):
#         if isinstance(exps, (tuple, list)):
#             for exp in exps:
#                 command = "x." + exp + "=v"
#                 exec(command)
#         else:
#             command = "x." + exps + "=v"
#             exec(command)

#     return setfunc


# set_functions = {
#     "dataset": setp("pipeline.datamanager.dataparser.data"),
#     "n_time": setp("pipeline.datamanager.n_time_for_dynamic"),
#     "sampling_ratio_start": setp("pipeline.datamanager.static_dynamic_sampling_ratio"),
#     "sampling_ratio_end": setp("pipeline.datamanager.static_dynamic_sampling_ratio_end"),
#     "sampling_ratio_decay": setp("pipeline.datamanager.static_ratio_decay_total_steps"),
#     "mask_loss_mult": setp("pipeline.model.mask_loss_mult"),
#     "mask_init_mean": setp(
#         ["pipeline.model.mask_init_mean", "pipeline.model.proposal_net_args_list[0]['mask_init_mean']"]
#     ),
#     "scene_scale": setp("pipeline.datamanager.dataparser.scene_scale"),
#     "mask_loss_for_proposal": setp("pipeline.model.mask_loss_for_proposal"),
# }

# potential_values = {
#     "n_time": [
#         lambda x: 1 if x < 1000 else 1 + 5 * np.sin((x - 1000) * np.pi / (2 * (40000 - 1000))),
#     ],
#     "sampling_ratio_start": [10, 8, 5],
#     "sampling_ratio_end": [4, 2],
#     "mask_loss_mult": [
#         0.01,
#         0.1,
#     ],
#     "scene_scale": [0.5, 0.25, 0.125, 0.0625],
#     "dataset": [
#         # Path("/data/machine/data/immersive/09_Alexa_Meade_Exhibit_2"),
#         Path("/data/machine/data/immersive/12_Cave_2"),
#         # Path("/data/machine/data/immersive/05_Horse_2"),
#     ],
#     "mask_loss_for_proposal": [True, False],
# }

# all_hyper_parameter_key = potential_values.keys()
# all_hyper_parameter_value = [potential_values[k] for k in all_hyper_parameter_key]
# import itertools
# import random

# all_specs = list(itertools.product(*all_hyper_parameter_value))
# all_specs = [{k: v for k, v in zip(all_hyper_parameter_key, spec)} for spec in all_specs]
# random.shuffle(all_specs)
# print("==== ALL SPECS ====")
# print(all_specs)

# for i, spec in enumerate(all_specs):
#     method_configs[f"anoynmous_method_{i}"] = copy.deepcopy(base_method)
#     for k, v in spec.items():
#         set_functions[k](method_configs[f"anoynmous_method_{i}"], v)
#     # print(method_configs[f"anoynmous_method_{i}"])
#     task_file = Path("/data/czl/nerf/MSTH_new/MSTH/scripts/task_4_19.txt")
#     with open("/data/czl/nerf/MSTH_new/MSTH/scripts/task_4_19.txt", "a") as f:
#         f.write(f"anoynmous_method_{i}\n")

for key in method_configs.keys():
    method_configs[key].wandb_name = key

external_methods, external_descriptions = discover_methods()
print(external_methods)
method_configs.update(external_methods)
descriptions.update(external_descriptions)

AnnotatedBaseConfigUnion = tyro.conf.SuppressFixed[  # Don't show unparseable (fixed) arguments in helptext.
    tyro.conf.FlagConversionOff[
        tyro.extras.subcommand_type_from_defaults(defaults=method_configs, descriptions=descriptions)
    ]
]
"""Union[] type over config types, annotated with default instances for use with
tyro.cli(). Allows the user to pick between one of several base configurations, and
then override values in it."""
