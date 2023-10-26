from __future__ import annotations
from copy import deepcopy
from typing import Dict, Optional
from typing import *
import copy
import tyro
from MSTH.directModel.deferred_nerfacto_model import DeferredNerfactoModel, DeferredNerfactoModelConfig
from nerfacc import ContractionType

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.datamanagers.depth_datamanager import DepthDataManagerConfig
from nerfstudio.data.datamanagers.semantic_datamanager import SemanticDataManagerConfig
from nerfstudio.data.datamanagers.variable_res_datamanager import (
    VariableResDataManagerConfig,
)
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.dataparsers.dnerf_dataparser import DNeRFDataParserConfig
from nerfstudio.data.dataparsers.dycheck_dataparser import DycheckDataParserConfig
from nerfstudio.data.dataparsers.instant_ngp_dataparser import (
    InstantNGPDataParserConfig,
)
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.dataparsers.phototourism_dataparser import (
    PhototourismDataParserConfig,
)

from dataclasses import dataclass, field

# from nerfstudio.data.dataparsers.sitcoms3d_dataparser import Sitcoms3DDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from MSTH.cos_scheduler import CosineDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
from nerfstudio.models.depth_nerfacto import DepthNerfactoModelConfig
from nerfstudio.models.instant_ngp import InstantNGPModelConfig
from nerfstudio.models.mipnerf import MipNerfModel
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.models.nerfplayer_nerfacto import NerfplayerNerfactoModelConfig
from nerfstudio.models.nerfplayer_ngp import NerfplayerNGPModelConfig
from nerfstudio.models.semantic_nerfw import SemanticNerfWModelConfig
from nerfstudio.models.tensorf import TensoRFModelConfig
from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipelineConfig
from nerfstudio.plugins.registry import discover_methods
from MSTH.streamable_model import StreamableNerfactoModelConfig
from MSTH.directModel.trainer import VideoTrainerConfig
from MSTH.datamanager import VideoDataManagerConfig
from MSTH.dataparser import VideoDataParserConfig
from MSTH.video_pipeline import VideoPipelineConfig

from MSTH.datamanager import VideoFeatureDataManagerConfig
from MSTH.SpaceTimeHashing.trainer import SpaceTimeHashingTrainerConfig
from MSTH.datamanager import SpaceTimeDataManagerConfig
from MSTH.video_pipeline import SpaceTimePipelineConfig
from MSTH.SpaceTimeHashing.model import SpaceTimeHashingModelConfig
from MSTH.SpaceTimeHashing.stmodel import DSpaceTimeHashingModelConfig
from pathlib import Path
import numpy as np
