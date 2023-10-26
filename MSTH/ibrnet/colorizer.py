import torch
import torch.nn as nn
import torch.nn.functional as F
from MSTH.projector import Projector
from MSTH.dataset import VideoDatasetWithFeature
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RaySamples

class Colorizer(nn.Module):
    def __init__(self, cameras: Cameras, dataset: VideoDatasetWithFeature, device) -> None:
        super().__init__()
        self.cameras = cameras
        self.dataset = dataset
        self.projector = Projector(cameras)
        
        self.n_feats = dataset.cur_frame_feats_buffer.shape[-1]
        
        self.mlp_base = nn.Sequential(nn.Linear(3 + self.n_feats, 32))
        
        self.mlp_head = nn.Linear(36, 1)

        self.device = device

    def get_images(self):
        return torch.from_numpy(self.dataset.cur_frame_buffer)
    
    def get_features(self):
        # return torch.from_numpy(self.dataset.cur_frame_feats_buffer).to(self.device)
        return self.dataset.cur_frame_feats_buffer
        
    def colorize(self, ray_samples: RaySamples, c2ws):
        positions = ray_samples.frustums.get_positions()
        ray_shape = positions.size()
        positions = positions.reshape(-1, 3)
        pixels, feats, masks = self.projector(positions, self.get_images(), self.get_features())
        
        # pixels: [b, ncams, 3]
        
        # print(feats.shape)
        # feats_mean = torch.mean(feats, dim=1, keepdim=True).repeat(1, feats.size(1), 1)
        # feats_var = torch.var(feats, dim=1, keepdim=True).repeat(1, feats.size(1), 1)
        # feats = torch.cat([pixels, feats, feats_mean, feats_var], dim=-1)
        pixels = pixels.to(self.device)
        feats = feats.to(self.device)
        masks = masks.to(self.device)
        feats = torch.cat([pixels, feats], dim=-1)

        cam_shap = feats.size()
        # feats = feats.reshape(-1, feats.size(-1))

        camera_indices = ray_samples.camera_indices.squeeze()
        
        dir_diff = self.projector.get_relative_directions_with_positions(positions, camera_indices, c2ws)
        
        latent = self.mlp_base(feats)
        
        latent = torch.cat([latent, dir_diff], dim=-1)

        weights = self.mlp_head(latent).squeeze()
        # [b, ncams]
        weights[~masks] = -100.

        # weights = weights.resize(*cam_shap[:-1], -1)
        # [b, 18]
        
        weights = F.softmax(weights, dim=-1).unsqueeze(-1)
        
        colors = torch.sum(weights * pixels, dim=1)

        return colors.reshape(ray_shape)

    def forward(self, ray_samples, c2ws):
        return self.colorize(ray_samples, c2ws)