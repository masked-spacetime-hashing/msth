import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType

from MSTH.utils import Timer
from nerfstudio.cameras.cameras import Cameras


def generate_neighbors(cams: Cameras):
    c2ws = cams.camera_to_worlds
    c2ws = torch.cat([c2ws, torch.FloatTensor([0, 0, 0, 1]).to(c2ws).reshape(1, 1, 4).expand(c2ws.shape[0], -1, -1)], dim=1)
    c2w = c2ws[0]
    print(torch.inverse(c2w))

    
def fetch_corresponding_pixels_from_cameras(positions, cams: Cameras):
    pass
    
def fetch_corresponding_pixels_from_adjacent_cameras(positions, cams: Cameras):
    pass

def fetch_corresponding_patches_from_adjacent_cameras(positions, cams: Cameras):
    pass

import torch


def extract_patch(image, center_x, center_y, patch_width, patch_height):
    # Calculate patch coordinates
    left = max(center_x - patch_width // 2, 0)
    right = min(center_x + patch_width // 2, image.shape[0])
    top = max(center_y - patch_height // 2, 0)
    bottom = min(center_y + patch_height // 2, image.shape[1])

    # Extract patch from image tensor
    patch = image[left:right, top:bottom, :]

    # Create a mask tensor for the patch
    mask = torch.zeros((patch_width, patch_height, image.shape[2]))
    patch_left = max(patch_width // 2 - center_x, 0)
    patch_top = max(patch_height // 2 - center_y, 0)
    patch_right = min(patch_width // 2 + image.shape[0] - center_x, patch_width)
    patch_bottom = min(patch_height // 2 + image.shape[1] - center_y, patch_height)
    try:
        mask[patch_left:patch_right, patch_top:patch_bottom, :] = patch
    except RuntimeError:
        pass
    # mask[patch_left:patch_right, patch_top:patch_bottom, :] = patch

    return mask

def extract_patch_batched_cxy(image, cxs, cys, w, h):
    patches = []
    for cx, cy in zip(cxs, cys):
        patches.append(extract_patch(image, cx.item(), cy.item(), w, h))

    return torch.stack(patches, dim=0) #[b, h, w, 3]

def extract_patch_batched_cameras(images, cxs, cys, w, h):
    patches = []
    num_images = images.shape[0]
    for i in range(num_images):
        patches.append(extract_patch_batched_cxy(images[i], cxs[i], cys[i], w, h))

    return torch.stack(patches, dim=1) # [b, c, h, w, 3]

class Projector(nn.Module):
    def __init__(self, cameras: Cameras) -> None:
        super().__init__()
        self.cameras = cameras
        self.c2ws = cameras.camera_to_worlds
        self.n_cameras = self.c2ws.size(0)
        self.c2ws = torch.cat([self.c2ws, torch.FloatTensor([0, 0, 0, 1]).to(self.c2ws).reshape(1, 1, 4).expand(self.c2ws.shape[0], -1, -1)], dim=1)
        self.register_buffer("w2cs", torch.inverse(self.c2ws))
        self._build_intrinsic_matrix(cameras)
        self._build_directions()
        # self.register_buffer("w2cs", torch.inverse(self.c2ws)[..., :3, :])
        
    def _build_intrinsic_matrix(self, cameras: Cameras) -> None:
        num_cams = cameras.camera_to_worlds.size(0)
        self.register_buffer("intrinsic", torch.zeros([num_cams, 3, 4]))
        
        self.intrinsic[..., 0, 0] = -cameras.fx[..., 0]
        self.intrinsic[..., 1, 1] = cameras.fy[..., 0]
        self.intrinsic[..., 2, 2] = 1.0
        self.intrinsic[..., 0, 2] = cameras.cx[..., 0]
        self.intrinsic[..., 1, 2] = cameras.cy[..., 0]
        
        # hw = 
        self.w = self.cameras.width[0].item()
        self.h = self.cameras.height[0].item()
        M = torch.einsum("cij,cjk->cik", self.intrinsic, self.w2cs)
        self.register_buffer("M", M)
        
    def _build_directions(self):
        print(self.c2ws.shape)
        directions = self.c2ws[..., :3, 3].unsqueeze(0)
        self.register_buffer("directions", directions)
        self.directions = F.normalize(self.directions, p=2, dim=-1)
        print(self.directions.size())
        assert self.directions.size() == torch.Size([1, self.n_cameras, 3])
        

    def positions_world_to_imaging_plane(self, positions: TensorType["batch_size", 3]):
        # NOTE: positions here should be inversed from contraction or ndc
        homo = torch.cat([positions, torch.ones(positions.size(0), 1).to(positions)], dim=1)
        
        # TODO: visibility with z ? 
        image_coord = torch.einsum("cij,bj->cbi", self.M, homo)

        image_coord /= image_coord[..., -1].unsqueeze(-1)
        
        image_coord = image_coord.to(torch.long)
        
        masks = torch.logical_and(torch.logical_and(image_coord[..., 0] >= 0, image_coord[..., 1] >= 0), torch.logical_and(image_coord[..., 1] < self.w, image_coord[..., 0] < self.h))
        
        return image_coord[..., :2], masks
    
    def fetch_corresponding_pixels(self, positions, images, features):
        # return shape [b, cams, channel]
        # images: [cams, h, w, channel]
        # image_coords: [cams, b, 2]
        # masks: [cams, b]
        
        image_coords, masks = self.positions_world_to_imaging_plane(positions)
        positions = positions.to(images)
        image_coords = image_coords.to(torch.long)
        image_coords[~masks] = 0
        
        num_cams = image_coords.size(0)
        fetched = []
        feats = []
        for i in range(num_cams):
            fetched.append(images[i][image_coords[i, :, 0], image_coords[i, :, 1], :])
            feats.append(features[i][image_coords[i, :, 0], image_coords[i, :, 1], :])

        fetched = torch.stack(fetched, dim=1)
        feats = torch.stack(feats, dim=1)
        
        masks = masks.transpose(0, 1)
        # fetched = images[image_coords, :]
        # fetched: [b, cams, channel]
        # masks: [b, cams]
        return fetched, feats, masks

    def forward(self, positions, images, features):
        return self.fetch_corresponding_pixels(positions, images, features)

    
    def fetch_corresponding_patches(self, positions, images, patch_w, patch_h):
        with Timer("fetching patches"):
            image_coords, _ = self.positions_world_to_imaging_plane(positions)
            # image_coords: [c, b, 2]
            # masks: [b, c]
            # images: [c, h, w, 3]
            image_coords = image_coords.to(images).long()
            
            # [b, c, h, w, 3]
            return extract_patch_batched_cameras(images, image_coords[..., 0], image_coords[..., 1], patch_w, patch_h)

    def get_relative_directions(self, dirs):
        # dirs: [b, 3]
        # self.directions: [1, cams, 3]
        dirs = F.normalize(dirs, p=2, dim=-1)
        dirs = dirs.unsqueeze(0).repeat(self.n_cameras, 1, 1) #[b, cams, 3]
        dir_diffs = F.normalize(dir - self.directions, p=2, dim=-1)
        dir_dots = torch.sum(dirs * self.directions, keepdim=True, dim=-1)
        return torch.cat([dir_diffs, dir_dots], dim=-1)
    
    
    def get_relative_directions_with_positions(self, positions, ray_indices, c2ws):
        # positions: [b, 3]
        # ray_indices: [b]
        # c2ws: [nc, 3, 4]
        
        # print(c2ws.shape)
        # print(positions.shape)
        ray_indices = ray_indices.reshape(-1)
        # print(c2ws[ray_indices].shape)
        # print(c2ws[ray_indices.squeeze()][..., 3].unsqueeze(1).shape)
        camera_pose = c2ws[ray_indices.squeeze()][..., 3].unsqueeze(1).repeat(1, self.n_cameras, 1).to(positions) - positions.unsqueeze(1) # [b, ncams_self, 3]
        camera_pose = F.normalize(camera_pose, p=2, dim=-1)

        ref_dirs = self.directions.repeat(positions.size(0), 1, 1) - positions.unsqueeze(1)
        ref_dirs = F.normalize(ref_dirs, dim=-1)
        
        dir_diff = camera_pose - ref_dirs
        
        dir_dot = torch.sum(camera_pose * ref_dirs, dim=-1, keepdim=True) 
        
        # returned shape [b, ncams_self, 4]

        return torch.cat([dir_diff, dir_dot], dim=-1)
        