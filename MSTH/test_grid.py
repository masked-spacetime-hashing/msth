from rich.console import Console
from gridencoder.grid import _backend
import numpy as np
import torch
import copy
import torch.nn as nn
import time
from utils import Timer
CONSOLE = Console()

grid_encode_forward = _backend.grid_encode_forward
grid_encode_hash_reinitialize = _backend.grid_encode_hash_reinitialize
grid_encode_set_static = _backend.grid_encode_set_static

log2_hashmap_size = 19
max_params = 2 ** log2_hashmap_size
num_levels = 16
base_resolution = 16
per_level_scale = 2
level_dim = 2
align_corners = False
input_dim = 3

offsets = []
offset = 0
for i in range(num_levels):
    resolution = int(np.ceil(base_resolution * per_level_scale ** i))
    params_in_level = min(max_params, (resolution if align_corners else resolution + 1) ** input_dim) # limit max number
    params_in_level = int(np.ceil(params_in_level / 8) * 8) # make divisible
    offsets.append(offset)
    offset += params_in_level
offsets.append(offset)
offsets = torch.from_numpy(np.array(offsets, dtype=np.int32)).cuda()

inputs = torch.rand(4096*48, 3).cuda()
embeddings = nn.Parameter(torch.empty(offset, level_dim)).cuda()
std = 1e-4
embeddings.data.uniform_(-std, std)

B, D = inputs.shape # batch size, coord dim
L = offsets.shape[0] - 1 # level
C = embeddings.shape[1] # embedding dim for each level
S = np.log2(per_level_scale) # resolution multiplier at each level, apply log2 for later CUDA exp2f
H = base_resolution # base resolution
outputs = torch.zeros(L, B, C, device=inputs.device, dtype=embeddings.dtype).cuda()

CONSOLE.log("start forwarding")
print(outputs.sum())
grid_encode_forward(inputs, embeddings, offsets, outputs, B, D, C, L, S, H, None, 0, align_corners, 0)
CONSOLE.log("end forwarding")
print(outputs.sum())

CONSOLE.log("start forwarding")
inputs = torch.rand(1, 3).cuda()
old_embeddings = torch.clone(embeddings)
s = time.time()
inputs[0][0] = 0.7321
inputs[0][1] = 0.8612
inputs[0][2] = 0.7708
B = 1
D = 3
C = 2
L = 1
S = 0
H = 512
align_corners = False
offsets = [0, 513**3+1]
offsets = torch.from_numpy(np.array(offsets, dtype=np.int32)).cuda()
# 0.7321, 0.8612, 0.7708
# grid_encode_hash_reinitialize(inputs, embeddings.to(torch.float16), offsets, B, D, C, L, S, H, 0, align_corners, 0, std)
grid_mask = torch.ones(offsets[-1]).float().cuda()
for i in range(100):
    with Timer(des="reinit"):
        grid_encode_hash_reinitialize(inputs, embeddings.to(torch.float16), offsets, B, D, C, L, S, H, 0, align_corners, 0, std, grid_mask.bool())
#grid_encode_set_static(inputs, grid_mask.to(torch.float16), offsets, B, D, C, L, S, H, 0, align_corners)
print((grid_mask==0).sum())
grid_encode_set_static(inputs, grid_mask, offsets, B, D, C, L, S, H, 0, align_corners)
print((grid_mask==0).sum())
print(time.time()-s)

