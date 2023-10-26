import torch
import torch.nn as nn
from stgrid import SpatialTemporalGridEncoder as stencoder
from grid import GridEncoder as encoder

pls = (1.2, 1.2, 1.2, 1.2)
bs = (16, 16, 16, 16)
stm = stencoder(input_dim=4, per_level_scale=pls, base_resolution=bs, std=1).cuda()
s = encoder(per_level_scale=pls[:3], base_resolution=bs[:3]).cuda()
t = encoder(
    input_dim=4,
    per_level_scale=pls,
    base_resolution=bs,
).cuda()
m = encoder(
    input_dim=3,
    num_levels=1,
    level_dim=1,
    per_level_scale=(1, 1, 1),
    base_resolution=(128, 128, 128),
    gridtype="tiled",
    log2_hashmap_size=21,
    interpolation="all_nearest",
).cuda()

print(m.embeddings.shape)
print(stm.membeddings.shape)

s.embeddings.data.copy_(stm.sembeddings.data)
t.embeddings.data.copy_(stm.tembeddings.data)
m.embeddings.data[..., 0].copy_(stm.membeddings.data)


class STMTorch(nn.Module):
    def __init__(self, sencoder, tencoder, mencoder):
        super().__init__()
        self.s = sencoder
        self.t = tencoder
        self.m = mencoder

    def forward(self, x):
        s = self.s(x[:, :3])
        t = self.t(x)
        m = self.m(x[:, :3]).sigmoid()
        print(t)
        print(m)
        return s + t * (1 - m)


stmtorch = STMTorch(s, t, m).cuda()

x = torch.rand(5, 4).cuda()
print(x)
# x = x.abs()
# x = x / x.max()[0]

y1 = stmtorch(x)
y2 = stm(x)
print(stm.toffsets)
print(stmtorch.t.offsets)
print(stm.soffsets)
print(stmtorch.s.offsets)
# print(y1)
# print(y2)
# print((y1).abs().mean())
# print((y1 - y2).abs().mean())
