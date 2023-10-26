import torch

pth = "/data/machine/nerfstudio/tmp/base_it40000_base_lrlonger/Spatial_Time_Hashing_With_Base/2023-04-16_202626/nerfstudio_models/step-000039999.ckpt"
a = torch.load(pth)
pipeline = a["pipeline"]
b = {}
for k in pipeline.keys():
    if k.startswith("_model.field") or k.startswith("_model.proposal"):
        b[k] = pipeline[k]

torch.save(b, "test.ckpt")
