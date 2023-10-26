import torch
import torch.nn as nn
from feature_extractor import ResUNet

def test_load_from_ckpt():
    u = ResUNet.load_from_pretrained("model_255000.pth")
    print(u)

def test_shape():
    # torch.set_default_tensor_type(torch.cuda.FloatTensor)
    u = ResUNet.load_from_pretrained("model_255000.pth")
    inputs = torch.randn(1, 2704, 2028, 3)
    outputs = u(inputs)
    print(outputs[0].shape)
    print(outputs[1].shape)

if __name__ == "__main__":
    test_shape()