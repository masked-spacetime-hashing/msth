import cv2
import sys
import numpy as np
import torch

if len(sys.argv) < 3:
    print("[Usage]: python frame2video.py [input .pt filename] [output filename] ([fps])")

video_data = torch.load(f"{sys.argv[1]}", map_location="cpu")

assert video_data.dtype is torch.uint8

video_data = video_data.numpy().astype(np.uint8)[..., [2, 1, 0]]

print(video_data.shape)

assert video_data.dtype == np.uint8

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30.0 if len(sys.argv) < 4 else float(sys.argv[3])

out = cv2.VideoWriter(f"{sys.argv[2]}.mp4", fourcc, fps, (video_data.shape[2], video_data.shape[1]))
for frame in video_data:
    out.write(frame)

out.release()