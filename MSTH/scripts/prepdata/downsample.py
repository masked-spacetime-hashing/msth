import cv2
import argparse
import os
from tqdm import tqdm
from pathlib import Path

input_dir = "flame_salmon_videos"
output_dir = "flame_salmon_videos_2"

parser = argparse.ArgumentParser()

parser.add_argument("--height", type=int, default=720)
parser.add_argument("-w", type=int, default=960)
parser.add_argument("-i", type=str, default="flame_salmon_videos")
parser.add_argument("-f", type=int, default=30)
parser.add_argument("-o", type=str, default="flame_salmon_videos_2")
parser.add_argument("-t", type=str, default="area")

args = parser.parse_args()

interps = {
    "area": cv2.INTER_AREA,
    "cubic": cv2.INTER_CUBIC,
    "linear": cv2.INTER_LINEAR,
}

opth = Path(args.o)
if not opth.exists():
    opth.mkdir(parents=True)
input_dir = args.i
output_dir = args.o

# 遍历输入文件夹中的所有视频文件
for filename in tqdm(os.listdir(input_dir)):
    if filename.endswith(".mp4") or filename.endswith(".avi") or filename.endswith(".MP4"):
        # 构造输入输出文件路径
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        # 打开输入视频文件
        cap = cv2.VideoCapture(input_path)
        # 获取视频帧率、宽度和高度
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 构造输出视频写入器
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 可以根据需要修改编码器
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (int(args.w), int(args.height)),
        )
        # 逐帧读取、下采样并写入输出视频
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # 下采样
                frame = cv2.resize(
                    frame,
                    dsize=(args.w, args.height),
                    interpolation=interps[args.t],
                )
                # 写入输出视频
                out.write(frame)
            else:
                break
        # 释放资源
        cap.release()
        out.release()
