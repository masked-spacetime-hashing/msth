import argparse
import os
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_fps", type=int, default=30)
    parser.add_argument("--input_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    for filename in tqdm(os.listdir(args.input_path)):
        if filename.endswith("mp4") or filename.endswith("MP4"):
            fp = os.path.join(args.input_path, filename)
            fpo = os.path.join(args.output_path, filename)
            cmd = f"ffmpeg -i {fp} -r {args.target_fps} {fpo}"
            os.system(cmd)
