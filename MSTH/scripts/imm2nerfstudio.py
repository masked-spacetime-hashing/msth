import argparse
import json
from copy import deepcopy
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def imm2nerfstudio(filename="models.json", video_path="", video_suffix=".mp4", num_frames=-1):
    imm = json.load(open(filename))
    nfs = {}
    nfs_val = {}
    initial_cam = imm[0]
    nfs["w"] = initial_cam["width"]
    nfs["h"] = initial_cam["height"]
    if initial_cam["projection_type"] == "fisheye":
        nfs["camera_model"] = "OPENCV_FISHEYE"
        nfs_val["camera_model"] = "OPENCV_FISHEYE"
    else:
        raise NotImplementedError
    nfs["frames"] = []
    nfs_val["w"] = nfs["w"]
    nfs_val["h"] = nfs["h"]
    nfs_val["frames"] = []
    for camera_setting in imm:
        new_cam = {}
        new_cam["file_path"] = camera_setting["name"] + video_suffix
        if len(video_path) > 1:
            new_cam["file_path"] = video_path + "/" + new_cam["file_path"]
        R = Rotation.from_rotvec(camera_setting["orientation"]).as_matrix()
        T = np.array(camera_setting["position"])
        pose = np.eye(4)
        pose[:3, :3] = R.T
        pose[:3, -1] = T
        pose_pre = np.eye(4)
        pose_pre[1, 1] *= -1
        pose_pre[2, 2] *= -1
        pose = pose_pre @ pose @ pose_pre
        k1 = camera_setting["radial_distortion"][0]
        k2 = camera_setting["radial_distortion"][1]
        k3 = 0
        k4 = 0
        fl_x = camera_setting["focal_length"]
        fl_y = fl_x
        cx = camera_setting["principal_point"][0]
        cy = camera_setting["principal_point"][1]
        new_cam["transform_matrix"] = pose.tolist()
        new_cam["k1"] = k1
        new_cam["k2"] = k2
        new_cam["k3"] = k3
        new_cam["k4"] = k4
        new_cam["fl_x"] = fl_x
        new_cam["fl_y"] = fl_y
        new_cam["cx"] = cx
        new_cam["cy"] = cy
        if camera_setting["name"] != "camera_0001":
            nfs["frames"].append(new_cam)
        else:
            nfs_val["frames"].append(new_cam)

    example_video = nfs["frames"][0]["file_path"]
    vc = cv2.VideoCapture(example_video)
    num_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT)) if num_frames < 0 else num_frames
    print(f"num frames: {num_frames}")
    nfs["num_frames"] = num_frames
    assert len(nfs_val["frames"]) == 1

    with open("transforms_train.json", "w") as f:
        json.dump(nfs, f, indent=4)
    with open("transforms_val.json", "w") as f:
        json.dump(nfs_val, f, indent=4)
    with open("transforms_test.json", "w") as f:
        json.dump(nfs_val, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, default=".")
    parser.add_argument("--suffix", type=str, default=".mp4")
    parser.add_argument("--video_path", type=str, default="")
    parser.add_argument("--num_frames", type=int, default=-1)

    opt = parser.parse_args()

    imm2nerfstudio(opt.filename, opt.video_path, opt.suffix)
