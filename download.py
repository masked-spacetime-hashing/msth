import argparse
import zipfile
from pathlib import Path

import wget

n3dv_scenes = ["sear_steak", "cut_roasted_beef", "cook_spinach", "coffee_martini", "flame_salmon"]
immersive_scenes = []
dnerf_scenes = []
campus_scenes = []

tmp_dir = Path("./.msth_tmp")
tmp_dir.mkdir(exist_ok=True)


def download_n3dv(scene: str, download_dir: Path):
    assert scene in n3dv_scenes, f"Scene {scene} not found in N3DV dataset"
    data_url = f"https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/{scene}.zip"
    wget.download(data_url, out=str(download_dir))
    zipfile.ZipFile(download_dir / f"{scene}.zip").extractall(download_dir)


def download_immersive():
    pass


def download_dnerf():
    pass


def download_campus():
    pass


download_fns = {
    "n3dv": download_n3dv,
    "immersive": download_immersive,
    "dnerf": download_dnerf,
    "campus": download_campus,
}


def main():
    parser = argparse.ArgumentParser("Download datasets")
    parser.add_argument("dataset", type=str, default="n3dv", choices=["n3dv", "immersive", "dnerf", "campus"])
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--download-dir", type=Path, default=Path("data"))

    opt = parser.parse_args()
    download_fns[opt.dataset](opt.scene)
