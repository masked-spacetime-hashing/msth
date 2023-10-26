# MSTH: Masked Space-Time Hash Encoding for Efficient Dynamic Scene Reconstruction

[Feng Wang]()<sup>1*</sup>, [Zilong Chen]()<sup>1*</sup>, Guokang Wang<sup>1</sup>, Yafei Song<sup>2</sup>, [Huaping Liu]()<sup>1</sup>
<sup>1</sup>Department of Computer Science and Technology, Tsinghua University <sup>2</sup>Alibaba Group
NeurIPS 2023 Splotlight

[Paper]() | [Project Page]() | [Data]() | [Video]()

### Introduction
We propose the Masked Space-Time Hash encoding (MSTH), a novel method for efficiently reconstructing dynamic 3D scenes from multi-view or monocular videos. Based on the observation that dynamic scenes often contain substantial static areas that result in redundancy in storage and computations, MSTH represents a dynamic scene as a weighted combination of a 3D hash encoding and a 4D hash encoding. The weights for the two components are represented by a learnable mask which is guided by an uncertainty-based objective to reflect the spatial and temporal importance of each 3D position. With this design, our method can reduce the hash collision rate by avoiding redundant queries and modifications on static areas, making it feasible to represent a large number of space-time voxels by hash tables with small size. Besides, without the requirements to fit the large numbers of temporally redundant features independently, our method is easier to optimize and converge rapidly with only twenty minutes of training for a 300-frame dynamic scene. We evaluate our method on extensive dynamic scenes. As a result, MSTH obtains consistently better results than previous state-of-the-art methods with only 20 minutes of training time and 130 MB of memory storage.

### Demos
#### [Plenoptic Dataset](https://neural-3d-video.github.io/)
#### Immersive Dataset
#### D-NeRF Dataset
#### Campus Dataset

### Instructions
#### Create env
```bash
conda create -n MSTH python=3.8
```
#### Install dependencies
```bash
pip install -e .
```
and install tiny-cuda-nn for fast feed forward NNs:
```bash
pip install
```
#### Download data
```bash
python download.py <dataset-name> --scene <scene-name>
```
#### Run MSTH
```bash
python train.py <config-name> --experiment-name <exp-name> --vis <logger> --output-dir <output-dir>
```
#### Viewer
Our code provides a viewer based on the [NeRFStudio web viewer]().

### Campus Dataset


### Acknowledgements
- NeRFStudio. Our code is built upon nerfstudio. 
- MixVoxel
- Nerfies