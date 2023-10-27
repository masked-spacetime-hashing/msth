# Masked Space-Time Hash Encoding for Efficient Dynamic Scene Reconstruction
[NeurIPS 2023 Spotlight]

[Project Page](https://masked-spacetime-hashing.github.io/) | [Paper](https://arxiv.org/pdf/2310.17527.pdf) | [Data](https://huggingface.co/datasets/masked-spacetime-hashing/Campus)

[Feng Wang]()<sup>1*</sup>, [Zilong Chen]()<sup>1*</sup>, Guokang Wang<sup>1</sup>, Yafei Song<sup>2</sup>, [Huaping Liu]()<sup>1</sup>

<sup>1</sup>Department of Computer Science and Technology, Tsinghua University <sup>2</sup>Alibaba Group


### Introduction
We propose the Masked Space-Time Hash encoding (MSTH), a novel method for efficiently reconstructing dynamic 3D scenes from multi-view or monocular videos. Based on the observation that dynamic scenes often contain substantial static areas that result in redundancy in storage and computations, MSTH represents a dynamic scene as a weighted combination of a 3D hash encoding and a 4D hash encoding. The weights for the two components are represented by a learnable mask which is guided by an uncertainty-based objective to reflect the spatial and temporal importance of each 3D position. With this design, our method can reduce the hash collision rate by avoiding redundant queries and modifications on static areas, making it feasible to represent a large number of space-time voxels by hash tables with small size. Besides, without the requirements to fit the large numbers of temporally redundant features independently, our method is easier to optimize and converge rapidly with only twenty minutes of training for a 300-frame dynamic scene. We evaluate our method on extensive dynamic scenes. As a result, MSTH obtains consistently better results than previous state-of-the-art methods with only 20 minutes of training time and 130 MB of memory storage.

### Demos
We recommend to visit our [project page](https://masked-spacetime-hashing.github.io/) for watching clear videos.
#### [Immersive Dataset](https://augmentedperception.github.io/deepviewvideo/)

https://github.com/masked-spacetime-hashing/msth/assets/43294876/c14dcb57-c600-43b9-adf1-f8a532785d8f


#### [Plenoptic Dataset](https://neural-3d-video.github.io/)

https://github.com/masked-spacetime-hashing/msth/assets/43294876/7094fee1-3cfb-49f4-abed-dc5f61a7fb72

#### [Campus Dataset](https://huggingface.co/datasets/masked-spacetime-hashing/Campus)

https://github.com/masked-spacetime-hashing/msth/assets/43294876/1fbc7417-e66b-4cdd-8e8c-1863f031fa30


### Instructions
#### Create env
```bash
conda create -n MSTH python=3.8
```
### Install dependencies
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
python -m MSTH.script.train <config-name> --experiment-name <exp-name> --vis <logger> --output-dir <output-dir>
```
#### Viewer
Our code provides a viewer based on the [NeRFStudio web viewer]().

### Campus Dataset

### Acknowledgements
Our code is based on [NeRFStudio](https://github.com/nerfstudio-project/nerfstudio)
