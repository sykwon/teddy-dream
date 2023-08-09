# (SODDY, TEDDY) & DREAM

[![license](https://img.shields.io/github/license/sykwon/teddy-dream?color=brightgreen)](https://github.com/sykwon/teddy-dream/blob/master/LICENSE)
[![ubuntu](https://img.shields.io/badge/ubuntu-v18.04-orange)](https://wiki.ubuntu.com/Releases)
![gcc 7.5](https://img.shields.io/badge/gcc-v7.5-blue)
![python 3.7](https://img.shields.io/badge/python-v3.7-blue)
![cuda 11.6](https://img.shields.io/badge/cuda-v11.6-blue)
![size](https://img.shields.io/github/repo-size/sykwon/teddy-dream?color=yellow)
![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fsykwon%2Fteddy-dream&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)

This repository implements training data generation algorithms (SODDY & TEDDY) and deep cardinality estimators (DREAM) proposed in our paper "Cardinality Estimation of Approximate Substring Queries using Deep Learning". It is created by Suyong Kwon, Woohwan Jung and Kyuseok Shim.

## Repository Overview

It consists of four folders each of which contains its own README file and script.

|Folder| Description |
|---|---|
| gen_train_data | training data generation algorithms |
| dream  | deep cardinality estimators for approximate substring queries |
| astrid | the modified version of Astrid starting from the astrid model downloaded from [[github](<https://github.com/saravanan-thirumuruganathan/astrid-string-selectivity>)]|
| plot | example notebook files |

## Installation and Requirements
It is recommended to run our code with the CUDA environment.
However, the non-CUDA version of our code is also working when the pytorch library does not supper GPU. (You may set CUDA_VISIBLE_DEVICES as -1 to enforce CPU mode.)

### Method 1: Use the Docker Image
To run the image needs the NVIDIA Container Toolkit. If you do not have the toolkit, refer to the [installation guide](<https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker>)

```bash
git clone https://github.com/sykwon/teddy-dream.git

# run docker image
docker run -it --gpus all --name dream -v ${PWD}:/workspace -u 1000:1000 sykwon/dream /bin/bash

# after starting docker
redis-server --daemonize yes
cd gen_train_data/
make clean && make && make info
cd ..
```

### Method 2: Create a Virtual Python Environment
This code needs Python-3.7 or higher.

```bash
sudo apt-get install -y redis-server git
sudo apt-get install -y binutils
sudo apt-get install -y texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super

conda create -n py37 python=3.7
source activate py37
conda install -y pytorch=1.7.1 torchvision=0.8.2 cudatoolkit=11.0 -c pytorch -c nvidia

pip install -r requirements.txt
```

### Datasets

* DBLP
* GENE
* WIKI
* IMDB

## Examples

These commands produces experimental results.

```bash
cd gen_train_data
./run.sh DBLP     # to generate training data from the DBLP dataset
# ./run.sh GENE   # to generate training data from the GENE dataset
# ./run.sh WIKI   # to generate training data from the WIKI dataset
# ./run.sh IMDB   # to generate training data from the IMDB dataset
# ./run.sh all    # to generate training data from all datasets
cd ..

cd dream
./run.sh DBLP    # to train all models except Astrid with the DBLP dataset
# ./run.sh GENE  # to train all models except Astrid with the GENE dataset
# ./run.sh WIKI  # to train all models except Astrid with the WIKI dataset
# ./run.sh IMDB  # to train all models except Astrid with the IMDB dataset
# ./run.sh all   # to train all models except Astrid with all datasets
cd ..

cd astrid
./run.sh DBLP    # to train the Astrid model with the DBLP dataset
# ./run.sh GENE  # to train the Astrid model with the GENE dataset
# ./run.sh WIKI  # to train the Astrid model with the WIKI dataset
# ./run.sh IMDB  # to train the Astrid model with the IMDB dataset
# ./run.sh all   # to train the Astrid model with all datasets
cd ..
```

Please refer to [[notebook](/plot/example.ipynb)] to see the experimental results.

## Citation

Please consider to cite our paper if you find this code useful:

```bibtex
@article{kwon2022cardinality,
    title={Cardinality estimation of approximate substring queries using deep learning},
    author={Kwon, Suyong and Jung, Woohwan and Shim, Kyuseok},
    journal={Proceddings of the VLDB Endowment},
    volume={15},
    number={11},
    year={2022}
}
```
