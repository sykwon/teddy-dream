# TEDDY & DREAM

[![license](https://img.shields.io/github/license/sykwon/teddy-dream?color=brightgreen)](https://github.com/sykwon/teddy-dream/blob/master/LICENSE)
[![ubuntu](https://img.shields.io/badge/ubuntu-v18.04-orange)](https://wiki.ubuntu.com/Releases)
![gcc 7.5](https://img.shields.io/badge/gcc-v7.5-blue)
![python 3.7](https://img.shields.io/badge/python-v3.7-blue)
![cuda 11.6](https://img.shields.io/badge/cuda-v11.6-blue)
![size](https://img.shields.io/github/repo-size/sykwon/teddy-dream?color=yellow)
![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fsykwon%2Fteddy-dream&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)

This repository implements training data generation algorithms (SODDY & TEDDY) and deep cardinality estimators (DREAM) proposed in our paper "Cardinality Estimation of Approximate Substring Queries using Deep Learning".

## Repository Overwiew

It consists of four folders each of which contains its own README file and script.

|Folder| Description |
|---|---|
| gen_train_data | training data generation algorithms                                              |
| dream  | deepp cardinality estimators for approximate substring queries                         |
| astrid | the modified version of Astrid starting from the astrid model downloaded from [[github](<https://github.com/saravanan-thirumuruganathan/astrid-string-selectivity>)]|
| plot | example notebook files |

## Installation and Requirements

```bash
sudo apt-get install -y redis-server
sudo apt-get install -y binutils
sudo apt-get install -y texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super

conda create -n py37 python=3.7
source activate py37
conda install -y pytorch=1.7.1 torchvision=0.8.2 cudatoolkit=11.0 -c pytorch -c nvidia

cd dream
pip install -r requirements.txt
cd ..

cd astrid
pip install -r requirements.txt
cd ..

cd plot
pip install -r requirements.txt
cd ..
```

## Examples

These commands produces experimental results.

```bash
cd teddy
./run.sh dblp
# ./run.sh all # to generate training data from all datasets
cd ..

cd dream
./run.sh dblp
# ./run.sh all # to train all models except astrid with all datasets
cd ..

cd astrid
./run.sh dblp
# ./run.sh all # to train the Astrid model with all datasets
cd ..
```

Please refer to [[notebook](/plot/example.ipynb)] to visualize results.
