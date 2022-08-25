# (SODDY, TEDDY) & DREAM

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

This code needs Python-3.7 or higher.

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

### Datasets

| in exp | in paper |
|--------|----------|
| dblp   |   DBLP   |
| wiki2  |   WIKI   |
| imdb2  |   IMDB   |
| egr1   |   GENE   |

## Examples

These commands produces experimental results.

```bash
cd gen_train_data
./run.sh dblp     # to generate training data from the DBLP dataset
# ./run.sh wiki2  # to generate training data from the WIKI dataset
# ./run.sh imdb2  # to generate training data from the IMDB dataset
# ./run.sh egr1   # to generate training data from the GENE dataset
# ./run.sh all    # to generate training data from all datasets
cd ..

cd dream
./run.sh dblp    # to train all models except Astrid with the DBLP dataset
# ./run.sh wiki2 # to train all models except Astrid with the WIKI dataset
# ./run.sh imdb2 # to train all models except Astrid with the IMDB dataset
# ./run.sh egr1  # to train all models except Astrid with the GENE dataset
# ./run.sh all   # to train all models except Astrid with all datasets
cd ..

cd astrid
./run.sh dblp    # to train the Astrid model with the DBLP dataset
# ./run.sh wiki2 # to train the Astrid model with the WIKI dataset
# ./run.sh imdb2 # to train the Astrid model with the IMDB dataset
# ./run.sh egr1  # to train the Astrid model with the GENE dataset
# ./run.sh all   # to train the Astrid model with all datasets
cd ..
```

Please refer to [[notebook](/plot/example.ipynb)] to see the experimental results.
