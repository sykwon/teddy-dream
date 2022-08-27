# Cardinality Estimation of Approximate Substring Queries

[![license](https://img.shields.io/github/license/sykwon/teddy-dream?color=brightgreen)](https://github.com/sykwon/teddy-dream/blob/master/LICENSE)
[![ubuntu](https://img.shields.io/badge/ubuntu-v18.04-orange)](https://wiki.ubuntu.com/Releases)
![python 3.7](https://img.shields.io/badge/python-v3.7-blue)
![cuda 11.6](https://img.shields.io/badge/cuda-v11.6-blue)

## Installation

```bash
sudo apt-get install redis-server
conda create -n py37 python=3.7
source activate py37
conda install pytorch=1.7.1 torchvision=0.8.2 cudatoolkit=11.0 -c pytorch -c nvidia
pip install -r requirements.txt # For python packages, see requirements.txt
```

## Training & evaluating cardinality estimators

To train and evaluate the cardinality estimators, run the following command:

```bash
python run.py
```

## Script

```bash
# quick test
./run.sh test 

# exp on DBLP 
./run.sh dblp

# all exp
./run.sh all
```

### Algorithms

| in exp | in paper |
|--------|----------|
| eqt    | LBS      |
| card   | CardNet  |
| rnn    | DREAM    |

### Datasets

| in exp | in paper |
|--------|----------|
| dblp   |   DBLP   |
| wiki2  |   WIKI   |
| imdb2  |   IMDB   |
| egr1   |   GENE   |
