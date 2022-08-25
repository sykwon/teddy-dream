# Traning Data Generation

[![license](https://img.shields.io/github/license/sykwon/teddy-dream?color=brightgreen)](https://github.com/sykwon/teddy-dream/blob/master/LICENSE)
[![ubuntu](https://img.shields.io/badge/ubuntu-v18.04-orange)](https://wiki.ubuntu.com/Releases)
![gcc 7.5](https://img.shields.io/badge/gcc-v7.5-blue)
![python 3.7](https://img.shields.io/badge/python-v3.7-blue)

## Installation

This code needs Python-3.7 or higher.

```bash
sudo apt-get install redis-server
sudo apt-get install binutils
pip install redis
```

## Example Usage

```bash
./run.sh dblp     # to generate training data from the DBLP dataset
# ./run.sh wiki2  # to generate training data from the WIKI dataset
# ./run.sh imdb2  # to generate training data from the IMDB dataset
# ./run.sh egr1   # to generate training data from the GENE dataset
# ./run.sh all    # to generate training data from all datasets
```

## Descriptions

### Algorithms

| Algorithm |
|-----------|
| NaiveGen  |
| Qgram     |
| TASTE     |
| SODDY     |
| TEDDY     |
| TEDDY-S   |
| TEDDY-R   |

### Datasets

| Dataset |
|---------|
| DBLP    |
| WIKI    |
| IMDB    |
| GENE    |
