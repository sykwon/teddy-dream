# Traning Data Generation

[![license](https://img.shields.io/github/license/sykwon/teddy-dream?color=brightgreen)](https://github.com/sykwon/teddy-dream/blob/master/LICENSE)
[![ubuntu](https://img.shields.io/badge/ubuntu-v18.04-orange)](https://wiki.ubuntu.com/Releases)
![gcc 7.5](https://img.shields.io/badge/gcc-v7.5-blue)
![python 3.7](https://img.shields.io/badge/python-v3.7-blue)

## Installation

```bash
sudo apt-get install redis-server
sudo apt-get install binutils
pip install redis
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

## Descriptions

### Algorithms

| in exp | in paper |
|--------|----------|
| allp   | NavieGen |
| topk   | Qgram    |
| taste  | TASTE    |
| soddy2 | SODDY    |
| teddy2 | TEDDY    |
| teddy0 | TEDDY-S  |
| abl1   | TEDDY-R  |

### Datasets

| in exp | in paper |
|--------|----------|
| dblp   |   DBLP   |
| wiki2  |   WIKI   |
| imdb2  |   IMDB   |
| egr1   |   GENE   |
