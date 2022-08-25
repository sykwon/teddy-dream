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

## Training data generation

To run an algorithm to generate a training data, run the following command:

```bash
python for.py -s <number_of_repetitions> -d <data_name> -a <algorithm_name> -pr <prefix_aug_flag> -t <delta_M> -hr <max_hours_to_execute>
```

The meaning of each parameter value in the above is as follows.
* <number_of_repetitions>: the number of repeted executions of generating a dataset  
  This parameter value is used to compute average execution time for experiments.
* <data_name>: the name of dataset (DBLP, GENE, WIKI or IMDB)  
The meanings of DBLP, GENE, WIKI and IMDB are described in Section 6 of our paper.
* <algorithm_name>: the name of the training data generation algorithm (NaiveGen, Qgram, TASTE, SODDY, TEDDY, TEDDY-S or TEDDY-R)  
The meanings of NaiveGen, Qgram, TASTE, SODDY, TEDDY, TEDDY-S and TEDDY-R are described in Section 6 of our paper.
* <prefix_aug_flag>: the flag to represent whether to generate the prefix-aug training data (0: base training data; 1: prefix-aug training data)
* <delta_M>: the maximum substring edit distance threshold
* <max_hours_to_execute>: the time to be allowed for generating training data (1 means an hour.)  
If the algorithm does not finish within ```<max_hours_to_execute>```, we stop and generate nothing.

## Example Usage

```bash
./run.sh DBLP     # to generate training data from the DBLP dataset
# ./run.sh WIKI   # to generate training data from the WIKI dataset
# ./run.sh IMDB   # to generate training data from the IMDB dataset
# ./run.sh GENE   # to generate training data from the GENE dataset
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
