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
python for.py -d <data_name> -a <algorithm_name> -pr <prefix_aug_flag> -pt <ratio_training> -th <delta_M> -nt <number_of_repetitions> -hr <max_hours_to_execute>
```

The meaning of each parameter value in the above is as follows.

* <data_name>: the name of dataset (DBLP, GENE, WIKI or IMDB)  
The meanings of DBLP, GENE, WIKI and IMDB are described in Section 6 of our paper.
* <algorithm_name>: the name of the training data generation algorithm (NaiveGen, Qgram, TASTE, SODDY or TEDDY)  
The meanings of NaiveGen, Qgram, TASTE, SODDY, TEDDY are described in Section 6 of our paper.
* <prefix_aug_flag>: the flag to represent whether to generate the prefix-aug training data (0: base training data; 1: prefix-aug training data)
* <ratio_training>:
* <delta_M>: the maximum substring edit distance threshold
* <number_of_repetitions>: the number of repeted executions of generating a dataset  
  This parameter value is used to compute average execution time for experiments.
* <max_hours_to_execute>: the time to be allowed for generating training data (1 means an hour.)  
If the algorithm does not finish within ```<max_hours_to_execute>```, we stop and generate nothing.

For ablation studies, ```<algorithm_name>``` can be TEDDY-S or TEDDY-R which is described in Section 6.1.

For example,

```bash
python for.py -nt 1 -d DBLP 
```

To run all algorithms to generate the training data for a dataset, run the following command:

```bash
./run.sh <data_name>
```

where <data_name> can be DBLP, GENE, WIKI, IMDB or all. Here, ```all``` represents to generate the training data with all datasets (i.e., DBLP, GENE, WIKI and IMDB).

The default value of ```<max_hours_to_execute>``` is set to 30 (30 hours) to invoke the python code for.py to execute each algorithm.
