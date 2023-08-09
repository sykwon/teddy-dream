# Traning Data Generation

[![license](https://img.shields.io/github/license/sykwon/teddy-dream?color=brightgreen)](https://github.com/sykwon/teddy-dream/blob/master/LICENSE)
[![ubuntu](https://img.shields.io/badge/ubuntu-v18.04-orange)](https://wiki.ubuntu.com/Releases)
![gcc 7.5](https://img.shields.io/badge/gcc-v7.5-blue)
![python 3.7](https://img.shields.io/badge/python-v3.7-blue)

## Installation

This code needs Python-3.7 or higher as well as gcc-7.5 or higher.

```bash
sudo apt-get install redis-server
sudo apt-get install binutils
pip install redis
make clean && make && make info
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
For ablation studies, ```<algorithm_name>``` can be TEDDY-S or TEDDY-R which is described in Section 6.1.
* <prefix_aug_flag>: the flag to represent whether to generate the prefix-aug training data (0: base training data; 1: prefix-aug training data)
* <ratio_training>: the sampling ratio of query strings used to generate the training data
* <delta_M>: the maximum substring edit distance threshold
* <number_of_repetitions>: the number of repeated executions of generating a dataset  
  This parameter value is used to compute the average execution time for experiments.
* <max_hours_to_execute>: the time to be allowed for generating training data (1 means an hour.)  
If the algorithm does not finish within ```<max_hours_to_execute>```, we stop and generate nothing.

For example, if we use the following command, we run the TEDDY algorithm to generate base training data for the DBLP dataset.
The output will provide where the generated training data is stored, the generation algorithm name, the training data type and its generation time. The training data will be stored in ```res/``` folder.

```bash
$ python for.py -d DBLP -a TEDDY -pr 0 -pt 1.0 -th 3 -nt 1 -hr 30
The training data is written as res/qs_DBLP_1.0_TEDDY_03_base.txt
TEDDY   base     151.777
```

The above output says that the training data is stored in the directory ```res/``` and its name is ```qs_DBLP_1.0_TEDDY_03_base.txt```.
Furthermore, it says that the TEDDY algorithm took 151.777 (seconds) to generate the base training data.

To measure the execution times of all algorithms for generating the training data for a dataset, run the following command:

```bash
./run.sh <data_name>
```

where ```<data_name>``` can be DBLP, GENE, WIKI, IMDB or all. Here, ```all``` represents to generate the training data with all datasets (i.e., DBLP, GENE, WIKI and IMDB).

In this case, the default value of ```<max_hours_to_execute>``` is set to 30 (30 hours) to invoke the python code for.py to execute each algorithm.
