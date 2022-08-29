# Astrid

![license](https://img.shields.io/github/license/sykwon/teddy-dream?color=brightgreen)
[![ubuntu](https://img.shields.io/badge/ubuntu-v18.04-orange)](https://wiki.ubuntu.com/Releases)
![python 3.7](https://img.shields.io/badge/python-v3.7-blue)
![cuda 11.6](https://img.shields.io/badge/cuda-v11.6-blue)

This is the modified version of Astrid starting from the astrid model downloaded from Astrid's authors repository [[github](<https://github.com/saravanan-thirumuruganathan/astrid-string-selectivity>)].

## Training and evaluating cardinality estimators

To train and evaluate the cardinality estimators, run the following command:

```bash
python AstridEmbed.py --dname <data_name> --delta <delta> --p-train <ratio_training> --seed <seed_number> --es <embedding_size> --bs <batch_size> --<learning_rate> --epoch <max_epoch> --emb-epoch <max_epoch_emb> --dsc <decoder_scale>
```

* <data_name>: the name of dataset (DBLP, GENE, WIKI or IMDB)  
The meanings of DBLP, GENE, WIKI and IMDB are described in Section 6 of our paper.
* <delta>: the substring edit distance threshold for training and test queries
* <ratio_training>: the ratio of training dataset
* <seed_number>: the random seed to generate the initial weights of the estimator model
* <learning_rate>: the learning rate of the gradient descent optimization
* <max_epoch>: the maximum number of epoches to train the estimator model
* <max_epoch_emb>: the maximum number of epoches to train the embedding model
* <batch_size>: the batch size of each training step
* <decoder_scale>: the scale of the decoder of the model
* <embedding_size>: the embedding size of concatenated embedding for character and distance

For example, if we use the following command, we train the Astrid model with the base training data with ```<delta>=1``` for the DBLP dataset and evaluate the model with the test data.  After training as well as evaluation of the Astrid model are done, the file pathes where the output file of the estimated cardinalities for test data and the trained models are printed.
In addition, the average q-error of estimated cardinalities is printed.

```bash
python AstridEmbed.py --dname DBLP --delta 1 --p-train 1.0 --seed 0 --es 512 --bs 2048 --lr 0.001 --epoch 64 --emb-epoch 8 --dsc 1024
start pretraining embedding model
The pretrained embedding model are written as log/DBLP/DBLP_0_512_2048_0.001_64_1.0_8_1024/embedding_model_1.pth
start training estimator model
The estimated cardinalities are written as log/DBLP/DBLP_0_512_2048_0.001_64_1.0_8_1024/selectivity_model_1.pth
The estimated cardinalities are written as log/DBLP/DBLP_0_512_2048_0.001_64_1.0_8_1024/analysis_ts_1.csv
average q-error: 2.64
```

The above output says that the trained models are stored at the file with the names ```embedding_model_1.pth``` and ```selectivity_model_1.pth``` in the directory ```log/DBLP/DBLP_0_512_2048_0.001_64_1.0_8_1024/```.
Furthermore, it says that the average q-error of estimated cardinalities is 2.64 for the DREAM model.

To measure the estimation errors of Astrid for a dataset, run the following command:

```bash
./run.sh <data_name>
```

where ```<data_name>``` can be DBLP, GENE, WIKI, IMDB or all. Here, all represents to generate the training data with all datasets (i.e., DBLP, GENE, WIKI and IMDB).
