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

## Training and evaluating cardinality estimators

To train and evaluate the cardinality estimators, run the following command:

```bash
python run.py --model <model_name> --dname <data_name> --p-train <ratio_training> --p-val <ratio_validation> --p-test <ratio_test> --seed <seed_number> --l2 <l2_regularization> --lr <learning_rate> --layer <number_encoder_layers> --pred-layer <number_decoder_layers> --cs <encoder_scale> --max-epoch <max_epoch> --patience <patience_number> --max-d <delta_M> --max-char <max_char> --bs <batch_size> --h-dim <decoder_scale> --es <embedding_size> --clip-gr <gradient_clipping> 
```

* <model_name>: the name of the cardinality estimator (DREAM, CardNet or LBS)  
The meanings of DREAM, Qgram and LBS are described in Section 6 of our paper.
* <data_name>: the name of the dataset (DBLP, GENE, WIKI or IMDB)  
The meanings of DBLP, GENE, WIKI and IMDB are described in Section 6 of our paper.
* <ratio_training>: the sampling ratio of query strings to train the estimators
* <ratio_validation>: the ratio of the validation dataset
* <ratio_test>: the ratio of the test dataset
* <seed_number>: the random seed to generate the initial weights of the estimator model
* <l2_regularization>: the coefficient of l2 regularization
* <learning_rate>: the learning rate of the gradient descent optimization
* <number_encoder_layers>: the number of layers in the encoder of the model
* <encoder_scale>: the scale of the encoder of the model
* <max_epoch>: the maximum number of epochs to train the model
* <patience_number>: the number of epochs without improvement after which training will be early stopped
* <delta_M>: the maximum substring edit distance threshold
* <max_char>: It is the maximum number of most frequent characters to keep for the cardinality estimator.  
    The remaining characters are considered unknown.
* <batch_size>: the number of samples processed before the model is updated
* <decoder_scale>: the scale of the decoder of the model
* <embedding_size>: the size of concatenated embeddings of a pair of a character and a distance (Note that the size of a distance embedding is fixed as 5.)
* <gradient_clipping>: the maximum norm of a gradient to ensure the stable learning

For example, if we use the following command, we train the DREAM model with the base training data for the DBLP dataset and evaluate the model with the test data.
The model parameters of the DREAM model will be printed based on ```pytorch```  before training the model. The description can be found in <https://pytorch.org/docs/1.7.1/generated/torch.nn.Module.html>.
While training the DREAM model, it will print the average q-error with the validation data.
After training as well as evaluation of the DREAM model are done, the file paths where the output file of the estimated cardinalities for test data and the trained model are printed.
In addition, the average q-error of estimated cardinalities is printed.

```bash
python run.py --model DREAM --dname DBLP --p-train 1.0 --p-val 0.1 --p-test 0.1 --seed 0 --l2 0.00000001 --lr 0.001 --layer 1 --pred-layer 3 --cs 512 --max-epoch 100 --patience 5 --max-d 3 --max-char 200 --bs 32 --h-dim 512 --es 100 --clip-gr 10.0

RNN_module(
  (embedding): ConcatEmbed(
    (char_embedding): Embedding(65, 95, padding_idx=0)
    (dist_embedding): Embedding(4, 5)
  )
  (rnns): ModuleList(
    (0): LSTM(100, 512, batch_first=True)
  )
  (rnn): LSTM(100, 512, batch_first=True)
  (pred): Sequential(
    (PRED-1): Linear(in_features=512, out_features=512, bias=True)
    (LeakyReLU-1): LeakyReLU(negative_slope=0.01)
    (PRED-2): Linear(in_features=512, out_features=512, bias=True)
    (LeakyReLU-2): LeakyReLU(negative_slope=0.01)
    (PRED-3): Linear(in_features=512, out_features=512, bias=True)
    (LeakyReLU-3): LeakyReLU(negative_slope=0.01)
    (PRED-OUT): Linear(in_features=512, out_features=1, bias=True)
  )
)
[epoch 01] average q-error of validation: 004.273
[epoch 02] average q-error of validation: 003.927
[epoch 03] average q-error of validation: 003.431
[epoch 04] average q-error of validation: 003.189
[epoch 05] average q-error of validation: 002.899
[epoch 06] average q-error of validation: 002.858
[epoch 07] average q-error of validation: 002.874
[epoch 08] average q-error of validation: 002.695
[epoch 09] average q-error of validation: 002.700
[epoch 10] average q-error of validation: 002.681
[epoch 11] average q-error of validation: 002.702
[epoch 12] average q-error of validation: 002.772
[epoch 13] average q-error of validation: 002.796
[epoch 14] average q-error of validation: 002.745
[epoch 15] average q-error of validation: 002.749
The trained model are written as model/DBLP/DREAM_DBLP_cs_512_layer_1_predL_3_hDim_512_es_100_lr_0.001_maxC_200_pVal_0.1_ptrain_1.0_l2_1e-08_pat_5_clipGr_10.0_seed_0_maxEpoch_100_maxD_3_pTest_0.1_bs_32/saved_model.pth 
The estimated cardinalities are written as exp_result/DBLP/DREAM_DBLP_cs_512_layer_1_predL_3_hDim_512_es_100_lr_0.001_maxC_200_pVal_0.1_ptrain_1.0_l2_1e-08_pat_5_clipGr_10.0_seed_0_maxEpoch_100_maxD_3_pTest_0.1_bs_32/analysis_lat_gpu.csv

average q-error: 2.87
```

The above output says that the trained model is stored at the file with the name ```saved_model.pth``` in the directory ```model/DBLP/DREAM_DBLP_cs_512_layer_1_predL_3_hDim_512_es_100_lr_0.001_maxC_200_pVal_0.1_ptrain_1.0_l2_1e-08_pat_5_clipGr_10.0_seed_0_maxEpoch_100_maxD_3_pTest_0.1_bs_32```.
Furthermore, it says that the average q-error of estimated cardinalities is 2.859777 for the DREAM model.

To measure the estimation errors of all estimators except Astrid for a dataset, run the following command:

```bash
./run.sh <data_name>
```

where ```<data_name>``` can be DBLP, GENE, WIKI, IMDB or all. Here, ```all``` represents to generate the training data with all datasets (i.e., DBLP, GENE, WIKI and IMDB).
