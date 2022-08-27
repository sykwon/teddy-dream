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
python run.py --model <model_name> --dname <data_name> --p-train <ratio_training> --p-val <ratio_validation> --p-test <ratio_test> --seed <seed> --l2 <l2_regularization> --lr <learning_rate> --layer <number_encoder_layers> --pred-layer <number_decoder_layers> --cs <model_scale> --max-epoch <max_epoch> --patience <patience> --max-d <delta_M> --max-char <max_char> --bs <batch_size> --h-dim 512 --es <embedding_size> --clip-gr <gradient_clipping> 
```

* <model_name>: the name of the cardinality estimator (DREAM, CardNet or LBS)  
The meanings of DREAM, Qgram and LBS are described in Section 6 of our paper.
* <data_name>: the name of dataset (DBLP, GENE, WIKI or IMDB)  
The meanings of DBLP, GENE, WIKI and IMDB are described in Section 6 of our paper.
* <delta_M>: the maximum substring edit distance threshold
* <seed_xx>: the random seed to generate the initial weights of the estimator model

For example, if we use the following command, we train the DREAM model with the base training data for the DBLP dataset and evaluate the model with test data.
The model parameters of the DREAM model will be printed by using the ```summary``` function by importing ```torchsummary``` before training the model. The description of the ```summary``` function can be found in <https://pypi.org/project/torch-summary/>.
After training as well as evaluation of the DREAM model are done, the file pathes where the output file of the estimated cardinalities for test data and the trained model are printed.
In addition, the average q-error of estimated cardinalities is printed.

```bash
python run.py --model DREAM --dname DBLP --p-train 1.0 --p-val 0.1 --p-test 0.1 --seed 0 --l2 0.00000001 --lr 0.001 --layer 1 --pred-layer 3 --cs 512 --max-epoch 100 --patience 5 --max-d 3 --max-char 200 --bs 32 --h-dim 512 --es 100 --clip-gr 10.0

RNN_module (
  (embedding): ConcatEmbed(
    (char_embedding): Embedding(65, 95, padding_idx=0)
    (dist_embedding): Embedding(4, 5)
  ), weights=((65, 95), (4, 5)), parameters=6195
  (rnns): ModuleList(
    (0): LSTM(100, 512, batch_first=True)
  ), weights=((2048, 100), (2048, 512), (2048,), (2048,)), parameters=1257472
  (rnn): LSTM(100, 512, batch_first=True), weights=((2048, 100), (2048, 512), (2048,), (2048,)), parameters=1257472
  (pred): Sequential (
    (PRED-1): Linear(in_features=512, out_features=512, bias=True), weights=((512, 512), (512,)), parameters=262656
    (LeakyReLU-1): LeakyReLU(negative_slope=0.01), weights=(), parameters=0
    (PRED-2): Linear(in_features=512, out_features=512, bias=True), weights=((512, 512), (512,)), parameters=262656
    (LeakyReLU-2): LeakyReLU(negative_slope=0.01), weights=(), parameters=0
    (PRED-3): Linear(in_features=512, out_features=512, bias=True), weights=((512, 512), (512,)), parameters=262656
    (LeakyReLU-3): LeakyReLU(negative_slope=0.01), weights=(), parameters=0
    (PRED-OUT): Linear(in_features=512, out_features=1, bias=True), weights=((1, 512), (1,)), parameters=513
  ), weights=((512, 512), (512,), (512, 512), (512,), (512, 512), (512,), (1, 512), (1,)), parameters=788481
)
total params: 3309620

The trained model are written as model/DBLP/DREAM_DBLP_cs_512_layer_1_predL_3_hDim_512_es_100_lr_0.001_maxC_200_pVal_0.1_ptrain_1.0_l2_1e-08_pat_5_clipGr_10.0_seed_0_maxEpoch_100_maxD_3_pTest_0.1_bs_32/saved_model.pth 
The estimated cardinalities are written as exp_result/DBLP/DREAM_DBLP_cs_512_layer_1_predL_3_hDim_512_es_100_lr_0.001_maxC_200_pVal_0.1_ptrain_1.0_l2_1e-08_pat_5_clipGr_10.0_seed_0_maxEpoch_100_maxD_3_pTest_0.1_bs_32/analysis_lat_gpu.csv

average q-error: 2.859777
```

The above output says that the trained model is stored at the file with the name ```saved_model.pth``` in the directory ```model/DBLP/DREAM_DBLP_cs_512_layer_1_predL_3_hDim_512_es_100_lr_0.001_maxC_200_pVal_0.1_ptrain_1.0_l2_1e-08_pat_5_clipGr_10.0_seed_0_maxEpoch_100_maxD_3_pTest_0.1_bs_32```.
Furthermore, it says that the average q-error of estimated cardinalities is 2.859777 for the DREAM model.

To measure the estimation errors of all estimators except Astrid for a dataset, run the following command:

```bash
./run.sh <data_name>
```

where ```<data_name>``` can be DBLP, GENE, WIKI, IMDB or all. Here, ```all``` represents to generate the training data with all datasets (i.e., DBLP, GENE, WIKI and IMDB).
