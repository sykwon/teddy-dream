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
python run.py --model <model_name> --dname <data_name> --p-train <ratio_training> --p-val <ratio_validation> --p-test <ratio_test> --seed <seed> --l2 <l2_regularization> --lr <learning_rate> --layer <number_encoder_layers> --pred-layer <number_decoder_layers> --cs <model_scale> --max-epoch <max_epoch> --patience <patience> --max-d <delta_M> --max-char <max_char> --bs <batch_size> --h-dim 512 --es <embedding_size> --clip-gr <gradient_clipping> 
```

### Example

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
train_loss: 001.342 train_q_error: 003.211: 100%|██████████| 2477/2477 [00:13<00:00, 186.27it/s]
[epoch 01] valid_loss: 001.234, valid_q_error: 004.310
train_loss: 001.390 train_q_error: 003.854: 100%|██████████| 2477/2477 [00:13<00:00, 186.10it/s]
[epoch 02] valid_loss: 001.121, valid_q_error: 003.904
...
```

<!-- ### Example 2: CardNet

```bash
python run.py --model DREAM --dname DBLP --p-train 1.0 --p-val 0.1 --p-test 0.1 --seed 0 --l2 0.00000001 --lr 0.001 --layer 1 --pred-layer 3 --cs 512 --max-epoch 100 --patience 5 --max-d 3 --max-char 200 --bs 32 --h-dim 512 --es 100 --clip-gr 10.0
```

### Example 3: LBS

```bash
python run.py --model DREAM --dname DBLP --p-train 1.0 --p-val 0.1 --p-test 0.1 --seed 0 --l2 0.00000001 --lr 0.001 --layer 1 --pred-layer 3 --cs 512 --max-epoch 100 --patience 5 --max-d 3 --max-char 200 --bs 32 --h-dim 512 --es 100 --clip-gr 10.0
``` -->

## Script

```bash
# exp on DBLP 
./run.sh DBLP

# all exp
./run.sh all
```
