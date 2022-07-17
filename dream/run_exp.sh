#!/bin/bash

source run_default.sh

dnames='dblp'
# dnames='dblp egr1 wiki2 imdb2' # uncomment for DBLP GENE WIKI IMDB datasets
seeds='0'
p_train=1.0

# run_all
rnn_all
Prnn_all
Ernn_all
card_all
Pcard_all
LBS_all
