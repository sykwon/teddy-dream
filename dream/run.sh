#!/bin/bash

exp=${1:-help}

source run_default.sh
preview=1

if [[ "$exp" = "help" ]]; then
    echo "Usage: ./run.sh {DBLP, GENE, WIKI, IMDB, all}"
    exit
elif [[ "$exp" = "all" ]]; then
    echo "Start experiments on all datasets"
    exp="DBLP GENE WIKI IMDB"
    dnames=$exp
else
    echo "Start experiments on the ${exp} dataset"
    dnames=$exp
fi

seeds='0'

rnn_all
# Prnn_all
# card_all
# Pcard_all
# LBS_all

# run_all
# Ernn_all
