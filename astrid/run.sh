#!/bin/bash

source run_default.sh
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
preview=0

astrid_default
astrid_analysis
# astrid_train_size
