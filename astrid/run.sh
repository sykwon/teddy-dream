#!/bin/bash

source run_default.sh
if [[ "$exp" = "test" ]]; then
    dnames='dblp'
    p_train=0.1
elif [[ "$exp" = "dblp" ]]; then
    dnames='dblp'
elif [[ "$exp" = "all" ]]; then
    dnames='dblp egr1 wiki2 imdb2' # uncomment for DBLP GENE WIKI IMDB datasets
else
    echo "type exp option among [test, dblp, all]"
    exit
fi

astrid_default
astrid_analysis
# astrid_train_size
