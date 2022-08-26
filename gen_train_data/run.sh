#!/bin/bash
exp=${1:-help}

if [[ "$exp" = "help" ]]; then
    echo "Usage: ./run.sh {DBLP, GENE, WIKI, IMDB, all}"
    exit
elif [[ "$exp" = "all" ]]; then
    echo "Start experiments on all datasets"
    exp="DBLP GENE WIKI IMDB"
else
    echo "Start experiments on the ${exp} dataset"
fi

make clean && make && make info
# python for.py -nt 1 -d $exp -a TEDDY -pr 1 -t 3 -hr 30 -L 20                                                  # data genration including the training, validation and test datasets
python for.py -d $exp -a NaiveGen Qgram TASTE SODDY TEDDY -pr 0 1 -t 3 -hr 30 -pt 0.01 0.03 0.1 0.3 1.0 -nt 1 # Comparison of data genration algorithms
# python for.py -nt 1 -d $exp -a TASTE TEDDY -pr 0 1 -t 3 -hr 30 -L 20 -pt 0.1 0.2 0.4 1.0                             # Tradeoff between time and accuracy
# python for.py -nt 1 -d DBLP GENE -a TEDDY TEDDY-S TEDDY-R NaiveGen -pr 0 1 -t 3 -hr 30 -L 20 -pt 1.0                                   # Ablation study (time)
# python for.py -nt 1 -d DBLP GENE -a TEDDY TEDDY-S TEDDY-R NaiveGen -pr 0 1 -t 3 -hr 30 -L 20 -pt 1.0 --info                                  # Ablation study (stat)
