#!/bin/bash
exp=${1}
make clean && make && make info
mkdir -p res
mkdir -p stat
mkdir -p time

if [[ "$exp" = "test" ]]; then
    echo "Start quick test"
    # run a test on 1% query strings for the DBLP dataset
    python for.py -s 0 -d dblp -a teddy2 -pr 0 1 -t 3 -hr 30 -L 20 -nq 0.01

    # run tests on 1% training data set for the DBLP dataset
    python for.py -s 0 -d dblp -a allp topk taste soddy2 teddy2 -pr 0 1 -t 3 -hr 30 -L 20 -pt 0.01 -nq 1.0
    python for.py -s 0 -d dblp -a teddy2 teddy0 abl1 allp -pr 0 1 -t 3 -hr 30 -L 20 -pt 0.01 -nq 1.0 --info                                  # Ablation study (stat)
elif [[ "$exp" = "dblp" ]]; then 
    echo "Start exp on the DBLP dataset"
    python for.py -s 0 -d dblp -a teddy2 -pr 0 1 -t 3 -hr 30 -L 20 -nq 1.0 # data genration including the training, validation and test datasets
    python for.py -s 0 -d dblp -a allp topk taste soddy2 teddy2 -pr 0 1 -t 3 -hr 30 -L 20 -pt 0.01 0.03 0.1 0.3 1.0 -nq 1.0  # Comparison of data genration algorithms
    python for.py -s 0 -d dblp -a taste teddy2 -pr 0 1 -t 3 -hr 30 -L 20 -pt 0.1 0.2 0.4 1.0 -nq 1.0                            # Tradeoff between time and accuracy
    python for.py -s 0 -d dblp -a teddy2 teddy0 abl1 allp -pr 0 1 -t 3 -hr 30 -L 20 -pt 1.0 -nq 1.0                                  # Ablation study (time)
    python for.py -s 0 -d dblp -a teddy2 teddy0 abl1 allp -pr 0 1 -t 3 -hr 30 -L 20 -pt 1.0 -nq 1.0 --info                                  # Ablation study (stat)
elif [[ "$exp" = "all" ]]; then 
    echo "Start exp on all datasets"
    python for.py -s 0 -d dblp egr1 wiki2 imdb2 -a teddy2 -pr 0 1 -t 3 -hr 30 -L 20 -nq 1.0 # data genration including the training, validation and test datasets
    python for.py -s 0 -d dblp egr1 wiki2 imdb2 -a allp topk taste soddy2 teddy2 -pr 0 1 -t 3 -hr 30 -L 20 -pt 0.01 0.03 0.1 0.3 1.0 -nq 1.0  # Comparison of data genration algorithms
    python for.py -s 0 -d dblp egr1 wiki2 imdb2 -a taste teddy2 -pr 0 1 -t 3 -hr 30 -L 20 -pt 0.1 0.2 0.4 1.0 -nq 1.0                            # Tradeoff between time and accuracy
    python for.py -s 0 -d dblp egr1 -a teddy2 teddy0 abl1 allp -pr 0 1 -t 3 -hr 30 -L 20 -pt 1.0 -nq 1.0                                  # Ablation study (time)
    python for.py -s 0 -d dblp egr1 -a teddy2 teddy0 abl1 allp -pr 0 1 -t 3 -hr 30 -L 20 -pt 1.0 -nq 1.0 --info                                  # Ablation study (stat)
else
    echo "type exp option among [test, dblp, all]"
fi
