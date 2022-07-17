#!/bin/bash

device=${1:-0}
delta_M=${2:-0}        
preview=${3:-1}
dnames='dblp egr1 wiki2 imdb2'
seeds='0'
p_trains='0.2 0.4 0.6 0.8 1.0'
dscs='128 256 512 1024'

seed=0                                     
dname=dblp                               
pt_r=1.0                          
max_l=20                  
es=512      
bs=2048     
lr=0.001            
epoch=64
emb_epoch=8
dsc=1024                    
p_train=1.0               
suffix="" # " --analysis ts" " --analysis qs" " --analysis lat"
                                              
function astrid_sub {     
    local delta
    for delta in $(seq 0 $delta_M); do
        if [[ $preview -ne 0 ]]; then
            echo CUDA_VISIBLE_DEVICES=${device} python AstridEmbed.py --path datasets/${dname}/ --prfx qs_${dname} --delta ${delta} --pt-r ${pt_r} --max-l ${max_l} --p-train ${p_train} --seed ${seed} --es ${es} --bs ${bs} --lr ${lr} --epoch ${epoch} --emb-epoch ${emb_epoch} --dsc ${dsc}${suffix}
        else
            CUDA_VISIBLE_DEVICES=${device} python AstridEmbed.py --path datasets/${dname}/ --prfx qs_${dname} --delta ${delta} --pt-r ${pt_r} --max-l ${max_l} --p-train ${p_train} --seed ${seed} --es ${es} --bs ${bs} --lr ${lr} --epoch ${epoch} --emb-epoch ${emb_epoch} --dsc ${dsc}${suffix}
        fi           
    done
}                           
                     
function astrid_default {    
    local seed
    local dname
    for seed in $seeds; do                      
        for dname in $dnames; do
            astrid_sub
        done
    done
    echo
}
function astrid_train_size {
    local seed
    local dname
    local p_train
    for seed in $seeds; do     
        for dname in $dnames; do                                  
            for p_train in $p_trains; do
                astrid_sub
            done
        done
    done                                                   
    if [[ $preview -eq 0 ]]; then
        echo
    fi
}                                                         
                 
function astrid_model_size {
    local dsc
    local es
    for seed in $seeds; do
        for dname in $dnames; do
            for dsc in $dscs; do
                es=$((dsc / 2 ))         
                astrid_sub        
            done          
        done    
    done    

    if [[ $preview -eq 0 ]]; then
        echo
    fi
}
                            
function astrid_analysis {
    local suffix
    suffix=" --analysis ts"
    astrid_default
}