#!/bin/bash

seed=0
dname=dblp
pt_r=1.0
max_l=20
delta=0
device=0
es=64
bs=128
lr=0.001
epoch=64
p_train=1.0
suffix=" --overwrite" # " --analysis ts" " --analysis qs" " --analysis lat"


for seed in 0; do
    for delta in 0 1 2 3; do
        # default
        echo "dname: ${dname} pt_r: ${pt_r} max_l: ${max_l} delta: ${delta} p_train ${p_train} seed: ${seed} es: ${es} bs ${bs} lr ${lr} epoch ${epoch}"
        echo CUDA_VISIBLE_DEVICES=${device} python AstridEmbed.py --path datasets/${dname}/ --prfx qs_${dname} --delta ${delta} --pt-r ${pt_r} --max-l ${max_l} --p-train ${p_train} --seed ${seed} --es ${es} --bs ${bs} --lr ${lr} --epoch ${epoch}${suffix}
        CUDA_VISIBLE_DEVICES=${device} python AstridEmbed.py --path datasets/${dname}/ --prfx qs_${dname} --delta ${delta} --pt-r ${pt_r} --max-l ${max_l} --p-train ${p_train} --seed ${seed} --es ${es} --bs ${bs} --lr ${lr} --epoch ${epoch}${suffix}

        # embed size
        for es in 32 96; do 
            echo "dname: ${dname} pt_r: ${pt_r} max_l: ${max_l} delta: ${delta} p_train ${p_train} seed: ${seed} es: ${es} bs ${bs} lr ${lr} epoch ${epoch}"
            echo CUDA_VISIBLE_DEVICES=${device} python AstridEmbed.py --path datasets/${dname}/ --prfx qs_${dname} --delta ${delta} --pt-r ${pt_r} --max-l ${max_l} --p-train ${p_train} --seed ${seed} --es ${es} --bs ${bs} --lr ${lr} --epoch ${epoch}${suffix}
            CUDA_VISIBLE_DEVICES=${device} python AstridEmbed.py --path datasets/${dname}/ --prfx qs_${dname} --delta ${delta} --pt-r ${pt_r} --max-l ${max_l} --p-train ${p_train} --seed ${seed} --es ${es} --bs ${bs} --lr ${lr} --epoch ${epoch}${suffix}
        done
        es=64

        for lr in 0.01 0.0001; do 
            echo "dname: ${dname} pt_r: ${pt_r} max_l: ${max_l} delta: ${delta} p_train ${p_train} seed: ${seed} es: ${es} bs ${bs} lr ${lr} epoch ${epoch}"
            echo CUDA_VISIBLE_DEVICES=${device} python AstridEmbed.py --path datasets/${dname}/ --prfx qs_${dname} --delta ${delta} --pt-r ${pt_r} --max-l ${max_l} --p-train ${p_train} --seed ${seed} --es ${es} --bs ${bs} --lr ${lr} --epoch ${epoch}${suffix}
            CUDA_VISIBLE_DEVICES=${device} python AstridEmbed.py --path datasets/${dname}/ --prfx qs_${dname} --delta ${delta} --pt-r ${pt_r} --max-l ${max_l} --p-train ${p_train} --seed ${seed} --es ${es} --bs ${bs} --lr ${lr} --epoch ${epoch}${suffix}
        done
        lr=0.001

        for bs in 64 256; do 
            echo "dname: ${dname} pt_r: ${pt_r} max_l: ${max_l} delta: ${delta} p_train ${p_train} seed: ${seed} es: ${es} bs ${bs} lr ${lr} epoch ${epoch}"
            echo CUDA_VISIBLE_DEVICES=${device} python AstridEmbed.py --path datasets/${dname}/ --prfx qs_${dname} --delta ${delta} --pt-r ${pt_r} --max-l ${max_l} --p-train ${p_train} --seed ${seed} --es ${es} --bs ${bs} --lr ${lr} --epoch ${epoch}${suffix}
            CUDA_VISIBLE_DEVICES=${device} python AstridEmbed.py --path datasets/${dname}/ --prfx qs_${dname} --delta ${delta} --pt-r ${pt_r} --max-l ${max_l} --p-train ${p_train} --seed ${seed} --es ${es} --bs ${bs} --lr ${lr} --epoch ${epoch}${suffix}
        done
        bs=128

        for epoch in 32 128; do 
            echo "dname: ${dname} pt_r: ${pt_r} max_l: ${max_l} delta: ${delta} p_train ${p_train} seed: ${seed} es: ${es} bs ${bs} lr ${lr} epoch ${epoch}"
            echo CUDA_VISIBLE_DEVICES=${device} python AstridEmbed.py --path datasets/${dname}/ --prfx qs_${dname} --delta ${delta} --pt-r ${pt_r} --max-l ${max_l} --p-train ${p_train} --seed ${seed} --es ${es} --bs ${bs} --lr ${lr} --epoch ${epoch}${suffix}
            CUDA_VISIBLE_DEVICES=${device} python AstridEmbed.py --path datasets/${dname}/ --prfx qs_${dname} --delta ${delta} --pt-r ${pt_r} --max-l ${max_l} --p-train ${p_train} --seed ${seed} --es ${es} --bs ${bs} --lr ${lr} --epoch ${epoch}${suffix}
        done
        epoch=64

    done
done