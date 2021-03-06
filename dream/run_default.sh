#!/bin/bash
device=${1:-0}
preview=${2:-1}
# analysis=${3:-0}
analysis=0
dnames='wiki2 imdb2 dblp'
seeds='0 1 2'
p_trains='0.1 0.2 0.4 0.6 0.8 1.0'
css='128 256 512 1024'
cscs='128 256 512 1024'
PTs='400 100 20 1'

dname=wiki2
seed=0
max_d=3
max_l=20
max_char=200
p_train=1.0
p_test=0.1
p_val=0.1
csc=512
vsc=256
cs=512
patience=5
l2=
lr=0.001
clr=0.001
vlr=0.001
layer=1
max_epoch=100
cmax_epoch=800
cmax_epoch_vae=100
pred_layer=3
prefix=true
delta=3
swa="" # swa=" --swa"
Ntbl=5
PT=20
L=10
bs=32
cbs=256
vbs=256
h_dim=512
es=100
vl2=0.01
vclip_lv=10.0
vclip_gr=0.01
clip_gr=
suffix="" # " --overwrite" " --analysis ts" # " --analysis lat" # " --analysis qs"



function set_rnn {
  l2=0.00000001
  clip_gr=10.0
}
function set_Prnn {
  l2=0.00000001
  clip_gr=10.0
}
function set_Ernn {
  l2=0.00000001
  clip_gr=10.0
}
function set_card {
  l2=0.00000001
  clip_gr=10.0
}
function set_Pcard {
  l2=0.00000001
  clip_gr=10.0
}

function param_search_Pcard_sub {
  local cbs
  local vbs
  local clr
  local vlr
  local csc
  local vsc
  cbs=${bs}
  vbs=${bs}
  clr=${lr}
  vlr=${lr}
  csc=$((sc * 2))
  vsc=${sc}

  # [Pcard]
  Pcard_sub
}

# param search Pcard
function param_search_Pcard {
  param_search_Pcard_sub
  local dname='dblp'
  local lr=0.001
  local sc=128
  local bs=256
  for bs in 4096 1024 64 16; do
    param_search_Pcard_sub
  done
  bs=256
  for lr in 0.01 0.0001; do
    param_search_Pcard_sub
  done
  lr=0.001
  for sc in 256 512 1024; do
    param_search_Pcard_sub
  done
  sc=128
}


# rnn subprocedure
function rnn_sub {
  echo [ rnn ] : CUDA_VISIBLE_DEVICES=${device} python run.py --model rnn --dname ${dname} --p-train ${p_train} --p-val ${p_val} --p-test ${p_test} --dsize max --seed ${seed} --l2 ${l2} --lr ${lr}${swa} --layer ${layer} --pred-layer ${pred_layer} --cs ${cs} --max-epoch ${max_epoch} --patience ${patience} --max-l ${max_l} --max-d ${max_d} --max-char ${max_char} --sep-emb --bs ${bs} --h-dim ${h_dim} --es ${es} --clip-gr ${clip_gr}${suffix}
  if [[ $preview -eq 0 ]]; then
    CUDA_VISIBLE_DEVICES=${device} python run.py --model rnn --dname ${dname} --p-train ${p_train} --p-val ${p_val} --p-test ${p_test} --dsize max --seed ${seed} --l2 ${l2} --lr ${lr}${swa} --layer ${layer} --pred-layer ${pred_layer} --cs ${cs} --max-epoch ${max_epoch} --patience ${patience} --max-l ${max_l} --max-d ${max_d} --max-char ${max_char} --sep-emb --bs ${bs} --h-dim ${h_dim} --es ${es} --clip-gr ${clip_gr}${suffix}
  fi
}


# Prnn subprocedure
function Prnn_sub {
  echo [Prnn ] : CUDA_VISIBLE_DEVICES=${device} python run.py --model rnn --dname ${dname} --p-train ${p_train} --p-val ${p_val} --p-test ${p_test} --dsize max --seed ${seed} --l2 ${l2} --lr ${lr}${swa} --layer ${layer} --pred-layer ${pred_layer} --cs ${cs} --max-epoch ${max_epoch} --patience ${patience} --max-l ${max_l} --max-d ${max_d} --max-char ${max_char} --sep-emb --prfx --bs ${bs} --h-dim ${h_dim} --es ${es} --clip-gr ${clip_gr}${suffix}
  if [[ $preview -eq 0 ]]; then
    CUDA_VISIBLE_DEVICES=${device} python run.py --model rnn --dname ${dname} --p-train ${p_train} --p-val ${p_val} --p-test ${p_test} --dsize max --seed ${seed} --l2 ${l2} --lr ${lr}${swa} --layer ${layer} --pred-layer ${pred_layer} --cs ${cs} --max-epoch ${max_epoch} --patience ${patience} --max-l ${max_l} --max-d ${max_d} --max-char ${max_char} --sep-emb --prfx --bs ${bs} --h-dim ${h_dim} --es ${es} --clip-gr ${clip_gr}${suffix}
  fi
}

# Ernn subprocedure
function Ernn_sub {
  echo [Ernn ] : CUDA_VISIBLE_DEVICES=${device} python run.py --model rnn --dname ${dname} --p-train ${p_train} --p-val ${p_val} --p-test ${p_test} --dsize max --seed ${seed} --l2 ${l2} --lr ${lr}${swa} --layer ${layer} --pred-layer ${pred_layer} --cs ${cs} --max-epoch ${max_epoch} --patience ${patience} --max-l ${max_l} --max-d ${max_d} --max-char ${max_char} --sep-emb --Eprfx --bs ${bs} --h-dim ${h_dim} --es ${es} --clip-gr ${clip_gr}${suffix}
  if [[ $preview -eq 0 ]]; then
    CUDA_VISIBLE_DEVICES=${device} python run.py --model rnn --dname ${dname} --p-train ${p_train} --p-val ${p_val} --p-test ${p_test} --dsize max --seed ${seed} --l2 ${l2} --lr ${lr}${swa} --layer ${layer} --pred-layer ${pred_layer} --cs ${cs} --max-epoch ${max_epoch} --patience ${patience} --max-l ${max_l} --max-d ${max_d} --max-char ${max_char} --sep-emb --Eprfx --bs ${bs} --h-dim ${h_dim} --es ${es} --clip-gr ${clip_gr}${suffix}
  fi
}

# Drnn subprocedure
function Drnn_sub {
  echo [Drnn ] : CUDA_VISIBLE_DEVICES=${device} python run.py --model rnn --dname ${dname} --p-train ${p_train} --p-val ${p_val} --p-test ${p_test} --dsize max --seed ${seed} --l2 ${l2} --lr ${lr}${swa} --layer ${layer} --pred-layer ${pred_layer} --cs ${cs} --max-epoch ${max_epoch} --patience ${patience} --max-l ${max_l} --delta ${delta} --max-char ${max_char} --sep-emb --prfx --bs ${bs} --h-dim ${h_dim} --es ${es} --clip-gr ${clip_gr}${suffix}
  if [[ $preview -eq 0 ]]; then
    CUDA_VISIBLE_DEVICES=${device} python run.py --model rnn --dname ${dname} --p-train ${p_train} --p-val ${p_val} --p-test ${p_test} --dsize max --seed ${seed} --l2 ${l2} --lr ${lr}${swa} --layer ${layer} --pred-layer ${pred_layer} --cs ${cs} --max-epoch ${max_epoch} --patience ${patience} --max-l ${max_l} --delta ${delta} --max-char ${max_char} --sep-emb --prfx --bs ${bs} --h-dim ${h_dim} --es ${es} --clip-gr ${clip_gr}${suffix}
  fi
}

# card subprocedure
function card_sub {
  echo [card ]: CUDA_VISIBLE_DEVICES=${device} python run.py --model card --dname ${dname} --p-train ${p_train} --p-val ${p_val} --p-test ${p_test} --dsize max --seed ${seed} --l2 ${l2} --lr ${clr} --vlr ${vlr} --csc ${csc} --vsc ${vsc} --max-epoch ${cmax_epoch} --patience ${patience} --max-l ${max_l} --max-d ${max_d} --max-char ${max_char} --bs ${cbs} --vbs ${vbs} --max-epoch-vae ${cmax_epoch_vae} --vl2 ${vl2} --vclip-lv ${vclip_lv} --vclip-gr ${vclip_gr} --clip-gr ${clip_gr}${suffix}
  if [[ $preview -eq 0 ]]; then
    CUDA_VISIBLE_DEVICES=${device} python run.py --model card --dname ${dname} --p-train ${p_train} --p-val ${p_val} --p-test ${p_test} --dsize max --seed ${seed} --l2 ${l2} --lr ${clr} --vlr ${vlr} --csc ${csc} --vsc ${vsc} --max-epoch ${cmax_epoch} --patience ${patience} --max-l ${max_l} --max-d ${max_d} --max-char ${max_char} --bs ${cbs} --vbs ${vbs} --max-epoch-vae ${cmax_epoch_vae} --vl2 ${vl2} --vclip-lv ${vclip_lv} --vclip-gr ${vclip_gr} --clip-gr ${clip_gr}${suffix}
  fi
}

# Pcard subprocedure
function Pcard_sub {
  echo [Pcard]: CUDA_VISIBLE_DEVICES=${device} python run.py --model card --dname ${dname} --p-train ${p_train} --p-val ${p_val} --p-test ${p_test} --dsize max --seed ${seed} --l2 ${l2} --lr ${clr} --vlr ${vlr} --csc ${csc} --vsc ${vsc} --max-epoch ${cmax_epoch} --patience ${patience} --max-l ${max_l} --max-d ${max_d} --max-char ${max_char} --Eprfx --bs ${cbs} --vbs ${vbs} --max-epoch-vae ${cmax_epoch_vae} --vl2 ${vl2} --vclip-lv ${vclip_lv} --vclip-gr ${vclip_gr} --clip-gr ${clip_gr}${suffix}
  if [[ $preview -eq 0 ]]; then
    CUDA_VISIBLE_DEVICES=${device} python run.py --model card --dname ${dname} --p-train ${p_train} --p-val ${p_val} --p-test ${p_test} --dsize max --seed ${seed} --l2 ${l2} --lr ${clr} --vlr ${vlr} --csc ${csc} --vsc ${vsc} --max-epoch ${cmax_epoch} --patience ${patience} --max-l ${max_l} --max-d ${max_d} --max-char ${max_char} --Eprfx --bs ${cbs} --vbs ${vbs} --max-epoch-vae ${cmax_epoch_vae} --vl2 ${vl2} --vclip-lv ${vclip_lv} --vclip-gr ${vclip_gr} --clip-gr ${clip_gr}${suffix}
  fi
}

# Dcard subprocedure
function Dcard_sub {
  echo [Pcard]: CUDA_VISIBLE_DEVICES=${device} python run.py --model card --dname ${dname} --p-train ${p_train} --p-val ${p_val} --p-test ${p_test} --dsize max --seed ${seed} --l2 ${l2} --lr ${clr} --vlr ${vlr} --csc ${csc} --vsc ${vsc} --max-epoch ${cmax_epoch} --patience ${patience} --max-l ${max_l} --delta ${delta} --max-char ${max_char} --Eprfx --bs ${cbs} --vbs ${vbs} --max-epoch-vae ${cmax_epoch_vae} --vl2 ${vl2} --vclip-lv ${vclip_lv} --vclip-gr ${vclip_gr} --clip-gr ${clip_gr}${suffix}
  if [[ $preview -eq 0 ]]; then
    CUDA_VISIBLE_DEVICES=${device} python run.py --model card --dname ${dname} --p-train ${p_train} --p-val ${p_val} --p-test ${p_test} --dsize max --seed ${seed} --l2 ${l2} --lr ${clr} --vlr ${vlr} --csc ${csc} --vsc ${vsc} --max-epoch ${cmax_epoch} --patience ${patience} --max-l ${max_l} --delta ${delta} --max-char ${max_char} --Eprfx --bs ${cbs} --vbs ${vbs} --max-epoch-vae ${cmax_epoch_vae} --vl2 ${vl2} --vclip-lv ${vclip_lv} --vclip-gr ${vclip_gr} --clip-gr ${clip_gr}${suffix}
  fi
}
# LBS subprocedure
function LBS_sub {
  echo [ LBS ]: python run.py --model eqt --dname ${dname} --p-test ${p_test} --dsize max --seed ${seed} --Ntbl ${Ntbl} --PT ${PT} --max-l ${max_l} --max-d ${max_d} --L ${L}${suffix}
  if [[ $preview -eq 0 ]]; then
    python run.py --model eqt --dname ${dname} --p-test ${p_test} --dsize max --seed ${seed} --Ntbl ${Ntbl} --PT ${PT} --max-l ${max_l} --max-d ${max_d} --L ${L}${suffix}
  fi
}

function rnn_default {
  local l2
  local clip_gr
  set_rnn
  rnn_sub
}
function Prnn_default {
  local l2
  local clip_gr
  set_Prnn
  Prnn_sub
}
function Ernn_default {
  local l2
  local clip_gr
  set_Ernn
  Ernn_sub
}
function card_default {
  local l2
  local clip_gr
  set_card
  card_sub
}
function Pcard_default {
  local l2
  local clip_gr
  set_Pcard
  Pcard_sub
}
function LBS_default {
  LBS_sub
}

function for_default_rnn {
  local seed
  local dname
  for seed in $seeds; do
    for dname in $dnames; do
      rnn_sub
    done
  done
}

function for_default_Prnn {
  local seed
  local dname
  for seed in $seeds; do
    for dname in $dnames; do
      Prnn_sub
    done
  done
}

function for_default_Ernn {
  local seed
  local dname
  for seed in $seeds; do
    for dname in $dnames; do
      Ernn_sub
    done
  done
}

function for_default_card {
  local seed
  local dname
  for seed in $seeds; do
    for dname in $dnames; do
      card_sub
    done
  done
}
function for_default_Pcard {
  local seed
  local dname
  for seed in $seeds; do
    for dname in $dnames; do
      Pcard_sub
    done
  done
}

function for_default_LBS {
  local seed
  local dname
  for seed in $seeds; do
    for dname in $dnames; do
      LBS_sub
    done
  done
}

function for_data_size_rnn {
  local seed
  local dname
  local p_train
  for seed in $seeds; do
    for dname in $dnames; do
      for p_train in $p_trains; do
        rnn_sub
      done
    done
  done
}
function for_data_size_Prnn {
  local seed
  local dname
  local p_train
  for seed in $seeds; do
    for dname in $dnames; do
      for p_train in $p_trains; do
        Prnn_sub
      done
    done
  done
}
function for_data_size_Ernn {
  local seed
  local dname
  local p_train
  for seed in $seeds; do
    for dname in $dnames; do
      for p_train in $p_trains; do
        Ernn_sub
      done
    done
  done
}
function for_data_size_card {
  local seed
  local dname
  local p_train
  for seed in $seeds; do
    for dname in $dnames; do
      for p_train in $p_trains; do
        card_sub
      done
    done
  done
}
function for_data_size_Pcard {
  local seed
  local dname
  local p_train
  for seed in $seeds; do
    for dname in $dnames; do
      for p_train in $p_trains; do
        Pcard_sub
      done
    done
  done
}

function for_model_size_Prnn {
  local seed
  local dname
  local cs
  local h_dim
  dname='dblp'
  for seed in $seeds; do
    for cs in $css; do
      h_dim=${cs}
      Prnn_sub
    done
  done
}

function for_model_size_Pcard {
  local seed
  local dname
  local csc
  local vsc
  dname='dblp'
  for seed in $seeds; do
    for csc in $cscs; do
      vsc=$((csc / 2))
      Pcard_sub
    done
  done
}

function for_model_size_LBS {
  local seed
  local dname
  local PT
  dname='dblp'
  for seed in $seeds; do
    for PT in $PTs; do
      LBS_sub
    done
  done
}

function for_analysis_Prnn {
  local suffix
  local dname
  local seed
  suffix=" --analysis"

  for seed in $seeds;do
    for dname in $dnames; do
      Prnn_sub
    done
  done
}

function for_analysis_Pcard {
  local suffix
  local dname
  local seed
  suffix=" --analysis"

  for seed in $seeds;do
    for dname in $dnames; do
      Pcard_sub
    done
  done
}

function for_each_delta_Prnn {
  local seed
  local dname
  local delta
  for seed in $seeds; do
    for dname in $dnames; do
      for delta in 0 1 2 3; do
        Drnn_sub
      done
    done
  done
}

function for_each_delta_Pcard {
  local seed
  local dname
  local delta
  for seed in $seeds; do
    for dname in $dnames; do
      for delta in 0 1 2 3; do
        Dcard_sub
      done
    done
  done
}

function rnn_all {
  local l2
  local clip_gr
  set_rnn

  if [[ $analysis -eq 0 ]]; then
    for_default_rnn
    # for_data_size_rnn
  fi
  echo
}

function Prnn_all {
  local l2
  local clip_gr
  set_Prnn

  if [[ $analysis -ne 0 ]]; then
    for_analysis_Prnn
  else
    for_default_Prnn
    # for_data_size_Prnn
    # for_model_size_Prnn
  fi 
  # for_each_delta_Prnn
  echo
}

function Ernn_all {
  local l2
  local clip_gr
  set_Ernn

  if [[ $analysis -eq 0 ]]; then
    for_default_Ernn
  fi
  echo
}

function card_all {
  local l2
  local clip_gr
  set_card

  if [[ $analysis -eq 0 ]]; then
    for_default_card
  fi
  echo
}

function Pcard_all {
  local l2
  local clip_gr
  set_Pcard

  if [[ $analysis -ne 0 ]]; then
    for_analysis_Pcard
  else
    for_default_Pcard
    # for_data_size_Pcard
    # for_model_size_Pcard
  fi
  # for_each_delta_Pcard
  echo
}

function LBS_all {
  if [[ $analysis -eq 0 ]]; then
    for_default_LBS
    # for_model_size_LBS
  fi
  echo
}

function run_all {
  rnn_all
  Prnn_all
  Ernn_all
  card_all
  Pcard_all
  LBS_all
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  if [[ $preview -ne 0 ]]; then
    rnn_default
    Prnn_default
    Ernn_default
    card_default
    Pcard_default
    LBS_default
    echo
    run_all
  fi
fi
