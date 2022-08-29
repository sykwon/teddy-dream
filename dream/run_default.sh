#!/bin/bash
exp=${1}
device=0
preview=0
analysis=0
dnames='WIKI IMDB DBLP GENE'
seeds='0 1 2'
p_trains='0.1 0.2 0.4 0.6 0.8 1.0'
css='128 256 512 1024'
cscs='128 256 512 1024'
PTs='400 100 20 1'

dname=WIKI
seed=0
max_d=3
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
function set_PDREAM {
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
function set_PCardNet {
  l2=0.00000001
  clip_gr=10.0
}

function param_search_PCardNet_sub {
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

  # [PCardNet]
  PCardNet_sub
}

# param search PCardNet
function param_search_PCardNet {
  param_search_PCardNet_sub
  local dname='DBLP'
  local lr=0.001
  local sc=128
  local bs=256
  for bs in 4096 1024 64 16; do
    param_search_PCardNet_sub
  done
  bs=256
  for lr in 0.01 0.0001; do
    param_search_PCardNet_sub
  done
  lr=0.001
  for sc in 256 512 1024; do
    param_search_PCardNet_sub
  done
  sc=128
}

# DREAM subprocedure
function rnn_sub {
  echo [ DREAM ] : CUDA_VISIBLE_DEVICES=${device} python run.py --model DREAM --dname ${dname} --p-train ${p_train} --p-val ${p_val} --p-test ${p_test} --seed ${seed} --l2 ${l2} --lr ${lr} --layer ${layer} --pred-layer ${pred_layer} --cs ${cs} --max-epoch ${max_epoch} --patience ${patience} --max-d ${max_d} --max-char ${max_char} --bs ${bs} --h-dim ${h_dim} --es ${es} --clip-gr ${clip_gr}${suffix}
  if [[ $preview -eq 0 ]]; then
    CUDA_VISIBLE_DEVICES=${device} python run.py --model DREAM --dname ${dname} --p-train ${p_train} --p-val ${p_val} --p-test ${p_test} --seed ${seed} --l2 ${l2} --lr ${lr} --layer ${layer} --pred-layer ${pred_layer} --cs ${cs} --max-epoch ${max_epoch} --patience ${patience} --max-d ${max_d} --max-char ${max_char} --bs ${bs} --h-dim ${h_dim} --es ${es} --clip-gr ${clip_gr}${suffix}
  fi
}

# PDREAM subprocedure
function PDREAM_sub {
  echo [PDREAM ] : CUDA_VISIBLE_DEVICES=${device} python run.py --model DREAM --dname ${dname} --p-train ${p_train} --p-val ${p_val} --p-test ${p_test} --seed ${seed} --l2 ${l2} --lr ${lr}${swa} --layer ${layer} --pred-layer ${pred_layer} --cs ${cs} --max-epoch ${max_epoch} --patience ${patience} --max-d ${max_d} --max-char ${max_char} --sep-emb --prfx --bs ${bs} --h-dim ${h_dim} --es ${es} --clip-gr ${clip_gr}${suffix}
  if [[ $preview -eq 0 ]]; then
    CUDA_VISIBLE_DEVICES=${device} python run.py --model DREAM --dname ${dname} --p-train ${p_train} --p-val ${p_val} --p-test ${p_test} --seed ${seed} --l2 ${l2} --lr ${lr}${swa} --layer ${layer} --pred-layer ${pred_layer} --cs ${cs} --max-epoch ${max_epoch} --patience ${patience} --max-d ${max_d} --max-char ${max_char} --sep-emb --prfx --bs ${bs} --h-dim ${h_dim} --es ${es} --clip-gr ${clip_gr}${suffix}
  fi
}

# Ernn subprocedure
function Ernn_sub {
  echo [Ernn ] : CUDA_VISIBLE_DEVICES=${device} python run.py --model DREAM --dname ${dname} --p-train ${p_train} --p-val ${p_val} --p-test ${p_test} --seed ${seed} --l2 ${l2} --lr ${lr}${swa} --layer ${layer} --pred-layer ${pred_layer} --cs ${cs} --max-epoch ${max_epoch} --patience ${patience} --max-d ${max_d} --max-char ${max_char} --sep-emb --Eprfx --bs ${bs} --h-dim ${h_dim} --es ${es} --clip-gr ${clip_gr}${suffix}
  if [[ $preview -eq 0 ]]; then
    CUDA_VISIBLE_DEVICES=${device} python run.py --model DREAM --dname ${dname} --p-train ${p_train} --p-val ${p_val} --p-test ${p_test} --seed ${seed} --l2 ${l2} --lr ${lr}${swa} --layer ${layer} --pred-layer ${pred_layer} --cs ${cs} --max-epoch ${max_epoch} --patience ${patience} --max-d ${max_d} --max-char ${max_char} --sep-emb --Eprfx --bs ${bs} --h-dim ${h_dim} --es ${es} --clip-gr ${clip_gr}${suffix}
  fi
}

# CardNet subprocedure
function card_sub {
  echo [card ]: CUDA_VISIBLE_DEVICES=${device} python run.py --model CardNet --dname ${dname} --p-train ${p_train} --p-val ${p_val} --p-test ${p_test} --seed ${seed} --l2 ${l2} --lr ${clr} --vlr ${vlr} --csc ${csc} --vsc ${vsc} --max-epoch ${cmax_epoch} --patience ${patience} --max-d ${max_d} --max-char ${max_char} --bs ${cbs} --vbs ${vbs} --max-epoch-vae ${cmax_epoch_vae} --vl2 ${vl2} --vclip-lv ${vclip_lv} --vclip-gr ${vclip_gr} --clip-gr ${clip_gr}${suffix}
  if [[ $preview -eq 0 ]]; then
    CUDA_VISIBLE_DEVICES=${device} python run.py --model CardNet --dname ${dname} --p-train ${p_train} --p-val ${p_val} --p-test ${p_test} --seed ${seed} --l2 ${l2} --lr ${clr} --vlr ${vlr} --csc ${csc} --vsc ${vsc} --max-epoch ${cmax_epoch} --patience ${patience} --max-d ${max_d} --max-char ${max_char} --bs ${cbs} --vbs ${vbs} --max-epoch-vae ${cmax_epoch_vae} --vl2 ${vl2} --vclip-lv ${vclip_lv} --vclip-gr ${vclip_gr} --clip-gr ${clip_gr}${suffix}
  fi
}

# PCardNet subprocedure
function PCardNet_sub {
  echo [PCardNet]: CUDA_VISIBLE_DEVICES=${device} python run.py --model CardNet --dname ${dname} --p-train ${p_train} --p-val ${p_val} --p-test ${p_test} --seed ${seed} --l2 ${l2} --lr ${clr} --vlr ${vlr} --csc ${csc} --vsc ${vsc} --max-epoch ${cmax_epoch} --patience ${patience} --max-d ${max_d} --max-char ${max_char} --Eprfx --bs ${cbs} --vbs ${vbs} --max-epoch-vae ${cmax_epoch_vae} --vl2 ${vl2} --vclip-lv ${vclip_lv} --vclip-gr ${vclip_gr} --clip-gr ${clip_gr}${suffix}
  if [[ $preview -eq 0 ]]; then
    CUDA_VISIBLE_DEVICES=${device} python run.py --model CardNet --dname ${dname} --p-train ${p_train} --p-val ${p_val} --p-test ${p_test} --seed ${seed} --l2 ${l2} --lr ${clr} --vlr ${vlr} --csc ${csc} --vsc ${vsc} --max-epoch ${cmax_epoch} --patience ${patience} --max-d ${max_d} --max-char ${max_char} --Eprfx --bs ${cbs} --vbs ${vbs} --max-epoch-vae ${cmax_epoch_vae} --vl2 ${vl2} --vclip-lv ${vclip_lv} --vclip-gr ${vclip_gr} --clip-gr ${clip_gr}${suffix}
  fi
}

# LBS subprocedure
function LBS_sub {
  echo [ LBS ]: python run.py --model LBS --dname ${dname} --p-test ${p_test} --seed ${seed} --Ntbl ${Ntbl} --PT ${PT} --max-d ${max_d} --L ${L}${suffix}
  if [[ $preview -eq 0 ]]; then
    python run.py --model LBS --dname ${dname} --p-test ${p_test} --seed ${seed} --Ntbl ${Ntbl} --PT ${PT} --max-d ${max_d} --L ${L}${suffix}
  fi
}

function rnn_default {
  local l2
  local clip_gr
  set_rnn
  rnn_sub
}
function PDREAM_default {
  local l2
  local clip_gr
  set_PDREAM
  PDREAM_sub
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
function PCardNet_default {
  local l2
  local clip_gr
  set_PCardNet
  PCardNet_sub
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

function for_default_PDREAM {
  local seed
  local dname
  for seed in $seeds; do
    for dname in $dnames; do
      PDREAM_sub
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
function for_default_PCardNet {
  local seed
  local dname
  for seed in $seeds; do
    for dname in $dnames; do
      PCardNet_sub
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

function for_data_size_PDREAM {
  local seed
  local dname
  local p_train
  for seed in $seeds; do
    for dname in $dnames; do
      for p_train in $p_trains; do
        PDREAM_sub
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

function for_data_size_PCardNet {
  local seed
  local dname
  local p_train
  for seed in $seeds; do
    for dname in $dnames; do
      for p_train in $p_trains; do
        PCardNet_sub
      done
    done
  done
}

function for_model_size_PDREAM {
  local seed
  local dname
  local cs
  local h_dim
  dname='DBLP'
  for seed in $seeds; do
    for cs in $css; do
      h_dim=${cs}
      PDREAM_sub
    done
  done
}

function for_model_size_PCardNet {
  local seed
  local dname
  local csc
  local vsc
  dname='DBLP'
  for seed in $seeds; do
    for csc in $cscs; do
      vsc=$((csc / 2))
      PCardNet_sub
    done
  done
}

function for_model_size_LBS {
  local seed
  local dname
  local PT
  dname='DBLP'
  for seed in $seeds; do
    for PT in $PTs; do
      LBS_sub
    done
  done
}

function for_analysis_rnn {
  local suffix
  local dname
  local seed
  suffix=" --analysis"

  for seed in $seeds; do
    for dname in $dnames; do
      rnn_sub
    done
  done
}

function for_analysis_PDREAM {
  local suffix
  local dname
  local seed
  suffix=" --analysis"

  for seed in $seeds; do
    for dname in $dnames; do
      PDREAM_sub
    done
  done
}

function for_analysis_Ernn {
  local suffix
  local dname
  local seed
  suffix=" --analysis"

  for seed in $seeds; do
    for dname in $dnames; do
      Ernn_sub
    done
  done
}

function for_analysis_card {
  local suffix
  local dname
  local seed
  suffix=" --analysis"

  for seed in $seeds; do
    for dname in $dnames; do
      card_sub
    done
  done
}

function for_analysis_PCardNet {
  local suffix
  local dname
  local seed
  suffix=" --analysis"

  for seed in $seeds; do
    for dname in $dnames; do
      PCardNet_sub
    done
  done
}

function for_analysis_LBS {
  local suffix
  local dname
  local seed
  suffix=" --analysis"

  for seed in $seeds; do
    for dname in $dnames; do
      LBS_sub
    done
  done
}

function rnn_all {
  local l2
  local clip_gr
  set_rnn

  for_default_rnn
  for_analysis_rnn
  # for_data_size_rnn
  echo
}

function PDREAM_all {
  local l2
  local clip_gr
  set_PDREAM

  for_default_PDREAM
  for_analysis_PDREAM
  # for_data_size_PDREAM
  # for_model_size_PDREAM
  echo
}

function Ernn_all {
  local l2
  local clip_gr
  set_Ernn

  for_default_Ernn
  for_analysis_Ernn
  echo
}

function card_all {
  local l2
  local clip_gr
  set_card

  for_default_card
  for_analysis_card
  echo
}

function PCardNet_all {
  local l2
  local clip_gr
  set_PCardNet

  for_default_PCardNet
  for_analysis_PCardNet
  # for_data_size_PCardNet
  # for_model_size_PCardNet
  echo
}

function LBS_all {
  for_default_LBS
  for_analysis_LBS
  # for_model_size_LBS
  echo
}

function run_all {
  rnn_all
  PDREAM_all
  Ernn_all
  card_all
  PCardNet_all
  LBS_all
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  if [[ $preview -ne 0 ]]; then
    rnn_default
    PDREAM_default
    Ernn_default
    card_default
    PCardNet_default
    LBS_default
    echo
    run_all
  fi
fi
