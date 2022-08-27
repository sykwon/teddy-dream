import argparse
from pyexpat import model

from src.LBS import LBS
from src.estimator import learn_Deep_Learning_model
from src.preprocess import load_query_strings, get_cardinalities_train_test_valid

from src.model_torch import DREAMEstimator, CardNetEstimator
from src.util import is_learning_model, ConfigManager
import src.util as ut
import sys
import os
import redis
import json

train_qry_outdir = "data/"
logger = None  # logger is disabled

est_class_dict = {
    "DREAM": DREAMEstimator,
    "CardNet": CardNetEstimator,
    "LBS": LBS,
}


def get_exp_name(args):
    _,  exp_name = get_config_manager_and_exp_name(args)
    return exp_name


def get_config_manager_and_exp_name(args):
    # ---- start applying args to cm ---- #
    cm = ConfigManager()
    exp_suffix = ""
    model_name = args.model
    cm['alg']['name'] = model_name
    dname = args.dname

    if model_name == 'DREAM':
        assert args.cs, "In RNN training, cell size should be given"
        cm["alg"]["cs"] = args.cs
        exp_suffix += f"_cs_{args.cs}"

        cm["alg"]["layer"] = args.layer
        exp_suffix += f"_layer_{args.layer}"

        cm["alg"]["pred_layer"] = args.pred_layer
        exp_suffix += f"_predL_{args.pred_layer}"

        if args.prfx:
            cm["alg"]["prfx"] = args.prfx
            exp_suffix += f"_prfx"

        if args.btS:
            cm["alg"]["btS"] = args.btS
            exp_suffix += f"_btS"

        if args.btA:
            cm["alg"]["btA"] = args.btA
            exp_suffix += f"_btA"

        if args.Eprfx:
            exp_suffix += f"_Eprfx"

        cm["alg"]["h_dim"] = args.h_dim
        exp_suffix += f"_hDim_{args.h_dim}"

        cm["alg"]["es"] = args.es
        exp_suffix += f"_es_{args.es}"

    if model_name == 'card':
        cm["alg"]["csc"] = args.csc
        exp_suffix += f"_csc_{args.csc}"
        cm["alg"]["vsc"] = args.vsc
        exp_suffix += f"_vsc_{args.vsc}"
        if args.Eprfx:
            exp_suffix += f"_Eprfx"
        cm["alg"]["vl2"] = args.vl2
        exp_suffix += f"_vl2_{args.vl2}"
        cm["alg"]["vclip_lv"] = args.vclip_lv
        exp_suffix += f"_vclipLv_{args.vclip_lv}"
        cm["alg"]["vclip_gr"] = args.vclip_gr
        exp_suffix += f"_vclipGr_{args.vclip_gr}"
        cm["alg"]["clip_gr"] = args.clip_gr
        exp_suffix += f"_clipGr_{args.clip_gr}"

    # Learning models
    if is_learning_model(model_name):
        if args.lr:
            cm["alg"]["lr"] = args.lr
            exp_suffix += f"_lr_{args.lr}"
        if args.vlr:
            assert model_name == 'card'
            cm["alg"]["vlr"] = args.vlr
            exp_suffix += f"_vlr_{args.vlr}"
        if args.max_char:
            cm["alg"]["max_char"] = args.max_char
            exp_suffix += f"_maxC_{args.max_char}"
        if args.p_val is not None:
            exp_suffix += f"_pVal_{args.p_val}"
        if args.p_train is not None:
            cm["alg"]["p_train"] = args.p_train
            exp_suffix += f"_ptrain_{args.p_train}"
        if args.l2 is not None:
            cm["alg"]["l2"] = args.l2
            exp_suffix += f"_l2_{args.l2}"

        if args.patience is not None:
            cm["alg"]["patience"] = args.patience
            exp_suffix += f"_pat_{args.patience}"
        if model_name == 'DREAM':
            cm["alg"]["clip_gr"] = args.clip_gr
            exp_suffix += f"_clipGr_{args.clip_gr}"

    # LBS models
    if model_name in 'LBS':
        cm["alg"]["N"] = args.Ntbl
        exp_suffix += f"_N_{args.Ntbl}"
        cm["alg"]["PT"] = args.PT
        exp_suffix += f"_PT_{args.PT}"
        cm["alg"]["L"] = args.L
        exp_suffix += f"_L_{args.L}"

    cm["alg"]["seed"] = args.seed
    exp_suffix += f"_seed_{args.seed}"
    cm["data"]["name"] = args.dname

    if args.max_epoch:
        cm["alg"]["max_epoch"] = args.max_epoch
        exp_suffix += f"_maxEpoch_{args.max_epoch}"

    cm["alg"]["max_d"] = args.max_d
    exp_suffix += f"_maxD_{args.max_d}"
    if args.p_test:
        exp_suffix += f"_pTest_{args.p_test}"

    if args.bs:
        cm["alg"]["bs"] = args.bs
        exp_suffix += f"_bs_{args.bs}"
    if args.vbs:
        cm["alg"]["vbs"] = args.vbs
        exp_suffix += f"_vbs_{args.vbs}"
    if args.max_epoch_vae:
        cm["alg"]["max_epoch_vae"] = args.max_epoch_vae
        exp_suffix += f"_maxEpochVae_{args.max_epoch_vae}"

    exp_name = f"{model_name}_{dname}"
    exp_name += exp_suffix
    cm['save']['exp_name'] = exp_name
    # ------ end applying args to cm ------ #
    assert len(exp_name) <= 256, len(exp_name)
    return cm, exp_name


if __name__ == "__main__":
    args, exp_key = ut.get_model_args(verbose=True)
    ut.varify_args(args)
    print("[args]:", args)

    conn = redis.Redis()
    # check if this experiment already done
    if not args.analysis:
        redis_value_model = conn.get(f"model:{exp_key}")
        redis_value_est = conn.get(f"est:{exp_key}")
        redis_value_conf = conn.get(f"conf:{exp_key}")

        if not args.overwrite:
            if redis_value_est is not None and redis_value_model is not None:
                print("This experiment is already done")
                print(
                    f"already exists in redis \n[key]: est:{exp_key} \n[model_val]: {redis_value_model} \n[est_val]: {redis_value_est}")
                if redis_value_conf is not None:
                    print(f"[conf_val]:{redis_value_conf}")
                exit()
            else:
                args.overwrite = True

    cm, exp_name = get_config_manager_and_exp_name(args)
    print(cm)
    print("[exp_key ]:", exp_key)
    print("[exp_name]:", exp_name)

    print("save pair (exp_key, exp_name)")
    args_dict_str = json.dumps(args.__dict__)
    if not args.analysis:
        conn.set(f"conf:{exp_key}", exp_name)
        conn.set(f"args:{exp_key}", args_dict_str)
    else:
        print(f"conf:{exp_key}")
        exp_name_saved = conn.get(f"conf:{exp_key}").decode()
        print("saved:", exp_name_saved)
        if exp_name_saved != exp_name:
            print(f"warning: exp_name mismatch \nsaved: {exp_name_saved} \ncurr : {exp_name}")
            exp_name = exp_name_saved

    # ------- prepare train and test dataset ----- #
    split_seed = 0
    query_strings = load_query_strings(args.dname, seed=split_seed)
    q_train, q_valid, q_test = ut.get_splited_train_valid_test_each_len(query_strings, split_seed, args)

    if args.outdata:
        ut.store_train_valid_query_string(q_train, q_valid, train_qry_outdir, args)
        exit()

    train_data, valid_data, test_data = get_cardinalities_train_test_valid(q_train, q_valid, q_test, split_seed, args)
    data = []
    if is_learning_model(args.model):
        data.append(train_data)
    data.extend([valid_data, test_data])

    n_qry, n_prfx, n_update = ut.get_stat_query_string(q_train, args)
    Est_class = est_class_dict[args.model]

    # start training
    est = learn_Deep_Learning_model(cm, Est_class, data, is_rewirte=args.rewrite, logger=logger, exp_key=exp_key,
                                    overwrite=args.overwrite, n_qry=n_qry, n_prfx=n_prfx, n_update=n_update, analysis=args.analysis)
