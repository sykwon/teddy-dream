import string
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import redis
import itertools
import argparse

# global variables
conn = redis.Redis()  # db connector
str_formatter = string.Formatter()
_latex_dir = "figures/"
_tbl_dir = "tables/"
_confirm = False
_verbose_fig = True
_verbose_tbl = False
_save_table = False

# default values
_dataNames = []
_algs = []
_xvals = []
_xvals_dict = {}
_xlabel = ""
_xlabel_dict = {}
_ylabel = ""
_ylabels = []
_zlabel = None
_zlabel_dict = None
_default_map = {}
_default_map_dict = {}
_seeds = []
_linewidth = 0.5
_markerwidth = 0.5

# plot configs
_figsize = (3, 1.6)  # 5:2.5 2:1
_zfontsize = 8
_zalignments = None
_zoffsets = None
_xlog = False
_ylog = False
_xlim = ()
_ylim = ()
_xticks = None
_yticks = None
_xticks_format = None
_yticks_format = None
_legend = True
_dpi = 400
_clip_on = False


class identity_dict(dict):
    def __missing__(self, key):
        return "".join(key.split("_"))


class default_dict(dict):
    def set_default(self, val):
        self.my_val = val

    def __missing__(self, key):
        return self.my_val


class format_dict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


# name mapping
alg_label_dict = identity_dict({
    # model
    "PDREAM": "DREAM",
    "DREAM": "DREAM",
    "LBS": "LBS",
    "card": "CardNet",
    "PCardNet": "CardNet",
    "PAstrid": "Astrid",

    # gen
    "NaiveGen": "NaiveGen",
    "Qgram": "Qgram",
    "TASTE": "TASTE",
    "SODDY": "SODDY",
    "TEDDY": "TEDDY",
    "Tallp": "NaiveGen",
    "TQgram": "Qgram",
    "TTASTE": "TASTE",
    "TSODDY2": "SODDY",
    "TTEDDY2ttt": "TEDDY",
})

table_label_dict = identity_dict({
    "est:err": "Avg.",
    "anal:err": "Avg.",
    "anal:q10": "10th",
    "anal:q20": "20th",
    "anal:q30": "30th",
    "anal:q40": "40th",
    "anal:q50": "50th",
    "anal:q60": "60th",
    "anal:q70": "70th",
    "anal:q80": "80th",
    "anal:q90": "90th",
    "anal:q25": "25th",
    "anal:q75": "75th",
    "est:q90": "90th",
    "anal:q90": "90th",
    "anal:q95": "95th",
    "anal:q99": "99th",
    "anal:q100": "Max.",
    "model:time": "time",
    "model:time(m)": "Time(min)",
    "model:time(h)": "Time",
    "n_qry": "# Train data",
    "gen_time": "time",
})

param_label_dict = identity_dict({
    "pq": "Sampling ratio",
    "p_train": r"Train data size (\%)",
    "pTrain": r"Train data size (\%)",
    "time": "Exe. time (sec)",
    "thrs": r"$\delta_M$",
    "delta": r"$\delta_M$",
    "size": "model size",
    "pq": r"Train data size (\%)",
    "join:time": "Generation time (sec)",
    "size": "model size",
    "est:err": "Q-error",
    "qerr": "Q-error",
    "quantile": "Quantile",
    "percentile": "Percentile",
    "model:time": "Training time (sec)",
    "model:query": "Number of query strings",
    "model:prefix": "Number of query strings",
    "model:size": "Model size",
    "est:time": "Est. time (sec)",
    "maxl": r"Space Budget (\%)",
    "n_qry": "Number of queries",
    "n_qs": "Number of query strings",
    "length": "Query length",
    "est_time": "Est. time (sec)",
})

data_label_dict = identity_dict({
    "WIKI": "WIKI",
    "IMDB": "IMDB",
    "DBLP": "DBLP",
    "GENE": "GENE",
})


arg_default_map_dict = {
    'PAstrid': {
        'dname': "WIKI",
        'delta': -3,
        'p_train': 1.0,
        'seed': 0,
        'es': 512,
        'bs': 2048,
        'lr': 0.001,
        'epoch': 64,
        'emb_epoch': 8,
        'dsc': 1024,
    },
    'DREAM': {
        'model': 'DREAM',
        'dname': 'WIKI',
        'p_train': 1.0,
        'p_val': 0.1,
        'p_test': 0.1,
        'seed': 0,
        'l2': 1e-8,
        'lr': 0.001,
        'layer': 1,
        'pred_layer': 3,
        'cs': 512,
        'max_epoch': 100,
        'patience': 5,
        'max_d': 3,
        'max_char': 200,
        'prfx': False,
        'Eprfx': False,
        'bs': 32,
        'h_dim': 512,
        'es': 100,
        'clip_gr': 10.0,
    },
    'PDREAM': {
        'model': 'DREAM',
        'dname': 'WIKI',
        'p_train': 1.0,
        'p_val': 0.1,
        'p_test': 0.1,
        'seed': 0,
        'l2': 1e-8,
        'lr': 0.001,
        'layer': 1,
        'pred_layer': 3,
        'cs': 512,
        'max_epoch': 100,
        'patience': 5,
        'max_d': 3,
        'max_char': 200,
        'prfx': True,
        'lr': 0.001,
        'bs': 32,
        'h_dim': 512,
        'es': 100,
        'clip_gr': 10.0,
    },
    'EDREAM': {
        'model': 'DREAM',
        'dname': 'WIKI',
        'p_train': 1.0,
        'p_val': 0.1,
        'p_test': 0.1,
        'seed': 0,
        'l2': 1e-8,
        'lr': 0.001,
        'layer': 1,
        'pred_layer': 3,
        'cs': 512,
        'max_epoch': 100,
        'patience': 5,
        'max_d': 3,
        'max_char': 200,
        'Eprfx': True,
        'lr': 0.001,
        'bs': 32,
        'h_dim': 512,
        'es': 100,
        'clip_gr': 10.0,
    },
    'PCardNet': {
        'dname': 'WIKI',
        'p_train': 1.0,
        'p_val': 0.1,
        'p_test': 0.1,
        'seed': 0,
        'l2': 0.00000001,
        'lr': 0.001,
        'vlr': 0.001,
        'csc': 512,
        'vsc': 256,
        'max_epoch': 800,
        'patience': 5,
        'Eprfx': True,
        'max_d': 3,
        'max_char': 200,
        'bs': 256,
        'vbs': 256,
        'max_epoch_vae': 100,
        'vl2': 0.01,
        'vclip_lv': 10.0,
        'vclip_gr': 0.01,
        'clip_gr': 10.0,
    },
    'CardNet': {
        'dname': 'WIKI',
        'p_train': 1.0,
        'p_val': 0.1,
        'p_test': 0.1,
        'seed': 0,
        'l2': 1e-8,
        'lr': 0.001,
        'vlr': 0.001,
        'csc': 512,
        'vsc': 256,
        'max_epoch': 800,
        'patience': 5,
        'Eprfx': False,
        'max_d': 3,
        'max_char': 200,
        'bs': 256,
        'vbs': 256,
        'max_epoch_vae': 100,
        'vl2': 0.01,
        'vclip_lv': 10.0,
        'vclip_gr': 0.01,
        'clip_gr': 10.0,
    },
    'LBS': {
        'model': 'LBS',
        'dname': 'WIKI',
        'p_test': 0.1,
        'seed': 0,
        'Ntbl': 5,
        'PT': 20,
        'max_d': 3,
        'L': 10,
    },
    'TEDDY': {
        'pq': 1.0,
        'pTrain': 1.0,
        'thrs': 3,
        'prfx': 0,
    }
}

arg_str_format_dict = {
    'join': "{alg} data/{dataName}.txt data/qs_{dataName}.txt {thrs} {prfx} 0",
    'DREAM': "--model DREAM --dname {dataName}",
    'CardNet': "--model CardNet --dname {dataName}",
    'LBS': "--model LBS --dname {dataName} --seed {seed} --Ntbl {Ntbl} --PT {PT}",
    'Astrid': "--dname {dataName} --delta {delta} --p-train {pTrain} --seed {seed} --es {es} --bs {bs} --lr {lr} --epoch {epoch}",
}

arg_model_format_dict = {
    'DREAM': "DREAM_{dataName}_max_cs_{cs}_layer_1_predL_{predLayer}",
    'CardNet': "CardNET_{dataName}_max_csc_{csc}_vsc_{vsc}",
    'LBS': "L_10_N_{Ntbl}_PT_{PT}_n_*_name_{dataName}_seed_{seed}.bin",
    'Astrid': "",
}

model_dir_dict = {
    'DREAM': "../dream/model/",
    'PDREAM': "../dream/model/",
    'CardNet': "../dream/model/",
    'PCardNet': "../dream/model/",
    'LBS': "../dream/pkl/load_extended_ngram_table_hash/",
    'Astrid': "../astrid/log/",
    'PAstrid': "../astrid/log/",
}

alg_style_dict = {
    'NaiveGen': {'marker': '^'},
    'Qgram': {'marker': '+', 'linestyle': "dashed"},
    'TASTE': {'marker': 'D'},
    'SODDY': {'marker': 'x', 'linestyle': "dotted"},
    'TEDDY': {'marker': 'o'},
}


def percent_foramt_from_ratio(number):
    output = number
    output = output * 100
    output = f"{output:f}"
    output = output.rstrip('0').rstrip('.')
    return output


def unpack_dict(input_dict):
    list_valued_dict = {}
    for key, val in input_dict.items():
        if isinstance(val, list):
            list_valued_dict[key] = val
        else:
            list_valued_dict[key] = [val]
    keys = list(list_valued_dict.keys())
    vals = [list_valued_dict[k] for k in list_valued_dict.keys()]
    output = []
    for tuple_value in itertools.product(*vals):
        elem_dict = {}
        for key, val in zip(keys, tuple_value):
            elem_dict[key] = val
        output.append(elem_dict)
    return output


def get_arg_str_pat(alg, default_map):
    if "DREAM" in alg:
        arg_str_pat = arg_str_format_dict["DREAM"]
        if "p_train" in default_map:
            arg_str_pat += " --p-train {p_train}"
        if "p_val" in default_map:
            arg_str_pat += " --p-val {p_val}"
        if "p_test" in default_map:
            arg_str_pat += " --p-test {p_test}"
        if "seed" in default_map:
            arg_str_pat += " --seed {seed}"
        if "l2" in default_map:
            arg_str_pat += " --l2 {l2}"
        if "lr" in default_map:
            arg_str_pat += " --lr {lr}"
        if "layer" in default_map:
            arg_str_pat += " --layer {layer}"
        if "pred_layer" in default_map:
            arg_str_pat += " --pred-layer {pred_layer}"
        if "cs" in default_map:
            arg_str_pat += " --cs {cs}"
        if "max_epoch" in default_map:
            arg_str_pat += " --max-epoch {max_epoch}"
        if "patience" in default_map:
            arg_str_pat += " --patience {patience}"
        if "max_d" in default_map:
            arg_str_pat += " --max-d {max_d}"
        if "max_char" in default_map:
            arg_str_pat += " --max-char {max_char}"
        if alg == 'EDREAM':
            arg_str_pat += " --Eprfx"
        elif "PDREAM" in alg:
            arg_str_pat += " --prfx"
        if "bs" in default_map:
            arg_str_pat += " --bs {bs}"
        if "h_dim" in default_map:
            arg_str_pat += " --h-dim {h_dim}"
        if "es" in default_map:
            arg_str_pat += " --es {es}"
        if "clip_gr" in default_map:
            arg_str_pat += " --clip-gr {clip_gr}"
    elif "CardNet" in alg:
        arg_str_pat = arg_str_format_dict["card"]
        if "p_train" in default_map:
            arg_str_pat += " --p-train {p_train}"
        if "p_val" in default_map:
            arg_str_pat += " --p-val {p_val}"
        if "p_test" in default_map:
            arg_str_pat += " --p-test {p_test}"
        if "seed" in default_map:
            arg_str_pat += " --seed {seed}"
        if "l2" in default_map:
            arg_str_pat += " --l2 {l2}"
        if "lr" in default_map:
            arg_str_pat += " --lr {lr}"
        if "vlr" in default_map:
            arg_str_pat += " --vlr {vlr}"
        if "csc" in default_map:
            arg_str_pat += " --csc {csc}"
        if "vsc" in default_map:
            arg_str_pat += " --vsc {vsc}"
        if "max_epoch" in default_map:
            arg_str_pat += " --max-epoch {max_epoch}"
        if "patience" in default_map:
            arg_str_pat += " --patience {patience}"
        if "max_d" in default_map:
            arg_str_pat += " --max-d {max_d}"
        if "max_char" in default_map:
            arg_str_pat += " --max-char {max_char}"
        if "PCardNet" in alg:
            arg_str_pat += " --Eprfx"
        if "bs" in default_map:
            arg_str_pat += " --bs {bs}"
        if "vbs" in default_map:
            arg_str_pat += " --vbs {vbs}"
        if "max_epoch_vae" in default_map:
            arg_str_pat += " --max-epoch-vae {max_epoch_vae}"
    elif "LBS" in alg:
        arg_str_pat = arg_str_format_dict["LBS"]
    elif "Astrid" in alg:
        arg_str_pat = arg_str_format_dict["Astrid"]
    else:
        assert any([join_name in alg for join_name in ['NaiveGen', 'TASTE', 'Qgram', 'SODDY', 'TEDDY', 'TEDDY-R']])
        arg_str_pat = arg_str_format_dict['join']
        if "pTrain" in default_map:
            arg_str_pat = arg_str_pat.replace("data/qs_{dataName}.txt",
                                              "data/qs_{dataName}_{pTrain}.txt")

        if "P" == alg[0]:
            alg = alg[1:]
        arg_str_pat = str_formatter.vformat(arg_str_pat, (), format_dict({'alg': alg}))
    return arg_str_pat


def get_arg_str_list(seed, dataName, alg, default_map):
    arg_str_pat = get_arg_str_pat(alg, default_map)
    arg_str_list = []
    for default_map_elem in unpack_dict(default_map):
        # print(arg_str_pat)
        # arg_str = arg_str_pat.format(**{'seed': seed, 'dataName': dataName, xlabel: xval, **default_map_elem})
        arg_str = arg_str_pat.format(**{'seed': seed, 'dataName': dataName, **default_map_elem})
        # print(arg_str)
        arg_str_list.append(arg_str)
    return arg_str_list


def get_distinct_prefixes(strings):
    prfx_set = set()
    for string in strings:
        for length in range(1, len(string)+1):
            prfx = string[:length]
            prfx_set.add(prfx)
    return list(sorted(prfx_set))


def print_strings_stat(strings, name=""):
    strings_len = [len(x) for x in strings]
    min_len = min(strings_len)
    avg_len = sum(strings_len) / len(strings_len)
    max_len = max(strings_len)
    print(
        f"[min, avg, max, N] {name} len: [{min_len}, {avg_len:.1f}, {max_len}, {len(strings)}]")
    return min_len, avg_len, max_len, len(strings)


def slice_column_df(df, source="alg", target=["alg", "delta"], lengths=[6, 1]):
    assert len(target) == len(lengths)
    elem_lengths = [len(x) for x in df[source]]
    assert all([elem_lengths[0] == x for x in elem_lengths])
    assert sum(lengths) == elem_lengths[0]
    start_pts = []
    end_pts = []
    pnt = 0
    for length in lengths:
        start_pts.append(pnt)
        pnt += length
        end_pts.append(pnt)

    for t_name, df_column in zip(target, list(map(df[source].str.slice, start_pts, end_pts))):
        df[t_name] = df_column
    return df


def aggregate_df(df, target="seed", agg="mean", counting=False, main_keys=None):
    """

    """
    df: pd.DataFrame = df
    if len(df) == 0:
        return df
    if main_keys is None:
        main_keys = ["seed", "data", "alg", "xlabel", "xval"]
    if target in main_keys:
        main_keys.remove(target)
    # df['agg:count']

    if target is not None:
        df = df.drop(columns=target)
    df_output = df.groupby(by=main_keys).agg(agg).reset_index(list(range(len(main_keys))))
    if isinstance(agg, (list,)):
        df_output.columns = [':'.join(col).rstrip(agg[0]).rstrip(":") for col in df_output.columns]
    if counting:
        df_count = df.groupby(by=main_keys).size().reset_index(name="counts")
        df_output['counts'] = df_count['counts']
    return df_output


def slice_df_by_dataName(df):
    global _dataNames
    dataNames = _dataNames

    df_list = []
    for dataName in dataNames:
        slice_df = df[df['data'] == dataName]
        df_list.append(slice_df)
    return df_list


def slice_df_by_column_name(df, column_name):
    df_list = []
    column_values = df[column_name].unique()
    for cv in column_values:
        slice_df = df[df[column_name] == cv]
        df_list.append(slice_df)
    return df_list


def clipping_df(df, target, max_val=None, min_val=None):
    if max_val is not None:
        df = df.drop(df[df[target] > max_val].index)

    if min_val is not None:
        df = df.drop(df[df[target] < min_val].index)
    return df


def union_df(df_list, ignore_index=True):
    union_df = pd.concat(df_list, ignore_index=ignore_index)
    return union_df


def assign_dname_to_query_dict(query_dict, dname, alg):
    if dname is not None:
        query_dict['dname'] = dname
    return query_dict


def get_default_exp_keys(alg, dname=None):
    query_dict = arg_default_map_dict[alg].copy()
    query_dict = assign_dname_to_query_dict(query_dict, dname, alg)

    exp_keys = get_matched_exp_keys(query_dict)
    return exp_keys


def get_df_model(verbose_level=2):
    # global connector is used
    global _seeds
    seeds = _seeds
    global _dataNames
    dataNames = _dataNames
    global _algs
    algs = _algs
    global _xvals
    global _xvals_dict
    xvals_dict = default_dict(_xvals_dict)
    xvals_dict.set_default(_xvals)
    global _xlabel
    global _xlabel_dict
    xlabel_dict = default_dict(_xlabel_dict)
    xlabel_dict.set_default(_xlabel)
    global _ylabels
    ylabels = _ylabels

    global _default_map
    global _default_map_dict
    default_map_dict = default_dict(_default_map_dict)
    default_map_dict.set_default(_default_map)

    # status
    # fail: 0
    # start: 1
    # done: 2
    # timeout: 3

    header = ["seed", "data", "alg", "xlabel", "xval", *ylabels]
    records = []
    # print(algs)
    for seed in seeds:
        for dataName in dataNames:
            # print(dataName)
            for alg in algs:
                # print(alg)
                xvals = xvals_dict[alg]
                xlabel = xlabel_dict[alg]
                default_map = default_map_dict[alg].copy()
                default_map['seed'] = seed
                default_map['dname'] = dataName

                for xval in xvals:
                    if isinstance(xlabel, tuple):
                        assert len(xlabel) == len(xval)
                        for lab, v in zip(xlabel, xval):
                            default_map[lab] = v
                    else:
                        default_map[xlabel] = xval
                    # print(default_map)
                    exp_keys = get_matched_exp_keys(default_map)
                    if len(exp_keys) == 0:
                        print(default_map, "non-matched")
                    assert len(exp_keys) <= 1, (default_map, exp_keys)
                    if len(exp_keys) == 0:
                        if verbose_level >= 2:
                            print(default_map)
                        continue
                    exp_key = exp_keys[0]
                    if verbose_level >= 2:
                        # print(default_map)
                        # print(exp_keys)
                        print(exp_key)
                    rec = [seed, dataName, alg, xlabel, xval]
                    is_fail = False
                    for ylabel in ylabels:
                        assert ":" in ylabel
                        tbl, exp_label = ylabel.split(":")
                        redis_key = tbl + ":" + exp_key
                        exp_res_byte = conn.get(redis_key)
                        if exp_res_byte is None:
                            if verbose_level >= 1:
                                print("[fail:NN]:", redis_key, "[label]:", ylabel)
                            is_fail = True
                            break
                        else:
                            if tbl == 'status':
                                yval = int(exp_res_byte)
                            else:
                                if "Traceback" in exp_res_byte.decode():
                                    if verbose_level >= 1:
                                        print("[fail:TB]:", redis_key)
                                    is_fail = True
                                    break
                                exp_res_dict = json.loads(exp_res_byte)
                                if exp_label not in exp_res_dict:
                                    if verbose_level >= 1:
                                        print(f"[fail:NI][key={exp_label}]:", redis_key)
                                    is_fail = True
                                    break
                                yval = exp_res_dict[exp_label]
                            rec.append(yval)
                    if not is_fail:
                        records.append(rec)

    df = pd.DataFrame(records, columns=header)
    return df


def get_df(verbose_level=2, **kwargs):
    # global connector is used
    global _seeds
    seeds = _seeds
    global _dataNames
    dataNames = _dataNames
    global _algs
    algs = _algs
    global _xvals
    global _xvals_dict
    xvals_dict = default_dict(_xvals_dict)
    xvals_dict.set_default(_xvals)
    global _xlabel
    global _xlabel_dict
    xlabel_dict = default_dict(_xlabel_dict)
    xlabel_dict.set_default(_xlabel)
    global _ylabels
    ylabels = _ylabels

    global _default_map
    global _default_map_dict
    default_map_dict = default_dict(_default_map_dict)
    default_map_dict.set_default(_default_map)

    # status
    # fail: 0
    # start: 1
    # done: 2
    # timeout: 3

    header = ["seed", "data", "alg", "xlabel", "xval", *ylabels]
    records = []
    for seed in seeds:
        for dataName in dataNames:
            for alg in algs:
                xvals = xvals_dict[alg]
                xlabel = xlabel_dict[alg]
                default_map = default_map_dict[alg].copy()
                for xval in xvals:
                    default_map[xlabel] = xval
                    arg_str_list = get_arg_str_list(seed, dataName, alg, default_map)
                    for arg_str in arg_str_list:
                        if verbose_level >= 2:
                            print(arg_str)
                        rec = [seed, dataName, alg, xlabel, xval]
                        is_fail = False
                        for ylabel in ylabels:
                            assert ":" in ylabel
                            tbl, exp_label = ylabel.split(":")
                            exp_key = tbl + ":" + arg_str
                            exp_res_byte = conn.get(exp_key)
                            if exp_res_byte is None:
                                if verbose_level >= 1:
                                    print("[fail:NN]:", exp_key, "[label]:", ylabel)
                                is_fail = True
                                break
                            else:
                                if tbl == 'status':
                                    yval = int(exp_res_byte)
                                else:
                                    if "Traceback" in exp_res_byte.decode():
                                        if verbose_level >= 1:
                                            print("[fail:TB]:", exp_key)
                                        is_fail = True
                                        break
                                    exp_res_dict = json.loads(exp_res_byte)
                                    if exp_label not in exp_res_dict:
                                        if verbose_level >= 1:
                                            print(f"[fail:NI][key={exp_label}]:", exp_key)
                                        is_fail = True
                                        break
                                    yval = exp_res_dict[exp_label]
                                rec.append(yval)
                        if not is_fail:
                            records.append(rec)

    df = pd.DataFrame(records, columns=header)
    return df


def plot_df_list(df_list, ofname_pat, is_paper=False, **kwargs):
    """
        currently only working in dataNames
    """
    global _dataNames
    dataNames = _dataNames
    for dataName, df in zip(dataNames, df_list):
        if ofname_pat is not None:
            ofname = ofname_pat.format(dataName=dataName)
        else:
            ofname = None
        if dataName in kwargs:
            plot_kwargs = kwargs[dataName]
        else:
            plot_kwargs = {}
        plot_df(df, ofname, is_paper, **plot_kwargs)
    return


def plot_df(df, ofname=None, is_paper=False, **kwargs):
    if len(df) == 0:
        return
    global _xlog
    xlog = _xlog
    global _ylog
    ylog = _ylog
    global _xlabel
    xlabel = _xlabel
    global _ylabel
    ylabel = _ylabel
    global _zlabel
    zlabel = _zlabel
    global _zfontsize
    zfontsize = _zfontsize
    global _algs
    algs = _algs
    global _figsize
    figsize = _figsize
    global _linewidth
    linewidth = _linewidth
    global _markerwidth
    markerwidth = _markerwidth
    global _legend
    legend = _legend
    global _xlim
    xlim = _xlim
    global _ylim
    ylim = _ylim
    global _xticks
    xticks = _xticks
    global _yticks
    yticks = _yticks
    global _xticks_format
    xticks_format = _xticks_format
    global _yticks_format
    yticks_format = _yticks_format

    global _confirm
    confirm = _confirm
    global _verbose_fig
    verbose = _verbose_fig
    global _save_table
    save_table = _save_table
    global _dpi
    dpi = _dpi
    if dpi is None:
        dpi = 'figure'
    global alg_style_dict

    global _clip_on

    data = 'data'
    dataName = df[data].tolist()[0]
    alg = 'alg'
    plt.rc('text', usetex=True)
    fig = plt.figure(figsize=figsize, dpi=120)
    fig.patch.set_facecolor('white')

    if len(df) > 0:
        if xlabel == 'xval':
            _xlabel_list = list(df['xlabel'])
            plt.xlabel(param_label_dict[_xlabel_list[0]])
        else:
            plt.xlabel(param_label_dict[xlabel])
    plt.ylabel(param_label_dict[ylabel])

    for idx, algName in enumerate(algs):
        group_df = df[df[alg] == algName]
        if len(group_df) == 0:
            continue
        algName_org = algName
        algName = alg_label_dict[algName]
        x_list = group_df[xlabel].tolist()
        y_list = group_df[ylabel].tolist()
        if _zlabel_dict is not None:
            zlabel = _zlabel_dict[(dataName, algName_org)]
        if zlabel is not None:
            z_list = group_df[zlabel].tolist()
        if ylim is not None and len(ylim) > 0:
            x_list = np.array(x_list)
            y_list = np.array(y_list)
            plt.ylim(ylim)
        style_dict = alg_style_dict[algName_org]
        style_dict['color'] = 'black'
        line, = plt.plot(x_list, y_list, label=algName, markersize=4, markerfacecolor='None',
                         markeredgewidth=markerwidth, linewidth=linewidth, **style_dict)
        if zlabel is not None:
            verticalalignment = 'baseline'
            horizontalalignment = 'left'
            if _zalignments is not None:
                if "default" in _zalignments:
                    horizontalalignment, verticalalignment = _zalignments["default"]
                zalign_key = (dataName, algName_org)
                if zalign_key in _zalignments:
                    horizontalalignment, verticalalignment = _zalignments[zalign_key]

            for xv, yv, zv in zip(x_list, y_list, z_list):
                xoffset = 1.0 if xlog else 0.0
                yoffset = 1.0 if ylog else 0.0

                if _zoffsets is not None:
                    if dataName in _zoffsets:
                        xoffset, yoffset = _zoffsets[dataName]
                    if algName_org in _zoffsets:
                        xoffset, yoffset = _zoffsets[algName_org]
                    zalign_key = (dataName, algName_org)
                    if zalign_key in _zoffsets:
                        xoffset, yoffset = _zoffsets[zalign_key]

                xv = xv * xoffset if xlog else xv+xoffset
                yv = yv * yoffset if ylog else yv+yoffset

                plt.annotate(zv, (xv, yv), fontsize=zfontsize,
                             horizontalalignment=horizontalalignment, verticalalignment=verticalalignment)

        line.set_clip_on(_clip_on)

    if legend:
        leg = plt.legend(frameon=False, prop={'size': 7})

    if 'xlim' in kwargs:
        xlim = kwargs['xlim']
    if xlim is not None and len(xlim) > 0:
        if xlim[0] is not None:
            plt.xlim(left=xlim[0])
        if xlim[1] is not None:
            plt.xlim(right=xlim[1])
    if 'ylim' in kwargs:
        ylim = kwargs['ylim']
    if ylim is not None and len(ylim) > 0:
        if ylim[0] is not None:
            plt.ylim(bottom=ylim[0])
        if ylim[1] is not None:
            plt.ylim(top=ylim[1])

    if "annotate" in kwargs:
        text, pos = kwargs["annotate"]
        plt.annotate(text, pos, xycoords="figure fraction")
    if xlog:
        plt.xscale('log')
    else:
        plt.xscale('linear')
    if ylog:
        plt.yscale('log')
    else:
        plt.yscale('linear')

    ax = plt.gca()
    if 'xticks' in kwargs:
        xticks = kwargs['xticks']
    if xticks is not None:
        ax.set_xticks(xticks)
    if 'xminor' in kwargs:
        ax.tick_params(axis='x', which='minor', bottom=kwargs['xminor'])

    if yticks is not None:
        yticks = kwargs['yticks']
    if 'yticks' in kwargs:
        ax.set_yticks(kwargs['yticks'])
    if 'yminor' in kwargs:
        ax.tick_params(axis='y', which='minor', left=kwargs['yminor'])

    if xticks_format is not None:
        ax.xaxis.set_major_formatter(xticks_format)
    if yticks_format is not None:
        ax.yaxis.set_major_formatter(yticks_format)

    if ofname:
        global _latex_dir
        ofpath = _latex_dir + ofname
        print("fig saved at:", ofpath)
        plt.savefig(ofpath, bbox_inches="tight", pad_inches=0.1, dpi=dpi)
        if save_table:
            global _tbl_dir
            df: pd.DataFrame = df
            ofpath_tbl = _tbl_dir + ofname.replace(".png", ".csv")
            print("tbl saved at:", ofpath_tbl)
            df.to_csv(ofpath_tbl, index=False)

        if confirm and is_paper:
            print(data_label_dict[df[data].tolist()[0]])
            plt.title("")
            ofpath = _latex_dir + ofname
            print("fig saved at:", ofpath)

            plt.savefig(ofpath, bbox_inches="tight", pad_inches=0.1, dpi=dpi)

    if verbose:
        pass
    else:
        plt.close(fig)
    return


def is_matched_dict(query_dict, args):
    for key, val in query_dict.items():
        if key not in args:
            return False

    for key, val in query_dict.items():
        if isinstance(val, list):
            if args.__dict__[key] not in val:
                return False
        elif val == '*':
            continue
        else:
            if args.__dict__[key] != val:
                return False
    return True


def get_matched_exp_keys(query_dict, fast=True):
    exp_keys = []
    grep_pat = "est:*"
    if fast:
        for key, val in query_dict.items():
            key = key.replace('_', '-')
            if isinstance(val, bool):
                continue
            if isinstance(val, list):
                continue
            if isinstance(val, float) and val < 0.0001 and val > 0.0:
                val = np.format_float_positional(val)
            if val == '*':
                grep_pat += f"--{key} *"
            else:
                grep_pat += f"--{key} {val}*"

    for key in conn.keys(grep_pat):
        key = key.decode()
        _, exp_key = key.split(":")
        args = get_parsed_args(exp_key)
        is_matched = is_matched_dict(query_dict, args)
        if is_matched:
            exp_keys.append(exp_key)
    return exp_keys


def get_parser_with_ignores_astrid(required=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dname", required=required, type=str, help="dataset name")
    parser.add_argument("--delta", required=required, type=int, help="train model")
    parser.add_argument("--p-train", required=required, type=float, help="ratio of training data")
    parser.add_argument("--p-val", default=0.1, type=float, help="ratio of valid data")
    parser.add_argument("--p-test", default=0.1, type=float, help="ratio of test data")
    parser.add_argument("--seed", required=required, type=int, help="random seed")
    parser.add_argument("--es", required=required, type=int, help="embedding size (default=64)")
    parser.add_argument("--bs", required=required, type=int, help="batch size (default=128)")
    parser.add_argument("--lr", required=required, type=float, help="learning rate (default=0.001)")
    parser.add_argument("--epoch", required=required, type=int, help="epoch for selectivity learner (default=64)")
    parser.add_argument("--emb-epoch", required=required, type=int, help="epoch for embedding learner (default=32)")
    parser.add_argument("--dsc", required=required, type=int, help="decoder scale (min_value 32) (default=128)")
    # parser.add_argument("--overwrite", default=False, type=bool, help="overwrite mode")

    # ignored options
    ignore_opt_list = []
    ow_option = '--overwrite'
    ignore_opt_list.append(ow_option)
    parser.add_argument(ow_option, action='store_true', help="do not use stored file")
    analysis_option = '--analysis'
    ignore_opt_list.append(analysis_option)
    parser.add_argument(analysis_option, choices=['ts', 'lat', 'qs'],
                        help="analysis of model [ts]: time series, [lat]:latency")
    return parser, ignore_opt_list


def get_parser_with_ignores():
    parser = argparse.ArgumentParser(description="parse")
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--dname', type=str, help='data name')
    parser.add_argument('--p-train', type=float, help='ratio of augmented training data')
    parser.add_argument('--p-val', type=float, help='ratio of valid')
    parser.add_argument('--p-test', type=float, help='ratio of test')
    group = parser.add_mutually_exclusive_group()
    parser.add_argument('--seed', type=int, help='estimator seed')
    parser.add_argument('--l2', type=float, help='L2 regularization ')
    parser.add_argument('--lr', type=float, help='train learning rate [default (RNN=0.001), (CardNet=0.001)]')
    parser.add_argument('--vlr', type=float, help='train learning rate for VAE in CardNet [default 0.0001]')
    parser.add_argument('--layer', type=int, help="number of RNN layers")
    parser.add_argument('--pred-layer', type=int, help="number of pred layers")
    parser.add_argument('--cs', type=int, help="rnn cell size (it should be even)")
    parser.add_argument('--csc', type=int, help="CardNet model scale (default=256)")
    parser.add_argument('--vsc', type=int, help="CardNet vae model scale (default=128)")
    parser.add_argument('--Ntbl', type=int, help='maximum length of extended n-gram table for LBS (default=5)')
    parser.add_argument('--PT', type=int, help='threshold for LBS (PT>=1) (default=20) ')
    parser.add_argument('--max-epoch', type=int,
                        help="maximum epoch (default=100 for RNN 800 for CardNet)")
    parser.add_argument('--patience', type=int, help="patience for training neural network")

    parser.add_argument('--max-d', type=int, help="maximum distance threshold")
    parser.add_argument('--max-char', type=int, help="maximum # of characters to support (default=200)")
    parser.add_argument('--sep-emb', action='store_true', help="char dist sep embed?")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--prfx', action='store_true', help="additional prefix information")
    group.add_argument('--Eprfx', action='store_true', help="additional prefix & enumerate prefixes")

    parser.add_argument('--bs', type=int, help="batch size (default=32)")
    parser.add_argument('--vbs', type=int, help="batch size (default=256)")
    parser.add_argument('--max-epoch-vae', type=int,
                        help="maximum epoch for VAE in CardNet (default=100)")
    parser.add_argument('--h-dim', type=int, help="prediction hidden dimension for RNN")
    parser.add_argument('--es', type=int, help="total(char+dist) embedding size for RNN")
    parser.add_argument('--L', type=int, help='minhash size for LBS (default=10)')
    parser.add_argument('--vl2', type=float, help='vae L2 regularization')
    parser.add_argument('--vclip-lv', type=float, help='vae soft value clipping on logvar')
    parser.add_argument('--vclip-gr', type=float, help='vae hard value clipping on gradient')
    parser.add_argument('--clip-gr', default=0.0, type=float, help='estimation model hard value clipping on gradient')

    # ignored options
    ignore_opt_list = []
    ow_option = '--overwrite'
    ignore_opt_list.append(ow_option)
    parser.add_argument(ow_option, action='store_true', help="do not use stored file")
    train_option = '--outdata'
    ignore_opt_list.append(train_option)
    parser.add_argument(train_option, action='store_true',
                        help="only output used training for this experiment without training procedure")
    analysis_option = '--analysis'
    ignore_opt_list.append(analysis_option)
    parser.add_argument(analysis_option, choices=['ts', 'lat', 'qs', 'tr'],
                        help="analysis of model [ts]: time series, [lat]:latency, [tr]:train data")
    rewrite_option = '--rewrite'
    ignore_opt_list.append(rewrite_option)
    parser.add_argument(rewrite_option, action='store_true', help='postprocessing from estimated result')

    return parser, ignore_opt_list


def get_parsed_args(exp_key):
    # assert "--model" in exp_key or "--path" in exp_key, exp_key
    if "--model" in exp_key:
        parser, _ = get_parser_with_ignores()
    else:
        parser, _ = get_parser_with_ignores_astrid(required=False)
    parser.usage = argparse.SUPPRESS
    args = None
    try:
        args = parser.parse_args(exp_key.split())
    except:
        print(exp_key)
        print("fail")
    return args
