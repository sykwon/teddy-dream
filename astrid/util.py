import subprocess
import time
import socket
import json
import argparse
import sys


def get_git_revision_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().rstrip()


def get_git_revision_short_hash():
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().rstrip()


def get_git_commit_message():
    return subprocess.check_output(["git", "show", "-s", "--format=%s"]).decode().rstrip()


def get_model_exp_dict(model, exe_time=None, size=None, err=None, log_time=None, query=None, prefix=None, update=None, **kwargs):
    exp_dict = {}
    if log_time is None:
        log_time = time.ctime()
    exp_dict["log_time"] = log_time
    exp_dict["model"] = model
    if exe_time is not None:
        exp_dict["time"] = exe_time
    if size is not None:
        exp_dict["size"] = size
    if err is not None:
        exp_dict["err"] = err
    if query is not None:
        exp_dict["query"] = query
    if prefix is not None:
        exp_dict["prefix"] = prefix
    if update is not None:
        exp_dict["update"] = update
    # if best_epoch is not None:
    #     exp_dict["best_epoch"] = update
    # if last_epoch is not None:
    #     exp_dict["last_epoch"] = last_epoch
    for k, v in kwargs.items():
        exp_dict[k] = v
    exp_dict["hostname"] = socket.gethostname()
    exp_dict["githash"] = get_git_revision_hash()
    exp_dict["gitmsg"] = get_git_commit_message()
    return exp_dict


def get_model_exp_json_str(model, time=None, size=None, err=None, log_time=None, query=None, prefix=None, update=None,
                           **kwargs):
    exp_dict = get_model_exp_dict(model, time, size, err, log_time, query, prefix, update, **kwargs)
    return json.dumps(exp_dict)


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
    parser.add_argument(analysis_option, action="store_true", help="analysis mode")
    # parser.add_argument(analysis_option, choices=['ts', 'lat', 'qs'],
    #                     help="analysis of model [ts]: time series, [lat]:latency")
    analysis_option = '--analysis-latency'
    ignore_opt_list.append(analysis_option)
    parser.add_argument(analysis_option, action="store_true", help="analysis mode for latency")
    return parser, ignore_opt_list


def get_model_args(verbose=1):
    parser, ignore_opt_list = get_parser_with_ignores_astrid()

    cmd_list = sys.argv[1:]
    if verbose:
        print("cmd:", cmd_list)

    parser_opts = [action.option_strings[-1] for action in parser._actions[1:]]
    if verbose:
        print("opt:", parser_opts)

    for opt in ignore_opt_list:
        if opt in cmd_list:
            cmd_list.remove(opt)
        if opt in parser_opts:
            parser_opts.remove(opt)

    cmd_opts = [opt for opt in cmd_list if opt.startswith("-")]
    if verbose:
        print(cmd_opts)
    parser_opts_filtered = list(filter(lambda x: x in cmd_opts, parser_opts))
    if verbose:
        print(parser_opts_filtered)

    assert len(cmd_opts) == len(
        parser_opts_filtered), f"# of filtered options should be equal to # of cmd_opts, {len(cmd_opts)}, {len(parser_opts_filtered)}\n>>> input: {' '.join(cmd_opts)} \n>>> orgin: {' '.join(parser_opts_filtered)}"
    assert all([cmd_opt == parser_opt for cmd_opt, parser_opt in zip(cmd_opts, parser_opts_filtered)]), \
        f"option order should follow the usage of the program\n>>> input: {' '.join(cmd_opts)} \n>>> orgin: {' '.join(parser_opts_filtered)}"
    exp_key = " ".join(cmd_list)

    if verbose:
        print("[exp_key]:", exp_key)
        print(f">>> model:{exp_key}")
        print(f">>> est:{exp_key}")

    args = parser.parse_args()
    return args, exp_key


def distinct_prefix(strings):
    prfx_set = set()
    output = []
    for string in strings:
        for i in range(1, len(string) + 1):
            prefix = string[:i]
            if prefix not in prfx_set:
                output.append(prefix)
            prfx_set.add(prefix)
    return output
