import argparse
import redis
import subprocess
import time
import os
import signal
import traceback
from subprocess import CalledProcessError, Popen, STDOUT, PIPE, TimeoutExpired
from util import *


def do_exp(conn, cmd, timeout, overwrite, tag_hash, clear):
    status_head = "status"
    exp_head = "join"
    git_hash = get_git_revision_hash()
    args = cmd.split()
    exe = args[0]
    if exe == "./main_info":
        status_head += "(info)"
        exp_head += "(info)"
    args_str = " ".join(args[1:])
    if tag_hash:
        cmd_key = args_str + ":" + git_hash
    else:
        cmd_key = args_str
    # args_str = " ".join(args[1:]).replace("/", "\\")
    print("[py]", cmd_key)
    status_key = f"{status_head}:{cmd_key}"
    exp_key = f"{exp_head}:{cmd_key}"

    if overwrite or clear:
        status = get_status_code(conn, status_key)
        print("saved status:", status_repr(status))
        if conn.exists(status_key):
            conn.delete(status_key)
        if conn.exists(exp_key):
            conn.delete(exp_key)
    if clear:
        return
    status = get_status_code(conn, status_key)
    print("[py]", "status:", status_repr(status))
    saved_result = conn.get(exp_key)
    if saved_result is not None:
        saved_result = saved_result.decode()
    print("[py]", "saved_result:", saved_result)

    if not status:  # check if fail or unexecuted
        print_status(1)
        conn.set(status_key, 1)  # set start status
        try:
            output = check_output(cmd, stderr=STDOUT, shell=True, timeout=timeout)
            print_status(2)
            conn.set(status_key, 2)  # set done status
            # with open(f"time/{args_str}.txt") as f:
            #     line = f.readline()
            exp_json_str = get_join_exp_json_str(cmd)
            conn.set(exp_key, exp_json_str)
            print("[py]", "set result:", exp_json_str)
            print("[py]", "output:", output)
            return 2
        except TimeoutExpired as e:
            print("[py]", "Error msg:", e)
            print("[py]", f"(Timeout) status timeout: '{cmd_key}'")
            print_status(3)
            conn.set(status_key, 3)  # set timeout status
            # with open(f"time/{args_str}.txt") as f:
            #     line = f.readline()
            exp_json_str = get_join_exp_json_str(cmd)
            conn.set(exp_key, exp_json_str)
            return 3
        except CalledProcessError as e:
            tb = traceback.format_exc()
            print("[tb]", tb)
            print("[py]", f"(output): '{e.stdout}'")
            print("[py]", f"(exit code): '{e.returncode}'")
            print("[py]", f"(CalledProcess) status fail: '{cmd_key}'")
            print_status(0)
            conn.set(status_key, 0)  # set fail status
            conn.set(exp_key, tb + "[stdout] " + e.stdout)  # set fail log
            raise
        except KeyboardInterrupt as e:
            tb = traceback.format_exc()
            # print("[tb]", tb)
            print("[py]", f"(Interrupt) status fail: '{cmd_key}'")
            print_status(0)
            conn.set(status_key, 0)  # set fail status
            conn.set(exp_key, tb)  # set fail log
            raise
        except Exception as e:
            tb = traceback.format_exc()
            print("[tb]", tb)
            print("[py]", f"(Exception) status fail: '{cmd_key}'")
            print_status(0)
            conn.set(status_key, 0)  # set fail status
            conn.set(exp_key, tb)  # set fail log
            raise


def clean_make():
    output = check_output("make clean", stderr=STDOUT, shell=True)
    print("[py]", "make clean:", output)
    output = check_output("make", stderr=STDOUT, shell=True)
    print("[py]", "make:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clear", "-c", action="store_true", default=False, help="clear previous results"
    )
    parser.add_argument(
        "--overwrite", "-o", action="store_true", default=False, help="overwrite previous result"
    )
    parser.add_argument(
        "--githash", "-g", action="store_true", default=False, help="augment git hash"
    )
    parser.add_argument(
        "--preview", "-p", action="store_true", default=False, help="only preview command"
    )
    parser.add_argument("-s", "--seed", nargs="+", help="iterate for seed")
    parser.add_argument("-nq", "--nq_list", nargs="+", help="iterate for nq_list")
    parser.add_argument("--data", "-d", nargs="+", required=True, help="data name")
    parser.add_argument("--alg", "-a", nargs="+", required=True, help="alg name")
    parser.add_argument("--prefix", "-pr", nargs="+", help="augment prefix 0: no prefix, 1: prefix")
    parser.add_argument("--maxlen", "-L", nargs="+", help="qry max_len")
    parser.add_argument("--p-train", "-pt", nargs="+", help="ratio of train dataset")
    parser.add_argument(
        "--exp_thres",
        "-et",
        action="store_true",
        default=False,
        help="experiment by varying threshold from 1 to 10",
    )
    parser.add_argument("--info", "-i", action="store_true", default=False, help="join info")
    parser.add_argument("--threshold", "-t", type=int, default=3, help="alg name")
    parser.add_argument("--hour", "-hr", type=float, help="limit hour")
    args = parser.parse_args()

    preview = args.preview
    clear_res = args.clear
    overwrite = args.overwrite
    tag_hash = args.githash
    seed_list = args.seed

    algName_list = args.alg
    dataName_list = args.data
    join_info = args.info
    threshold = args.threshold
    exp_thres = args.exp_thres
    hour = args.hour
    maxlen_list = args.maxlen
    if maxlen_list is None:
        maxlen_list = [""]
    prefix_list = args.prefix
    conn = redis.Redis()

    if args.seed is None:
        seed_list = [0]
    
    assert args.nq_list is not None
    nq_list = args.nq_list
    ptrain_list = args.p_train
    if ptrain_list is None:
        ptrain_list = [""]
    else:
        assert len(maxlen_list[0]) > 0
        assert len(nq_list) == 1 and nq_list[0] == "1.0"

    for seed in seed_list:
        for dataName in dataName_list:
            if args.hour is None:
                if "wiki2" in dataName:
                    hour = 15
                elif "imdb2" in dataName:
                    hour = 10
                else:
                    hour = 10
            for prefix in prefix_list:
                for algName in algName_list:
                    db_path = f"data/{dataName}.txt"
                    timeout = 3600 * hour
                    for p_train in ptrain_list:
                        for maxlen in maxlen_list:
                            qs_path = (
                                f"data/qs_{dataName}_0_{maxlen}_{p_train}".rstrip("_") + ".txt"
                            )
                            # sample: cmd = "./main 0.1 0.1 intv data/wiki.txt data/qs_wiki.txt 3 0"

                            if not (preview or clear_res):
                                # clean_make()
                                os.makedirs("time/", exist_ok=True)
                                os.makedirs("stat/", exist_ok=True)

                            if exp_thres:
                                thres_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                                timeout = None
                                for threshold in thres_list:
                                    cmd = f"./main 0.1 0.1 {algName} {db_path} {qs_path} {threshold} {prefix} {seed}"
                                    print("[py]", cmd)
                                    if not preview:
                                        code = do_exp(
                                            conn, cmd, timeout, overwrite, tag_hash, clear_res
                                        )
                                continue

                            if join_info:
                                timeout_h = timeout // 3600
                                nq = nq_list[0]
                                cmd = f"./main_info 1.0 {nq} {algName} {db_path} {qs_path} {threshold} {prefix} {seed}"
                                print(f"[py TO:{timeout_h}h]", cmd)
                                if not preview:
                                    code = do_exp(
                                        conn, cmd, timeout, overwrite, tag_hash, clear_res
                                    )
                                continue
                            for nq in nq_list:
                                # cmd = f"./main_info 1.0 {nq} {algName} {db_path} {qs_path} {threshold} {prefix}"
                                cmd = f"./main 1.0 {nq} {algName} {db_path} {qs_path} {threshold} {prefix} {seed}"
                                timeout_h = timeout // 3600
                                print(f"[py TO:{timeout_h}h]", cmd)
                                if not preview:
                                    code = do_exp(
                                        conn, cmd, timeout, overwrite, tag_hash, clear_res
                                    )
                                    if status_string(code) == "timeout":
                                        timeout = 5
