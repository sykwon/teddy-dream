import argparse
import redis
import subprocess
import time
import os
import signal
import traceback
from subprocess import CalledProcessError, Popen, STDOUT, PIPE, TimeoutExpired
from util import *


def do_exp(conn, cmd, timeout, overwrite, clear):
    status_head = "status"
    exp_head = "join"
    args = cmd.split()
    exe = args[0]
    if exe == "./main_info":
        status_head += "(info)"
        exp_head += "(info)"
    cmd_key = " ".join(args[1:])
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
            exp_json_str = get_join_exp_json_str(cmd)
            conn.set(exp_key, exp_json_str)
            print("[py]", "set result:", exp_json_str)
            print("[py]", "output:")
            print(output)
            return 2
        except TimeoutExpired as e:
            print("[py]", "Error msg:", e)
            print("[py]", f"(Timeout) status timeout: '{cmd_key}'")
            print_status(3)
            conn.set(status_key, 3)  # set timeout status
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", nargs="+", required=True, help="data name")
    parser.add_argument("--alg", "-a", nargs="+", required=True, help="alg name")
    parser.add_argument("--prefix", "-pr", nargs="+",
                        help="augment prefix (0: base training data, 1: prefix-aug training data)")
    parser.add_argument("--p-train", "-pt", nargs="+", help="ratio of training dataset")
    parser.add_argument("--info", "-i", action="store_true", default=False, help="join info")
    parser.add_argument("--threshold", "-th", type=int, help="maximum distance threshold (delta_M)")
    parser.add_argument("--n_try", "-nt", type=int, help="number of tries")
    parser.add_argument("--hour", "-hr", type=float, help="timeout to execute algorithm")
    parser.add_argument(
        "--clear", "-c", action="store_true", default=False, help="clear previous results"
    )
    parser.add_argument(
        "--overwrite", "-o", action="store_true", default=False, help="overwrite previous result"
    )
    parser.add_argument(
        "--preview", "-p", action="store_true", default=False, help="show command only"
    )
    args = parser.parse_args()

    preview = args.preview
    clear_res = args.clear
    overwrite = args.overwrite
    n_try = args.n_try

    alg_list = args.alg
    dname_list = args.data
    join_info = args.info
    threshold = args.threshold
    hour = args.hour
    prefix_list = args.prefix
    conn = redis.Redis()

    ptrain_list = args.p_train
    if ptrain_list is None:
        ptrain_list = [""]

    for try_id in range(n_try):
        for dname in dname_list:
            for prefix in prefix_list:
                for alg in alg_list:
                    db_path = f"data/{dname}.txt"
                    timeout = 3600 * hour
                    for p_train in ptrain_list:
                        qs_path = (
                            f"data/qs_{dname}_{p_train}".rstrip("_") + ".txt"
                        )

                        if join_info:
                            timeout_h = timeout // 3600
                            cmd = f"./main_info {alg} {db_path} {qs_path} {threshold} {prefix} {try_id}"
                            print(f"[py TO:{timeout_h}h]", cmd)
                            if not preview:
                                code = do_exp(
                                    conn, cmd, timeout, overwrite, clear_res
                                )
                            continue

                        cmd = f"./main {alg} {db_path} {qs_path} {threshold} {prefix} {try_id}"
                        timeout_h = timeout // 3600
                        print(f"[py TO:{timeout_h}h]", cmd)
                        if not preview:
                            code = do_exp(
                                conn, cmd, timeout, overwrite, clear_res
                            )
                            if status_string(code) == "timeout":
                                timeout = 5
