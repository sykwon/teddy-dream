import os
import json
import signal
import subprocess
import traceback
from time import monotonic as _time
import time


status_dict = {
    None: "unexecuted",
    0: "fail",
    1: "start",
    2: "done",
    3: "timeout",
}


def get_git_revision_short_hash():
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().rstrip()


def check_output(*popenargs, timeout=None, **kwargs):
    r"""Run command with arguments and return its output.

    If the exit code was non-zero it raises a CalledProcessError.  The
    CalledProcessError object will have the return code in the returncode
    attribute and output in the output attribute.

    The arguments are the same as for the Popen constructor.  Example:

    >>> check_output(["ls", "-l", "/dev/null"])
    b'crw-rw-rw- 1 root root 1, 3 Oct 18  2007 /dev/null\n'

    The stdout argument is not allowed as it is used internally.
    To capture standard error in the result, use stderr=STDOUT.

    >>> check_output(["/bin/sh", "-c",
    ...               "ls -l non_existent_file ; exit 0"],
    ...              stderr=STDOUT)
    b'ls: non_existent_file: No such file or directory\n'

    There is an additional optional argument, "input", allowing you to
    pass a string to the subprocess's stdin.  If you use this argument
    you may not also use the Popen constructor's "stdin" argument, as
    it too will be used internally.  Example:

    >>> check_output(["sed", "-e", "s/foo/bar/"],
    ...              input=b"when in the course of fooman events\n")
    b'when in the course of barman events\n'

    By default, all communication is in bytes, and therefore any "input"
    should be bytes, and the return value will be bytes.  If in text mode,
    any "input" should be a string, and the return value will be a string
    decoded according to locale encoding, or by "encoding" if set. Text mode
    is triggered by setting any of text, encoding, errors or universal_newlines.
    """
    if "stdout" in kwargs:
        raise ValueError("stdout argument not allowed, it will be overridden.")

    with subprocess.Popen(
        *popenargs, stdout=subprocess.PIPE, **kwargs, preexec_fn=os.setsid
    ) as process:
        try:
            output = process.communicate(timeout=timeout)[0]
            if output is not None:
                output = output.decode().rstrip()
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGINT)  # send signal to the process group
            output = process.communicate()[0]
            if output:
                output = output.decode().rstrip()
            raise subprocess.TimeoutExpired(process.args, timeout, output=output)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            traceback.print_exc()
            raise Exception
        if process.returncode > 0:
            raise subprocess.CalledProcessError(
                returncode=process.returncode, cmd=process.args, output=output
            )
    return output


def get_status_code(conn, key):
    val = conn.get(key)
    if val is not None:
        val = int(val.decode())
    return val


def status_string(status_code):
    return status_dict[status_code]


def status_repr(status_code):
    return f"{status_dict[status_code]}({status_code})"


def print_status(status_code):
    print("[py]", "status:", status_repr(status_code))


def get_join_exp_dict(cmd):
    # header = ["log_time", "alg", "n_qry", "n_rec", "n_prfx", "d", "p", "time"]
    # types = [str, str, int, int, int, int, bool, float]
    # log_time, infos = line.rstrip()[1:].split("]")
    # values = [log_time]
    # values.extend(infos.split())
    # print(values)
    # exp_dict = {}
    # for i, (k, v) in enumerate(zip(header, values)):
    #     exp_dict[k] = types[i](v)
    # if exp_dict["time"] <= 0.0:
    #     exp_dict["time"] = None
    # return exp_dict
    args = cmd.split()
    exe = args[0]

    args_str = " ".join(args[1:]).replace("/", "\\")
    if exe == "./main":
        with open(f"time/{args_str}.txt") as f:
            line = f.readline()
        exp_dict = parse_join_exp_time_result(line)

    elif exe == "./main_info":
        with open(f"stat/{args_str}.txt") as f:
            text = f.read().rstrip()
        exp_dict = parse_join_exp_info_result(text)
    return exp_dict


def parse_join_exp_time_result(line):
    header = ["log_time", "alg", "n_qry", "n_rec", "n_prfx", "d", "p", "time"]
    types = [str, str, int, int, int, int, bool, float]
    log_time, infos = line.rstrip()[1:].split("]")
    values = [log_time]
    values.extend(infos.split())
    print(values)
    exp_dict = {}
    for i, (k, v) in enumerate(zip(header, values)):
        if k == "p":
            if v == "True":
                exp_dict[k] = True
            else:
                exp_dict[k] = False
        else:
            exp_dict[k] = types[i](v)
    if exp_dict["time"] <= 0.0:
        exp_dict["time"] = None
    return exp_dict


def parse_join_exp_info_result(text):
    keys = [
        "log_time",
        "Total line",
        "Computed line",
        "Pruned line",
        "Total cell",
        "Computed cell",
        "Pruned cell",
    ]
    exp_dict = {}
    lines = text.split("\n")
    header, lines = lines[0], lines[1:]
    log_time = header[1:].split("]")[0]
    values = [log_time]

    assert len(lines) == len(keys[1:])
    for line in lines:
        value = line.split()[-1]
        values.append(int(value))
    for k, v in zip(keys, values):
        exp_dict[k] = v
    return exp_dict


def get_join_exp_json_str(cmd):
    exp_dict = get_join_exp_dict(cmd)
    return json.dumps(exp_dict)
