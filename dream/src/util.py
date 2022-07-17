from collections.abc import Iterable
import os
import pickle
import heapq
import struct
import inspect
import sys
import warnings
import socket
import time
from functools import reduce
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

import re
import torch
from torch import nn
from tqdm import tqdm
import math
import json
import subprocess
import time
import argparse

debug_print = False
meta_chars = r".^$*+?{}[]\|()"
exclude_chars = r"\?"
esc_qsmrk = "\u203C"
status_dict = {
    None: "unexecuted",
    0: "fail",
    1: "start",
    2: "done",
    3: "timeout",
}


def deco_logging_time(origin_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = origin_fn(*args, **kwargs)
        end_time = time.time()
        duration_time = end_time - start_time
        print(f"Execution Time of [{origin_fn.__name__}]: {duration_time} sec")
        return result

    return wrapper_fn


def deco_load_from_pkl_silent(origin_fn):
    def wrapper_fn(*args, **kwargs):
        return deco_load_from_pkl(origin_fn)(*args, silent=True, **kwargs)

    return wrapper_fn


def deco_load_from_pkl(origin_fn):
    def wrapper_fn(*args, silent=False, **kwargs):
        for arg in args:
            if isinstance(arg, (int, str, bool, float)):
                raise TypeError(f"{arg} should be given with key in {origin_fn.__name__}")

        filepath = kwargs_and_function_to_file_path(origin_fn, kwargs, "pkl")
        if silent:
            sys.stdout = None

        if os.path.exists(filepath):
            print(f"while loading file from path {filepath} ... ", end='', flush=True)
            output = pickle.load(open(filepath, "rb"))
            print("done")
        else:
            print(f"while saving result to path {filepath} ...")
            output = origin_fn(*args, **kwargs)
            makedirs(filepath)
            pickle.dump(output, open(filepath, "wb"))
            print(f"saving result to path {filepath} done")
            print(f"uploading result to path {filepath} done")

        if silent:
            sys.stdout = sys.__stdout__
        return output

    return wrapper_fn


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


class Map(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """

    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]


def read_string_db_by_file_path(n, file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        count = 0
        while True:
            row = f.readline()
            if n:
                if count >= n:
                    break
            if not row:
                break
            data.append(row.rstrip())
            count += 1
    return data


def read_string_db_by_name(n, db_name):
    file_path = f"data/{db_name}.txt"
    db = read_string_db_by_file_path(n, file_path)
    return db


def kwargs_and_function_to_file_path(func, kw_dict, ext, dirname="pkl"):
    func_str = func
    if hasattr(func_str, "__name__"):
        func_str = func.__name__

    kv_list = sorted(kw_dict.items(), key=lambda x: x[0])
    kv_list = map(lambda x: str(x), reduce(lambda x, y: x + y, kv_list, tuple()))
    file_name = "_".join(kv_list) + f".{ext}"
    file_path = f"{dirname}/{func_str}/" + file_name
    return file_path


def load_extended_ngram_table_set(db, n, name, N, over_write=False):
    kwargs = {'n': n, 'name': name, 'N': N}
    file_path = kwargs_and_function_to_file_path(load_extended_ngram_table_set, kwargs, "bin")

    makedirs(file_path)
    if db is None:
        assert os.path.exists(file_path), file_path
        assert os.path.getsize(file_path) > 0, f"{file_path} is empty"
    if not over_write and os.path.exists(file_path):
        eqt_set = read_bin_eqt_set(N, n, name)
    else:
        assert db is not None
        eqt_set = create_extended_qgram_table_set(db, N)
        write_bin_eqt_set(eqt_set, N, n, name)
    return eqt_set


def create_extended_qgram_table_set(db, N):
    """
    :param db: database which consists of records with string attribute
    :param N: the maximum of ngram length
    :param PT: min frequency (engram_hash whose count less than PT will be deleted)
    :return: a dictionary whose key and value are extended ngram and set of record ids, respectively
    """
    print(f"start {create_extended_qgram_table_set.__str__().split()[1]}")
    engram = {}
    for rid, record in tqdm(enumerate(db), total=len(db)):
        length = len(record)
        # print(record)
        for l in range(1, N + 1):
            for s in range(length - l + 1):
                token = record[s:s + l]
                for sub_token in generator_replaced_string(token):
                    if sub_token not in engram:
                        engram[sub_token] = [rid]
                    else:
                        if engram[sub_token][-1] != rid:
                            engram[sub_token].append(rid)
        if (rid + 1) % 100000 == 0:
            print(f"{rid + 1} record are processed in create_extended_qgram_table_set")
    for key, val in engram.items():
        engram[key] = np.array(val, dtype=np.int32)
    return engram


def load_extended_ngram_table_hash(db, n, name, N, PT, L, seed, over_write=False):
    kwargs = {'n': n, 'name': name, 'N': N, 'PT': PT, 'L': L, 'seed': seed}
    filepath = kwargs_and_function_to_file_path(load_extended_ngram_table_hash, kwargs, "bin")

    makedirs(filepath)
    if not over_write and os.path.exists(filepath):
        eqt_hash = read_bin_eqt_hash(N, n, name, L, PT, seed)
    else:
        write_bin_eqt_hash(db, N, n, name, L, PT, seed, over_write=over_write)
        eqt_hash = read_bin_eqt_hash(N, n, name, L, PT, seed)

    default_hash = [n]
    for _ in range(L):
        default_hash.append(0)
    eqt_hash[''] = default_hash
    return eqt_hash


@deco_load_from_pkl_silent
def load_limited_minimal_perms(n, d):
    perm_str_idx_list = []
    sample_str = ''.join([chr(ord('a') + i) for i in range(n)]) + '?'
    gen_DIR = get_DIR_minimal_combinations(d=d, is_sort=True)
    count = 0
    max_count = 10000
    for D, I, R in gen_DIR:
        indexed_perms = calculate_indexed_perms(D=D, I=I, R=R, n=n)
        for indexed_perm in indexed_perms:
            tranformed_str = "".join([sample_str[x] for x in indexed_perm])
            perm_str_idx_list.append((indexed_perm, tranformed_str, count))
            count += 1
            if count >= max_count:
                break
        if count >= max_count:
            break

    # ---------- filter basestring by substring condition ----------- #
    queue = sorted(perm_str_idx_list, key=lambda x: x[1], reverse=True)
    queue = sorted(queue, key=lambda x: len(x[1]), reverse=True)
    queue = [x[2] for x in queue]

    minimal_idices = []
    while len(queue) > 0:
        token = queue.pop()
        minimal_idices.append(token)
        token_str = perm_str_idx_list[token][1]
        compiled = is_general_substring_build(token_str)
        for elem in reversed(queue):
            elem_str = perm_str_idx_list[elem][1]
            # if token_str in elem_str:
            if compiled.search(elem_str):
                queue.remove(elem)
    # ---------------------------------------------------------------- #
    output = [perm_str_idx_list[idx][0] for idx in minimal_idices]
    return output


# @deco_load_from_pkl_silent
# def load_get_all_minimal_perms(n, d):
#     perm_str_idx_list = []
#     sample_str = ''.join([chr(ord('a') + i) for i in range(n)]) + '?'
#     gen_DIR = get_DIR_minimal_combinations(d=d)
#     count = 0
#     for D, I, R in gen_DIR:
#         indexed_perms = calculate_indexed_perms(D=D, I=I, R=R, n=n)
#         for indexed_perm in indexed_perms:
#             tranformed_str = "".join([sample_str[x] for x in indexed_perm])
#             perm_str_idx_list.append((indexed_perm, tranformed_str, count))
#             count += 1

#     # ---------- filter basestring by substring condition ----------- #
#     queue = sorted(perm_str_idx_list, key=lambda x: x[1], reverse=True)
#     queue = sorted(queue, key=lambda x: len(x[1]), reverse=True)
#     queue = [x[2] for x in queue]

#     minimal_idices = []
#     while len(queue) > 0:
#         token = queue.pop()
#         minimal_idices.append(token)
#         token_str = perm_str_idx_list[token][1]
#         for elem in reversed(queue):
#             elem_str = perm_str_idx_list[elem][1]
#             if token_str in elem_str:
#                 queue.remove(elem)
#     # ---------------------------------------------------------------- #
#     output = [perm_str_idx_list[idx][0] for idx in minimal_idices]
#     return output


@deco_load_from_pkl_silent
def load_get_all_perms(n, d):
    output = []
    gen_DIR = get_DIR_combinations(d=d)
    for D, I, R in gen_DIR:
        indexed_perms = calculate_indexed_perms(D=D, I=I, R=R, n=n)
        output.extend(indexed_perms)
    return output


def gen_DIR_minimal_combinations(d):
    """
        generate minimal_combinations less than d operations
    :param d:
    :return:
    """
    for I in range(d+1):
        for D in range(d - I, -1, -1):
            R = d - D - I
            yield (D, I, R)


def get_DIR_minimal_combinations(d, is_sort=False):
    if is_sort:
        return sorted(gen_DIR_minimal_combinations(d), key=lambda x: x[1] - x[0])
    else:
        return list(gen_DIR_minimal_combinations(d))


def get_DIR_combinations(d):
    """
        generate less than d operations
    :param d:
    :return:
    """
    output = []
    for D in range(0, d + 1):
        for I in range(0, d + 1 - D):
            for R in range(0, d + 1 - D - I):
                output.append((D, I, R))
    return output
    # print("D, I, R: %d %d %d" % (D, I, R))


def generator_DIR_length(d, length):
    """
        generate exact d operations and DIR pattern with many "?" first
    :param d:
    :param length:
    :return:
    """
    abs_length = abs(length)
    assert abs_length <= d
    for R in range(d - abs_length, -1, -2):
        I = (d + length - R) // 2
        D = (d - length - R) // 2
        yield D, I, R


def is_match_string_pattern(string, sp):
    """
    :param string: given string
    :param sp: string pattern
    :return: boolean which two given inputs are matched
    """
    if len(string) is not len(sp):
        return False

    for c1, c2 in zip(string, sp):
        if c2 != '?':
            if c1 != c2:
                return False
    return True


def msp(items):
    '''
    Yield the permutations of `items` where items is either a list
    of integers representing the actual items or a list of hashable items.
    The output are the unique permutations of the items given as a list
    of integers 0, ..., N-1 that represent the N unique elements in
    `items`.

    Examples
    ========

    # >>> for i in msp('xoxox'):
    # ...   print(i)

    [1, 1, 1, 0, 0]
    [0, 1, 1, 1, 0]
    [1, 0, 1, 1, 0]
    [1, 1, 0, 1, 0]
    [0, 1, 1, 0, 1]
    [1, 0, 1, 0, 1]
    [0, 1, 0, 1, 1]
    [0, 0, 1, 1, 1]
    [1, 0, 0, 1, 1]
    [1, 1, 0, 0, 1]

    Reference: "An O(1) Time Algorithm for Generating Multiset Permutations", Tadao Takaoka
    https://pdfs.semanticscholar.org/83b2/6f222e8648a7a0599309a40af21837a0264b.pdf
    '''

    def visit(head):
        (rv, k) = ([], head)
        for i in range(N):
            (dat, k) = E[k]
            rv.append(dat)
        return rv

    # item to index
    # u = list(set(items))
    # E = list(reversed(sorted([u.index(i) for i in items])))

    E = list(items)
    N = len(E)
    # put E into linked-list format
    (val, nxt) = (0, 1)
    for i in range(N):
        E[i] = [E[i], i + 1]
    E[-1][nxt] = None
    head = 0
    afteri = N - 1
    beforei = afteri - 1
    test = visit(head)
    yield test
    # yield visit(head)
    while E[afteri][nxt] is not None or E[afteri][val] < E[head][val]:
        j = E[afteri][nxt]  # added to algorithm for clarity
        if j is not None and E[beforei][val] >= E[j][val]:
            beforek = afteri
        else:
            beforek = beforei
        k = E[beforek][nxt]
        E[beforek][nxt] = E[k][nxt]
        E[k][nxt] = head
        if E[k][val] < E[head][val]:
            beforei = k
        afteri = E[beforei][nxt]
        head = k

        test = visit(head)
        yield test


def gen_minimal_base_substring(string, d, reduced=True):
    """
        This function uses the wildcard '?' which means allow any single character
        D, I, R means 'deletion', 'insertion' and 'replacement', respectively
        It removes redundant base substrings
    :param string: given string
    :param d: delta
    :return: list of base substrings
    """
    # ---------- filter basestring by substring condition ----------- #
    queue = gen_base_substring(string, d, reduced=reduced)
    queue = list(reversed(queue))
    output = []
    while len(queue) > 0:
        token = queue.pop()
        output.append(token)
        for elem in reversed(queue):
            if token in elem:
                queue.remove(elem)
    # ---------------------------------------------------------------- #

    # --- filter basestring by generalized substring condition ------- #
    queue = list(reversed(output))
    output = []
    while len(queue) > 0:
        token = queue.pop()
        output.append(token)
        if "?" not in token:  # filter non-generalized token
            continue
        compiled = is_general_substring_build(token)
        for elem in reversed(queue):
            if len(elem) > len(token):  # skip all elems with different length
                break
            if compiled.search(elem) is not None:
                if debug_print:
                    print("gen", token, elem)
                queue.remove(elem)
            # if is_general_substring(token, elem):
            #     if debug_print: print("gen", token, elem)
            #     queue.remove(elem)
    # ---------------------------------------------------------------- #
    return output


def generator_replaced_string(string):
    """
        This function considers replacement operation only.
    :param string:
    :return:
    """

    # N = len(string)
    # total = 2 ** N
    # i = 0
    # while i < total:
    #
    #     i += 1

    def idx2onoff(x):
        output = []
        curr = x
        for _ in range(n):
            output.append(curr % 2 == 1)
            curr //= 2
        return reversed(output)

    assert esc_qsmrk not in string
    if "?" in string:
        string = string.replace("?", esc_qsmrk)
    assert "?" not in string
    n = len(string)
    for idx in range(2 ** n):
        output = []
        onoff = idx2onoff(idx)
        for i, is_change in enumerate(onoff):
            if is_change:
                # output += '?'
                output.append("?")
            else:
                # output += string[i]
                output.append(string[i])
        yield "".join(output)


def DIR_perm(len_origin, D, I, R):
    """
        D: 1, R: 2, I: 3
    :param len_origin:
    :param D:
    :param I:
    :param R:
    :return:
    """

    n = len_origin
    np.zeros(n + I, dtype=np.int32)
    msp_input = np.zeros(n + I)
    msp_input[:I] = 3
    msp_input[I:I + R] = 2
    msp_input[I + R:I + R + D] = 1
    gen_perm = msp(msp_input)
    return gen_perm


def perm_to_idx(perm):
    output = []
    pos = 1
    for str_type in perm:
        if str_type == 0:
            output.append(pos)
            pos += 1
        elif str_type == 1:
            pos += 1
        elif str_type == 2:
            output.append(0)
            pos += 1
        elif str_type == 3:
            output.append(0)
    return output


def perm_to_str(input_str, perm):
    """
        Origin: 0, D: 1, R: 2, I: 3
    :param input_str:
    :param perm:
    :return:
    """
    output = []
    pos = 0
    for str_type in perm:
        if str_type == 0:
            output.append(input_str[pos])
            pos += 1
        elif str_type == 1:
            pos += 1
        elif str_type == 2:
            output.append("?")
            pos += 1
        elif str_type == 3:
            output.append("?")

    output = "".join(output)
    return output


def perm2idx(perm, n):
    # 0, 1,2,3 = O, D, R, I
    perm = np.asarray(perm)
    output = np.zeros_like(perm)
    output[perm < 3] = range(n)
    output[perm > 0] = -1
    output = output[perm != 1]
    return output


def calculate_indexed_perms(D, I, R, n):
    output = []
    msp_input = np.zeros(n + I, dtype=np.int32)
    msp_input[:I] = 3
    msp_input[I:I + R] = 2
    msp_input[I + R:I + R + D] = 1
    # gen_perm = list(msp(msp_input))
    gen_perm = msp(msp_input)
    for perm in gen_perm:
        idx = perm2idx(perm, n)
        output.append(idx)
    return output


# @deco_logging_time
def gen_base_substring(string, d, reduced=False):
    """
        This function uses the wildcard '?' which means allow any single character
        D, I, R means 'deletion', 'insertion' and 'replacement', respectively
        It prunes the
    :param string: given string
    :param d: delta
    :return: list of minimum base substrings
    """
    output = []
    if "?" in string:
        assert esc_qsmrk not in string
        string = string.replace("?", esc_qsmrk)
    assert "?" not in string
    n = len(string)
    string += '?'
    if reduced:
        indexed_perms = load_limited_minimal_perms(n=n, d=d)
    else:
        indexed_perms = load_get_all_perms(n=n, d=d)

    for idx in indexed_perms:
        string_pattern = "".join([string[x] for x in idx])
        output.append(string_pattern)
    return sorted(sorted(set(output)), key=lambda x: len(x))  # must sort set(output) by alphabetical order


def q_error(y, y_hat, reduction="mean"):
    if not isinstance(y, Iterable):
        y = [y]
    if not isinstance(y_hat, Iterable):
        y_hat = [y_hat]
    assert len(y) == len(y_hat)
    y = np.array(y, dtype=float)
    y_hat = np.array(y_hat, dtype=float)
    y[y <= 1.0] = 1
    y_hat[y_hat <= 1.0] = 1
    output = np.max(np.vstack([y / y_hat, y_hat / y]), axis=0)
    if reduction == "mean":
        output = float(np.mean(output))
    elif "q" == reduction[0]:  # q90 q10
        q_val = int(reduction[1:]) / 100
        output = np.quantile(output, q_val)
    return output


def q_error_np(y, y_hat, eps=1e-3, is_list=False):
    warnings.warn(FutureWarning("To be change"))

    if not isinstance(y, np.ndarray):
        y = np.array(y)
    if not isinstance(y_hat, np.ndarray):
        y_hat = np.array(y_hat)
    y_hat = y_hat.copy()
    y_hat[y_hat < 0] = 0.0
    y_eps = y + eps
    y_hat_eps = y_hat + eps
    if is_list:
        output = np.max(np.vstack([y_eps / y_hat_eps, y_hat_eps / y_eps]), axis=0)
    else:
        output = np.mean(np.max(np.vstack([y_eps / y_hat_eps, y_hat_eps / y_eps]), axis=0))
    return output


def set2min_hash_order(input_set, a, b, order_funcs, prime=(1 << 31) - 1):
    min_hash = ((np.expand_dims(np.array(input_set, dtype=np.uint64), axis=1) * a + b) % prime).min(axis=0)
    assert len(a) == len(b)
    assert len(a) == len(order_funcs)
    L = len(a)
    for i in range(L):
        order_func = order_funcs[i]
        min_hash[i] = order_func[int(min_hash[i])]
    return min_hash


def gen_min_hash_permutation(n, L, seed, prime=(1 << 31) - 1):
    A, B = gen_min_hash_coefficient(L, seed, prime)
    order_dicts = gen_min_hash_order_dict_list(n, A, B, prime)
    return A, B, order_dicts


def gen_min_hash_coefficient(L, seed=None, prime=(1 << 31) - 1):
    np.random.seed(seed)
    A = np.random.randint(1, prime, L, dtype=np.uint64)
    B = np.random.randint(0, prime, L, dtype=np.uint64)
    return A, B


def gen_min_hash_order_dict_list(n, A, B, prime):
    assert len(A) == len(B)
    L = len(A)
    order_dicts = []
    for i in range(L):
        # min_hash = ((np.expand_dims(val, axis=1) * A + B) % prime).min(axis=0)
        order_dict = {}
        hashes = (np.arange(n, dtype=np.uint64) * A[i] + B[i]) % prime
        heap = [x for x in hashes]
        heapq.heapify(heap)
        order_index = 0
        while len(heap) > 0:
            smallest = heapq.heappop(heap)
            order_dict[smallest] = order_index
            order_index += 1
            if debug_print:
                print("ordered_dict:", order_dict)
        assert len(order_dict) == n
        order_dicts.append(order_dict)
    return order_dicts


def average_list(x: list):
    return sum(x) / len(x)


def std_list(x: list):
    mean = average_list(x)
    var = average_list(list(map(lambda i: (i - mean) ** 2, x)))
    return np.sqrt(var)


class ConfigManager(Map):
    options = ["data", "qry", "alg", "save"]

    def __init__(self, file_path_or_dict=None):
        super().__init__()
        if file_path_or_dict is None:
            file_path_or_dict = {
                "data": {},
                "qry": {},
                "alg": {},
                "save": {}
            }
        self.load_conf(file_path_or_dict)

    def load_conf(self, file_path_or_dict):
        if isinstance(file_path_or_dict, str):
            confs = json.load(open(file_path_or_dict, "r"))
        else:
            confs = file_path_or_dict
        self.data = Map(confs["data"])
        self.qry = Map(confs["qry"])
        self.alg = Map(confs["alg"])
        self.save = Map(confs["save"])

    def save_conf(self, file_path):
        file_path = file_path
        confs = {"data": self.data, "qry": self.qry, "alg": self.alg, "save": self.save}
        json.dump(confs, open(file_path, "w"), indent=2)

    def get_dict(self):
        output = {}
        for key, val in self.__dict__.items():
            output[key] = dict(val)
        return output

    def __str__(self):
        return str(self.__dict__)


def is_general_substring_build(pattern):
    wc = "?"
    assert '\\' not in pattern
    for chr in meta_chars:
        if chr in [wc, '\\']:
            continue
        pattern = pattern.replace(chr, f"\\{chr}")
    pattern = pattern.replace(wc, ".")
    return re.compile(pattern)


def is_general_substring(pattern, text):
    pattern_compiled = is_general_substring_build(pattern)
    # wc = "?"
    # assert '\\' not in pattern
    # for chr in meta_chars:
    #     if chr in [wc, '\\']:
    #         continue
    #     pattern = pattern.replace(chr, f"\\{chr}")
    # pattern = pattern.replace(wc, ".")
    return pattern_compiled.search(text) is not None


def makedirs(file_path):
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)


def char_set_from_db(db, max_char=None):
    """

    :param db:
    :param max_char: if max_char is None, return all characters. Otherwise, return at most max_char characters by their
        frequencies
    :return:
    """
    if max_char:
        char_dict = dict()
        for rid, record in enumerate(db):
            for char in record:
                if char not in char_dict:
                    char_dict[char] = 0
                char_dict[char] += 1
        char_set = list(zip(*sorted(char_dict.items(), key=lambda x: x[1], reverse=True)[:max_char]))[
            0]  # frequency order
    else:
        char_set = set()
        for record in db:
            for char in record:
                char_set.add(char)
        char_set = sorted(char_set)  # alphabetical order
    return char_set


def char_dict_from_db(db, max_char=None):
    """

    Args:
        db: list of strings

    Returns:
        dictionary whose key and value are character and index, respectively.
        The index 0 is kept for [PAD] token.
        The index 1 is also kept for [UNK] token.
    """
    char_set = char_set_from_db(db, max_char=max_char)
    # char_list = list(char_set)
    char_dict = dict()
    # if max_char is not None and len(char_list) > max_char:
    # char_list.insert(0, "[UNK]")
    # char_list.insert(0, "[PAD]")

    for i, char in enumerate(char_set):
        # char_dict[char] = i + 1 # for [PAD]
        # if max_char is None or len(char_set) < max_char:
        #     char_dict[char] = i + 1  # for [PAD]
        # else:
        char_dict[char] = i + 2  # for [PAD] and [UNK]
    return char_dict


def get_class_name(cls):
    return str(cls.__class__).split("'")[1].split(".")[-1]


def string_query_encoding(q_s, char_dict, d=0):
    # assert "[PAD]" in char_dict
    output = []
    n_char = len(char_dict) + 1  # +1 means [UNK] token
    for char in q_s:
        idx = 1  # initialize [UNK] token
        if char in char_dict:
            idx = char_dict[char]
        idx += n_char * d
        output.append(idx)
    return output


def string_query_decoding(input_seq, char_list):
    n_char = len(char_list)
    char_list = sorted(char_list)
    idx = input_seq[0]
    d = idx // n_char
    q_s = []
    for idx in input_seq:
        idx -= d * n_char
        q_s.append(char_list[idx])
    q_s = "".join(q_s)
    return q_s, d


def get_dbsize_from_dsize(dsize):
    if dsize == "full":
        n = 1000000
    elif dsize == "mid":
        n = 500000
    elif dsize == "small":
        n = 10000
    elif dsize == "tiny":
        n = 100
    elif dsize == "max":
        n = 0
    else:
        raise ValueError(dsize)
    return n


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def get_size(obj, seen=None):
    """Recursively finds size of objects in bytes"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if hasattr(obj, '__dict__'):
        for cls in obj.__class__.__mro__:
            if '__dict__' in cls.__dict__:
                d = cls.__dict__['__dict__']
                if inspect.isgetsetdescriptor(d) or inspect.ismemberdescriptor(d):
                    size += get_size(obj.__dict__, seen)
                break
    if isinstance(obj, dict):
        size += sum((get_size(v, seen) for v in obj.values()))
        size += sum((get_size(k, seen) for k in obj.keys()))
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum((get_size(i, seen) for i in obj))

    if hasattr(obj, '__slots__'):  # can have __slots__ with __dict__
        size += sum(get_size(getattr(obj, s), seen) for s in obj.__slots__ if hasattr(obj, s))

    return size


def print_torch_summarize(model):
    summarized_str, n_total_params = torch_summarize(model)
    for line in summarized_str.split('\\n'):
        print(line)
    print("total params:", n_total_params)


def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    assert isinstance(model, nn.Module)
    tmpstr = model.__class__.__name__ + ' (\n'
    total_params = 0
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr, params = torch_summarize(module)
        else:
            modstr = module.__repr__()
            params = sum([np.prod(p.size()) for p in module.parameters()])
        modstr = _addindent(modstr, 2)

        params_back = sum([np.prod(p.size()) for p in module.parameters()])
        assert params == params_back
        total_params += params
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr += ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr, total_params


def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


def pack_bin_key_set_elems(key, elems):
    encoded_key = key.encode(encoding="utf8")
    output = struct.pack("i", len(encoded_key)) + encoded_key
    output += struct.pack("i", len(elems)) + struct.pack("i" * len(elems), *elems)
    return output


def generator_unpack_bin_key_set_elems(fp):
    first_readed = fp.read(4)
    while len(first_readed) > 0:
        len_key = struct.unpack("i", first_readed)[0]
        key = fp.read(len_key).decode(encoding='utf8')
        len_elems = struct.unpack("i", fp.read(4))[0]
        elems = struct.unpack("i" * len_elems, fp.read(len_elems * 4))
        yield key, list(elems)
        first_readed = fp.read(4)


def pack_bin_key_hash_elems(key, elems):
    return pack_bin_key_set_elems(key, elems)


def generator_unpack_bin_key_hash_elems(fp):
    return generator_unpack_bin_key_set_elems(fp)


def read_bin_eqt_set(N, n, name):
    kwargs = {"N": N, "n": n, "name": name}
    file_path = kwargs_and_function_to_file_path(load_extended_ngram_table_set, kwargs, ext="bin")
    rel_path = os.path.realpath(file_path)
    assert os.path.exists(file_path), file_path
    assert os.path.getsize(file_path) > 0, f"{file_path} is empty"

    eqt_set = {}
    print(f"while loading result to path {rel_path} ...")
    with open(file_path, "rb") as f:
        for key, elems in tqdm(generator_unpack_bin_key_set_elems(f)):
            elems = list(elems)
            eqt_set[key] = elems
            del elems
    print("done")
    return eqt_set


def write_bin_eqt_set(eqt_set, N, n, name):
    kwargs = {"N": N, "n": n, "name": name}
    file_path = kwargs_and_function_to_file_path(load_extended_ngram_table_set, kwargs, ext="bin")
    rel_path = os.path.realpath(file_path)
    print(f"while saving result to path {rel_path} ...")
    with open(file_path, "wb") as f:
        for key, elems in tqdm(eqt_set.items(), total=len(eqt_set)):
            f.write(pack_bin_key_set_elems(key, elems))
            del elems


def generator_read_bin_eqt_hash(N, n, name, L, PT, seed):
    kwargs = {"N": N, "n": n, "name": name}
    file_path = kwargs_and_function_to_file_path(load_extended_ngram_table_set, kwargs, ext="bin")
    rel_path = os.path.realpath(file_path)
    assert os.path.exists(file_path), file_path
    assert os.path.getsize(file_path) > 0, f"{file_path} is empty"
    eqt_set = {}
    print(f"while loading result to path {rel_path} ...")
    with open(file_path, "rb") as f:
        for key, elems in tqdm(generator_unpack_bin_key_set_elems(f)):
            if len(elems) < PT:
                continue
            elems = list(elems)
            eqt_set[key] = elems
            yield key, elems
            del elems
    print("done")


def read_bin_eqt_hash(N, n, name, L, PT, seed):
    kwargs = {"N": N, "n": n, "name": name, "L": L, "PT": PT, "seed": seed}
    file_path = kwargs_and_function_to_file_path(load_extended_ngram_table_hash, kwargs, ext="bin")
    rel_path = os.path.realpath(file_path)
    assert os.path.exists(file_path), file_path
    assert os.path.getsize(file_path) > 0, f"{file_path} is empty"
    eqt_hash = {}
    print(f"while loading result to path {rel_path} ...")
    with open(file_path, "rb") as f:
        for key, elems in tqdm(generator_unpack_bin_key_set_elems(f)):
            elems = list(elems)
            count = elems[0]
            assert count >= PT, f"count: {count}, PT: {PT}, should count>=PT"
            eqt_hash[key] = elems
            del elems
    print("done")
    return eqt_hash


def generator_eqt_hash_by_reading_bin_eqt_set(N, n, name, L, PT, seed):
    """
        PT applied

    Args:
        N:
        n:
        name:
        L:
        PT:
        seed:

    Returns:

    """
    kwargs = {"N": N, "n": n, "name": name}
    file_path = kwargs_and_function_to_file_path(load_extended_ngram_table_set, kwargs, ext="bin")
    assert os.path.exists(file_path), file_path
    assert os.path.getsize(file_path) > 0, f"{file_path} is empty"
    print(f"while loading result to path {file_path} ...")
    A, B, order_dicts = gen_min_hash_permutation(n, L, seed)
    with open(file_path, "rb") as f:
        for key, elems in generator_unpack_bin_key_set_elems(f):
            # elems = list(elems)
            count = len(elems)
            if count < PT:
                continue
            hash = set2min_hash_order(elems, A, B, order_dicts)
            sig = [count, *hash]
            yield key, sig
            del elems
            del hash

    print("done")


def write_bin_eqt_hash(db, N, n, name, L, PT, seed, over_write=False):
    """
    should run eqt_set once

    Args:
        eqt_hash: if eqt_hash is None, load from eqt_set
        N:
        n:
        name:
        L:
        PT:
        seed:

    Returns:

    """
    kwargs = {"N": N, "n": n, "name": name, "L": L, "PT": PT, "seed": seed}
    file_path = kwargs_and_function_to_file_path(load_extended_ngram_table_hash, kwargs, ext="bin")
    # load_extended_ngram_table_set(db, n, name, N, over_write=over_write)
    load_extended_ngram_table_set(db, n, name, N, over_write=over_write)
    eqt_gen = generator_eqt_hash_by_reading_bin_eqt_set(N, n, name, L, PT, seed)
    total_len = None
    # key, elems = next(eqt_gen)

    print(f"while saving result to path {file_path} ...")
    with open(file_path, "wb") as f:
        # f.write(pack_bin_key_set_elems(key, elems))
        for key, elems in tqdm(eqt_gen, total=total_len):
            count = elems[0]
            assert count >= PT
            f.write(pack_bin_key_set_elems(key, elems))
            del elems


def keras_pad_sequences(sequences, maxlen=None, dtype='int32',
                        padding='pre', truncating='pre', value=0.):
    """Pads sequences to the same length.

    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.

    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the end.

    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.

    Pre-padding is the default.

    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value.

    # Returns
        x: Numpy array with shape `(len(sequences), maxlen)`

    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    for x in sequences:
        try:
            lengths.append(len(x))
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, str) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def clip_query_string_each_len(x, prob):
    # for CardNet dataset
    assert isinstance(prob, float)
    if len(x) == 0 or prob == 1:
        return x
    group = {}
    for rec in x:
        query = rec
        if len(query) not in group:
            group[len(query)] = []
        group[len(query)].append(rec)

    output = []
    for length, g in group.items():
        prob2 = max(prob, 1 / len(g))
        selected, discarded = train_test_split(g, train_size=prob2, random_state=0)
        output.extend(selected)

    # d = len(x[0]) - 2  # exclude string and d=0
    # df = pd.DataFrame(x, columns=['str', *['d' + str(x) for x in range(d + 1)]])
    # df['len'] = [len(x) for x in df['str']]

    # group = df.groupby(['len'])
    # for df_tuple in group:
    #     df_part: pd.DataFrame = df_tuple[1]
    #     selected_df = df_part.sample(frac=prob, random_state=0).drop('len', axis=1)
    #     output.extend(selected_df.values.tolist())
    return output


def pack_padded_sequence_by_true_values(y, y_hat):
    y = np.array(y)
    y[y < 0] = 0
    output_y, output_y_hat = y[y.nonzero()], y_hat[y.nonzero()]
    assert all([d == d_hat for d, d_hat in zip(y.shape, y_hat.shape)])
    return output_y, output_y_hat


def pack_padded_sequence(x, lengths):
    output = []
    assert len(x) == len(lengths), f"length should be matched, but we get {len(x)} and {len(lengths)}"
    for row, length in zip(x, lengths):
        output.extend(row[:length])
    return np.array(output)


def get_git_revision_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().rstrip()


def get_git_revision_short_hash():
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().rstrip()


def get_git_commit_message():
    return subprocess.check_output(["git", "show", "-s", "--format=%s"]).decode().rstrip()


def status_string(status_code):
    return status_dict[status_code]


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


def get_join_exp_dict(line):
    header = ["log_time", "alg", "n_qry", "n_rec", "n_prfx", "d", "p", "time"]
    types = [str, str, int, int, int, int, bool, float]
    log_time, infos = line.rstrip()[1:].split("]")
    values = [log_time]
    values.extend(infos.split())
    print(values)
    exp_dict = {}
    for i, (k, v) in enumerate(zip(header, values)):
        exp_dict[k] = types[i](v)
    if exp_dict["time"] <= 0.0:
        exp_dict["time"] = None
    return exp_dict


def get_join_exp_json_str(line):
    exp_dict = get_join_exp_dict(line)
    return json.dumps(exp_dict)


def get_model_exp_json_str(model, time=None, size=None, err=None, log_time=None, query=None, prefix=None, update=None,
                           **kwargs):
    exp_dict = get_model_exp_dict(model, time, size, err, log_time, query, prefix, update, **kwargs)
    return json.dumps(exp_dict)


def distinct_prefix(strings):
    prfx_set = set()
    for string in strings:
        for i in range(1, len(string) + 1):
            prfx_set.add(string[:i])
    return prfx_set


def num_distinct_prefix(strings):
    return len(distinct_prefix(strings))


def clip_by_limiting_number_of_prefixes(strings, max_count):
    prfx_set = set()
    fail_idx = 0
    for string in strings:
        for pos in range(1, len(string) + 1):
            prfx_set.add(string[:pos])
        n_prfx = len(prfx_set)
        if n_prfx > max_count:
            break
        fail_idx += 1
    return strings[:fail_idx]


def get_parser_with_ignores():
    parser = argparse.ArgumentParser(description="parse")
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--dname', type=str, help='data name')
    # parser.add_argument('--n-train', type=int, help='number of training data')
    parser.add_argument('--p-train', type=float, help='ratio of augmented training data')
    parser.add_argument('--p-val', type=float, help='ratio of valid')
    parser.add_argument('--p-test', type=float, help='ratio of test')
    parser.add_argument('--bi-direct', action='store_true', help='bi-directional LSTM for RNN method')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--n-rec', type=int, help='number of records')
    group.add_argument('--dsize', type=str, choices=['tiny', 'small', 'mid', 'full', 'max'], help="data scale")
    parser.add_argument('--seed', type=int, help='estimator seed')
    # parser.add_argument('--short', action='store_true', help='short train test query')
    # parser.add_argument('--ncores', default=6, type=int, help='number of cores to multi-process')
    parser.add_argument('--l2', type=float, help='L2 regularization ')
    parser.add_argument('--lr', type=float, help='train learning rate [default (RNN=0.001), (CardNet=0.001)]')
    parser.add_argument('--vlr', type=float, help='train learning rate for VAE in CardNet [default 0.0001]')
    parser.add_argument('--swa', action='store_true', help='apply stochastic weight averaging')
    # parser.add_argument('--rewrite', action='store_true', help='postprocessing from estimated result')
    parser.add_argument('--multi', action='store_true', help='multi RNN for each delta')
    # parser.add_argument('--card', action='store_true', help='total data split by cardnet manner')
    # parser.add_argument('--card', default=True, type=bool,
    #                     help='total data split by cardnet manner. '
    #                          'If it is False, we will partition the training data for each length and delta')
    # parser.add_argument('--fullstr', action='store_true', help='full string mode')
    parser.add_argument('--layer', type=int, help="number of RNN layers")
    parser.add_argument('--pred-layer', type=int, help="number of pred layers")
    parser.add_argument('--cs', type=int, help="rnn cell size (it should be even)")
    parser.add_argument('--csc', type=int, help="CardNet model scale (default=256)")
    parser.add_argument('--vsc', type=int, help="CardNet vae model scale (default=128)")
    # parser.add_argument('--n-channel', default=32, type=int, help="number of output channels")
    parser.add_argument('--Ntbl', type=int, help='maximum length of extended n-gram table for LBS (default=5)')
    parser.add_argument('--PT', type=int, help='threshold for LBS (PT>=1) (default=20) ')
    parser.add_argument('--max-epoch', type=int,
                        help="maximum epoch (default=100 for RNN 800 for CardNet)")
    parser.add_argument('--patience', type=int, help="patience for training neural network")
    parser.add_argument('--min-l', type=int, help="minimum length of query")
    parser.add_argument('--max-l', type=int, help="maximum length of query")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--max-d', type=int, help="maximum distance threshold")
    group.add_argument('--delta', type=int, help="single distance threshold")
    parser.add_argument('--max-char', type=int, help="maximum # of characters to support (default=200)")
    parser.add_argument('--sep-emb', action='store_true', help="char dist sep embed?")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--prfx', action='store_true', help="additional prefix information")
    group.add_argument('--Sprfx', action='store_true', help="additional prefix & separate prefixes")
    group.add_argument('--Eprfx', action='store_true', help="additional prefix & enumerate prefixes")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--btS', action='store_true', help="bulk training with sharing")
    group.add_argument('--btA', action='store_true', help="additionally bulk training")
    parser.add_argument('--Mprfx', action='store_true', help="limit the number of additional prefixes")

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
    parser.add_argument('--clip-gr', type=float, help='estimation model hard value clipping on gradient')
    parser.add_argument('--n-heads', type=int, help='number of heads for attention models')

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
    parser.add_argument(analysis_option, action='store_true', help="analysis model")
    analysis_option = '--latency'
    ignore_opt_list.append(analysis_option)
    parser.add_argument(analysis_option, action='store_true', help="latency evaluation")
    rewrite_option = '--rewrite'
    ignore_opt_list.append(rewrite_option)
    parser.add_argument(rewrite_option, action='store_true', help='postprocessing from estimated result')

    return parser, ignore_opt_list


def get_model_args(verbose=1, mode="test"):
    parser, ignore_opt_list = get_parser_with_ignores()

    args = parser.parse_args()

    cmd_list = sys.argv[1:]
    parser_opts = [action.option_strings[-1] for action in parser._actions[1:]]

    for opt in ignore_opt_list:
        if opt in cmd_list:
            cmd_list.remove(opt)
        if opt in parser_opts:
            parser_opts.remove(opt)

    cmd_opts = [opt for opt in cmd_list if opt.startswith("-")]
    parser_opts_filtered = list(filter(lambda x: x in cmd_opts, parser_opts))

    if not args.outdata:
        assert len(cmd_opts) == len(
            parser_opts_filtered), f"# of filtered options should be equal to # of cmd_opts, {len(cmd_opts)}, {len(parser_opts_filtered)}\n>>> input: {' '.join(cmd_opts)} \n>>> orgin: {' '.join(parser_opts_filtered)}"
        assert all([cmd_opt == parser_opt for cmd_opt, parser_opt in zip(cmd_opts, parser_opts_filtered)]), \
            f"option order should follow the usage of the program\n>>> input: {' '.join(cmd_opts)} \n>>> orgin: {' '.join(parser_opts_filtered)}"
    exp_key = " ".join(cmd_list)
    if verbose:
        print("[exp_key]:", exp_key)
        print(f">>> model:{exp_key}")
        print(f">>> est:{exp_key}")
    return args, exp_key


def is_learning_model(alg):
    return "rnn" in alg or "card" in alg or "attn" in alg


def varify_args(args):
    # if model_name == 'card':
    #     assert args.card
    model_name = args.model
    assert args.model is not None
    assert args.model in ['rnn', 'eqt', 'card', 'attn']

    assert args.dname is not None
    assert args.dname in ['imdb', 'wiki', 'tpch', 'imdb2', 'wiki2',
                          'tpch2', 'dblp', 'dblps', 'dblpTL', 'dblpAU', 'egr1', 'dblpa']

    assert args.n_rec is not None or args.dsize is not None
    assert args.seed is not None

    assert args.max_l is not None

    assert args.max_d is not None or args.delta is not None

    assert args.p_test > 0 and args.p_test <= 1.0

    if args.n_rec is None:
        assert args.dsize is not None, args.dsize

    if args.prfx:
        assert 'rnn' in model_name or 'attn' in model_name, model_name
        # assert args.dname == 'wiki', args.dname

    if args.csc or args.vsc:
        assert model_name == 'card', model_name

    if is_learning_model(args.model):
        assert args.l2 is not None
        assert args.lr is not None
        assert args.max_epoch is not None
        assert args.patience is not None
        assert args.max_char is not None
        assert args.bs is not None
        assert args.p_train is not None
        assert args.p_val is not None
        assert args.clip_gr is not None

        assert args.p_val >= 0, args.p_val <= 1.0
        assert args.p_val + args.p_test < 1
        assert args.p_train > 0, args.p_train <= 1.0
    else:
        assert args.l2 is None
        assert args.lr is None
        assert not args.swa
        assert args.max_epoch is None
        assert args.patience is None
        assert args.max_char is None
        assert args.bs is None
        assert args.p_train is None
        assert args.p_val is None
        assert args.clip_gr is None

    if 'rnn' in args.model:
        assert args.cs is not None
        assert args.layer is not None
        assert args.pred_layer is not None
        assert args.sep_emb == True
        assert args.h_dim is not None
        assert args.es is not None
    elif 'attn' in args.model:
        assert args.cs is not None
        assert args.pred_layer is not None
        assert args.sep_emb == True
        assert args.h_dim is not None
        assert args.es is not None
        assert args.n_heads is not None
    else:
        assert not args.bi_direct
        assert args.cs is None
        assert args.layer is None
        assert args.pred_layer is None
        assert not args.sep_emb
        assert not args.prfx
        assert args.h_dim is None
        assert args.es is None

    if 'card' in args.model:
        assert args.csc is not None
        assert args.vsc is not None
        assert args.vlr is not None
        assert args.vbs is not None
        assert args.max_epoch_vae is not None
        assert args.vl2 is not None
        assert args.vclip_lv is not None
        assert args.vclip_gr is not None
    else:
        assert args.csc is None
        assert args.vsc is None
        assert args.vlr is None
        assert args.vbs is None
        assert args.max_epoch_vae is None
        assert args.vl2 is None
        assert args.vclip_lv is None
        assert args.vclip_gr is None

    if 'eqt' in args.model:
        assert args.Ntbl is not None
        assert args.PT is not None
        assert args.L is not None
    else:
        assert args.Ntbl is None
        assert args.PT is None
        assert args.L is None


def get_n_rec_from_args(args):
    if args.dsize:
        n_rec = get_dbsize_from_dsize(args.dsize)
    else:
        assert args.n_rec is not None
        n_rec = args.n_rec
    return n_rec


def get_exp_key(verbose=1):
    cmd_list = sys.argv[1:]
    exp_key = " ".join(cmd_list)
    if verbose:
        print("[exp_key]:", exp_key)
        print(f">>> model:{exp_key}")
        print(f">>> est:{exp_key}")

    return exp_key


def store_train_valid_query_string(q_train, q_valid, train_qry_outdir, args):
    dname = args.dname
    seed = args.seed
    p_train = args.p_train

    # store train_valid query string dataset
    q_train_valid = []
    q_train_valid.extend(q_train)
    q_train_valid.extend(q_valid)
    q_train_valid = sorted(q_train_valid)
    print(q_train_valid[:5])
    os.makedirs(train_qry_outdir, exist_ok=True)

    max_l = args.max_l if args.max_l else 20
    filename = f"qs_{dname}_{seed}_{max_l}_{p_train}.txt"
    train_qry_path = train_qry_outdir + filename
    if not os.path.exists(train_qry_path):
        print("saved at", train_qry_path)
        with open(train_qry_path, "w") as f:
            for qry in q_train_valid:
                f.write(qry + "\n")
    else:
        print("already exists at", train_qry_path)


def get_splited_train_valid_test_each_len(query_strings, split_seed, args):
    # query strings
    p_train = args.p_train
    p_valid = args.p_val
    p_test = args.p_test
    # seed = args.seed
    seed = split_seed

    query_len = [len(x) for x in query_strings]

    q_train, q_test = train_test_split(query_strings, test_size=p_test, random_state=seed, stratify=query_len)
    if p_valid is None:  # use all train data as valid data
        assert args.model == 'eqt'
        return None, q_train, q_test

    q_train_len = [len(x) for x in q_train]
    q_train, q_valid = train_test_split(q_train, test_size=p_valid / (1 - p_test), random_state=seed,
                                        stratify=q_train_len)

    # q_train: random shuffled
    q_train = clip_query_string_each_len(q_train, p_train)
    if args.Mprfx:
        dist_prfx = len(distinct_prefix(q_train))
        q_train = clip_by_limiting_number_of_prefixes(q_train, dist_prfx)
    return q_train, q_valid, q_test


def get_stat_query_string(q_train, args):
    n_qry = None
    n_prfx = None
    n_update = None
    model_name = args.model
    if model_name in ['rnn', 'card']:
        n_qry = len(q_train)
        n_prfx = num_distinct_prefix(q_train)
        n_update = n_qry
        if args.prfx:
            n_update = sum([len(qry) for qry in q_train])
            assert model_name not in ['card']
        if args.Sprfx:
            n_update = n_prfx
        if args.Eprfx:
            n_update = sum([len(qry) for qry in q_train])
    return n_qry, n_prfx, n_update


def verify_test_data_packed(test_data):
    try:
        for i, (q_s, y_list) in enumerate(test_data):
            assert isinstance(q_s, str)
            assert all([isinstance(x, int) for x in y_list])
            # assert len(delta_list) == len(y_list)
            if i == 100:
                break
    except:
        print(test_data[0])
        raise ValueError


def verify_sequence_test_data(test_data):
    try:
        for i, (q_s, y_seq) in enumerate(test_data):
            assert isinstance(q_s, str)
            assert all([len(y_seq[0]) == len(y_list) for y_list in y_seq])
            for y_list in y_seq:
                assert all([isinstance(x, int) for x in y_list])
            # assert len(delta_list) == len(y_list)
            if i == 100:
                break
    except:
        print(test_data[0])
        raise ValueError


def unpack_test_data(test_data):
    verify_test_data_packed(test_data)

    unpacked = []
    for q_s,  y_list in test_data:
        for delta, y in enumerate(y_list):
            unpacked.append((q_s, delta, y))
    return unpacked


def enumerate_sequence_test_data_to_packed(test_data):
    verify_sequence_test_data(test_data)

    packed = []
    ps_dict = set()
    for q_s,  y_seq in test_data:
        for i, y_list in enumerate(y_seq):
            p_s = q_s[:i+1]
            if p_s not in ps_dict:
                ps_dict.add(p_s)
                packed.append([p_s, y_list])
            # packed.append([p_s, y_list])
    return packed


def gen_LBS_model_path(cm):
    dconf = cm.data
    econf = cm.alg

    dname = dconf.name
    n = dconf.n
    N = econf.N
    L = econf.L
    PT = econf.PT
    seed = econf.seed  # for engram_hash & permutation
    engram_dict = {'n': n, 'name': dname, 'N': N, 'PT': PT, 'L': L, 'seed': seed}
    filepath = kwargs_and_function_to_file_path(load_extended_ngram_table_hash, engram_dict, "bin")
    return filepath


def check_nan_inf(input_tensor):
    if bool(torch.isnan(input_tensor).any()):
        assert False, "nan explode"
    if bool(torch.isinf(input_tensor).any()):
        assert False, "inf explode"


def is_grad_explode_in_model(model):
    is_explode = any([bool(torch.isnan(param.grad).any()) or bool(
        torch.isinf(param.grad).any()) for param in model.parameters()])
    return is_explode
