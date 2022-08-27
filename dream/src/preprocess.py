import src.util as ut
import numpy as np
import os
import pandas as pd


def load_query_strings(dname, seed=None):
    file_path = f"data/qs_{dname}.txt"

    assert os.path.exists(file_path), file_path
    output = []
    with open(file_path) as f:
        for line in f:
            line = line.rstrip()
            # if len(line) <= 20:
            output.append(line)
    return output


def get_query_result(dname, max_d, prfx=False):
    max_d_training = 5
    assert(max_d < max_d_training)
    file_dir = f"data/res_{dname}_{max_d_training}/"
    if prfx:
        file_path = file_dir + "p_pref.csv"
    else:
        file_path = file_dir + "p.csv"
    assert os.path.exists(file_path), file_path
    output = {}
    query_results = pd.read_csv(file_path, na_filter=False)
    for row in query_results.itertuples():
        value = list(row[2:])
        if max_d is not None:
            value = value[:(max_d+1)]
        output[row[1]] = value
    return output


def fetch_cardinality(queries, query_results, max_d, prfx=False, test=False, Eprfx=False):

    output = []

    if Eprfx:
        tmp_queries = queries
        queries = []
        for query in tmp_queries:
            for i in range(1, len(query) + 1):
                queries.append(query[:i])
        np.random.seed(0)
        np.random.shuffle(queries)
        assert not prfx

    for query in queries:
        card = []
        if prfx and not test:
            for i in range(1, len(query) + 1):
                card.append(query_results[query[:i]][:max_d + 1])
        else:
            card = query_results[query][:max_d + 1]
        output.append([query, card])
    return output


def get_cardinalities_train_test_valid(q_train, q_valid, q_test, args):
    dname = args.dname
    # seed = args.seed
    analysis = args.analysis  # analysis
    analysis_latency = args.latency  # latency
    prfx_mode = args.prfx or args.Eprfx
    if analysis and analysis_latency:
        prfx_mode = True

    # delta = None if args.max_d == 3 else args.max_d
    delta = args.max_d
    query_results = get_query_result(dname, delta, prfx_mode)
    if analysis and analysis_latency:
        test_data = fetch_cardinality(q_test, query_results, max_d=args.max_d, prfx=True, test=False)
        assert len(q_test) == len(test_data), f"{len(q_test)}, {len(test_data)}"
        assert sum([len(x) for x in q_test]) == sum([len(v_list) for q_s, v_list in test_data]), "length sum"
    else:
        test_data = fetch_cardinality(q_test, query_results, max_d=args.max_d,
                                      prfx=args.prfx, test=True)
    valid_data = fetch_cardinality(q_valid, query_results, max_d=args.max_d,
                                   prfx=args.prfx, test=True)
    if q_train is None:
        train_data = None
    else:
        train_data = fetch_cardinality(q_train, query_results, max_d=args.max_d, prfx=args.prfx,
                                       Eprfx=args.Eprfx)

    return train_data, valid_data, test_data
