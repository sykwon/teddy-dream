import src.util as ut
import numpy as np
import os
import pandas as pd


def load_query_strings(dname, dsize, seed=None, max_l=None):
    assert dsize == "max"
    if seed is None:
        file_path = f"data/qs_{dname}.txt"
    else:
        file_path = f"data/qs_{dname}_{seed}.txt"
    if max_l is not None:
        assert seed is not None
        file_path = f"data/qs_{dname}_{seed}_{max_l}.txt"

    assert os.path.exists(file_path), file_path
    output = []
    with open(file_path) as f:
        for line in f:
            line = line.rstrip()
            # if len(line) <= 20:
            output.append(line)
    return output


def get_query_result(dname, dsize, prfx=False, seed=None, max_l=None, delta=None):
    if seed is None:
        file_dir = f"data/res_{dname}_{dsize}/"
    else:
        file_dir = f"data/res_{dname}_{seed}_{dsize}/"
    if max_l is not None:
        assert seed is not None
        file_dir = f"data/res_{dname}_{seed}_{max_l}_{dsize}/"
    if delta is not None:
        assert max_l is not None
        assert seed is not None
        file_dir = f"data/res_{dname}_{seed}_{max_l}_5_{dsize}/"
    if prfx:
        file_path = file_dir + "p_pref.csv"
    else:
        file_path = file_dir + "p.csv"
    assert os.path.exists(file_path), file_path
    output = {}
    query_results = pd.read_csv(file_path, na_filter=False)
    for row in query_results.itertuples():
        value = list(row[2:])
        if delta is not None:
            value = value[:(delta+1)]
        output[row[1]] = value
    return output


def fetch_cardinality(queries, query_results, max_d=None, prfx=False, test=False, Sprfx=False, Eprfx=False, btS=False, btA=False, delta=None):
    if max_d is None:
        max_d = 3

    if delta is not None:
        max_d = 0

    output = []

    assert not (btS and btA)
    bt_mode = btS or btA

    if bt_mode:
        if not test:
            query_set = set(queries)
        if btS:
            raise NotImplementedError
        assert not prfx

    if Sprfx:
        queries = list(ut.distinct_prefix(queries))
        np.random.seed(0)
        np.random.shuffle(queries)
        assert not prfx
        assert not Eprfx

    if Eprfx:
        tmp_queries = queries
        queries = []
        for query in tmp_queries:
            for i in range(1, len(query) + 1):
                queries.append(query[:i])
        np.random.seed(0)
        np.random.shuffle(queries)
        assert not prfx
        assert not Sprfx

    for query in queries:
        card = []
        if prfx and not test:
            # if test:
            #     # card.append(query_results[query][:max_d+1])
            #     # card = query_results[query][-1][:max_d + 1]
            #     card = query_results[query][:max_d + 1]
            #     # output.append([query, card])
            # else:
            for i in range(1, len(query) + 1):
                if delta is not None:
                    card.append(query_results[query[:i]][delta:delta + 1])
                else:
                    card.append(query_results[query[:i]][:max_d + 1])
                # for qr in query_results[query]:
                #     card.append(qr[:max_d + 1])
                # output.append([query, card])
                # card = query_results[query]
        elif bt_mode and not test:
            for i in range(1, len(query) + 1):
                prfx_qry = query[:i]
                if prfx_qry in query_set:
                    if delta is not None:
                        card.append(query_results[prfx_qry][delta:delta + 1])
                    else:
                        card.append(query_results[prfx_qry][:max_d + 1])
                else:
                    card.append([-1 for _ in range(max_d + 1)])
        else:
            if delta is not None:
                card = query_results[query][delta:delta + 1]
            else:
                card = query_results[query][:max_d + 1]
            # output.append([query, card])
        output.append([query, card])
    return output


def get_cardinalities_train_test_valid(q_train, q_valid, q_test, split_seed, args):
    dname = args.dname
    dsize = args.dsize
    # seed = args.seed
    seed = split_seed
    analysis = args.analysis  # analysis
    analysis_latency = args.latency  # latency
    prfx_mode = args.prfx or args.Sprfx or args.Eprfx
    if analysis and analysis_latency:
        prfx_mode = True

    # delta = None if args.max_d == 3 else args.max_d
    delta = args.max_d
    query_results = get_query_result(dname, dsize, prfx_mode, seed=seed, max_l=args.max_l, delta=delta)
    if analysis and analysis_latency:
        test_data = fetch_cardinality(q_test, query_results, max_d=args.max_d, prfx=True, test=False, delta=args.delta)
        assert len(q_test) == len(test_data), f"{len(q_test)}, {len(test_data)}"
        assert sum([len(x) for x in q_test]) == sum([len(v_list) for q_s, v_list in test_data]), "length sum"
    else:
        test_data = fetch_cardinality(q_test, query_results, max_d=args.max_d,
                                      prfx=args.prfx, test=True, delta=args.delta)
    valid_data = fetch_cardinality(q_valid, query_results, max_d=args.max_d,
                                   prfx=args.prfx, test=True, delta=args.delta)
    if q_train is None:
        train_data = None
    else:
        train_data = fetch_cardinality(q_train, query_results, max_d=args.max_d, prfx=args.prfx, Sprfx=args.Sprfx,
                                       Eprfx=args.Eprfx, btS=args.btS, btA=args.btA, delta=args.delta)

    return train_data, valid_data, test_data
