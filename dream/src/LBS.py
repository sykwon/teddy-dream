import os.path

import numpy as np
from tqdm import tqdm

import src.util as ut
from src.estimator import Estimator
from src.util import get_class_name


def estimate_coverage(query_result, engram_hash, N, engram, n_sample=100, seed=None):
    """
        query_result should be positive
    Args:
        query_result:
        engram_hash:
        N:
        n_sample:
        seed:

    Returns:

    """
    if len(query_result[0]) == 2:  # query, y_list for each delta
        query_result_bak = query_result
        query_result = []
        for query_string, cards in query_result_bak:
            for d, card in enumerate(cards):
                query_result.append([query_string, d, card])

    nq = len(query_result)
    n_sample = min(nq, n_sample)
    np.random.seed(seed)
    idx_sample = np.random.choice(range(nq), n_sample, replace=False)
    query_sample = [query_result[i] for i in idx_sample]
    fractions = []
    for q_s, q_d, q_r in tqdm(query_sample):
        # assert q_r > 0, (q_s, q_d, q_r)

        mbs_list = ut.gen_minimal_base_substring(q_s, q_d)
        n_max = 0
        for str_b in mbs_list:
            if len(str_b) <= N:
                if str_b in engram_hash:
                    mbs_count = engram_hash[str_b][0]
                    n_max = max(n_max, mbs_count)
            else:  # slicing
                # mbs_count = q_r
                sig = long_pattern_signature(str_b, engram, N)
                if len(sig) > 1:
                    mbs_count = sig[0]
                    # for s in range(len(str_b) - N + 1):
                    #     token = str_b[s:s + N]
                    #     if token in engram_hash:
                    #         mbs_count = min(mbs_count, engram_hash[token][0])
                    #     else:
                    #         mbs_count = 0
                    n_max = max(n_max, mbs_count)

        if q_r == 0:
            # assert n_max == 0
            continue
        else:
            # n_max = min(n_max, q_r)
            fraction = n_max / q_r
        if fraction > 0:
            fractions.append(fraction)
    assert len(fractions) > 0
    return ut.average_list(fractions)


def window_substring(string, w):
    output = []
    l = len(string)
    for i in range(l - w + 1):
        output.append(string[i: i + w])
    return output


def inter_sig_list_est(sig_list):
    hashes = []
    for x in sig_list:
        hashes.append(x[1:])
    hashes = np.array(hashes)
    # hash_union is sig for || A1 | A2 | ... ||
    return hashes.max(axis=0)


def union_sig_list(sig_list):
    hashes = []
    for x in sig_list:
        hashes.append(x[1:])
    hashes = np.array(hashes)
    # hash_union is sig for || A1 | A2 | ... ||
    return hashes.min(axis=0)


def get_sig_default(L):
    max_perm = (1 << 31)
    sig_default = [0]
    sig_default.extend([max_perm] * L)
    return sig_default


def intersection_estimation(sig_list):
    hashes = []
    for x in sig_list:
        hashes.append(x[1:])
    # MOF is (freq, sig) for maximum size
    MOF = max(sig_list, key=lambda x: x[0])
    # jacc_A_U ~= || A_max || / || A1 | A2 | ... ||
    hash_union = union_sig_list(sig_list)
    jacc_A_U = jaccard_similarity_estimation(hash_union, MOF[1:])

    min_jacc_A_U = MOF[0] / sum([x[0] for x in sig_list])
    if jacc_A_U <= min_jacc_A_U:
        jacc_A_U = min_jacc_A_U
    # union_freq ~= || A1 | A2 | ... ||
    union_freq = MOF[0] / jacc_A_U
    # jacc_I_U ~= ||A1 & A2 & ... || / || A1 | A2 | ... ||
    jacc_I_U = jaccard_similarity_estimation_multi(hashes)
    inter_freq = union_freq * jacc_I_U  # inter_freq ~= || A1 & A2 & A3 ||
    return inter_freq


def long_pattern_signature(string, eqt_hash, N, PST=None, is_LSH=False):
    """

    Args:
        string:
        eqt_hash:
        N:
        PST:

    Returns:
        if any token has no hash result and PST is not given, return []
        if any token has no hash result and PST is given, return [est_count]
        if all token has has result, [est_count, hash1, hash2, ..., hashL]
    """
    assert not is_LSH, "currently disabled"
    assert PST is None, "currently disabled"
    sig = []
    freq = None
    if PST is not None:
        freq = PST.MO_Count_with_wc(string)
        sig.append(freq)

    sig_list = []
    if len(string) <= N:
        print(Warning("too short"))
        token = string
        if token not in eqt_hash:
            return sig
        else:
            return eqt_hash[token]

    # len(string) > N
    for i in range(len(string) - N + 1):  # slicing for long string pattern
        token = string[i:i + N]
        if token not in eqt_hash:
            return sig
        sig_part = eqt_hash[token]
        sig_list.append(sig_part)
        if not is_LSH:  # EQT method
            overlap_token = token[:-1]
            assert overlap_token in eqt_hash
            if i == 0:
                freq = sig_part[0]
            else:
                sig_over = eqt_hash[overlap_token]
                c1 = sig_part[0]
                c2 = sig_over[0]
                freq *= c1 / c2

    assert len(sig_list) > 0
    assert len(sig_list) == len(string) - N + 1
    if freq is None:
        freq = intersection_estimation(sig_list)
    if PST is None:
        sig.append(freq)

    hash_inter = inter_sig_list_est(sig_list)
    sig.extend(hash_inter)
    return sig


def jaccard_similarity_estimation_multi(sig_list):
    n_total = len(sig_list[0])
    same_count = 0
    for sig in zip(*sig_list):
        is_same = True
        for elem in sig:
            if sig[0] != elem:
                is_same = False
                break
        if is_same:
            same_count += 1
    return same_count / n_total


def jaccard_similarity_estimation(hash1, hash2):
    assert len(hash1) == len(hash2)
    L = len(hash1)
    output = sum(map(lambda pair: pair[0] == pair[1], zip(hash1, hash2))) / L
    return output


def LBS_with_one_minima(engram: dict, queries, coverage, N=None, L=None, PST=None, is_LSH=False, y=None, res_dir=None,
                        PT=None, silent=False):
    """
    :param engram:
    :param queries:
    :param N: the maximum of ngram length
    :param L: hash size
    :return:
    """
    assert not is_LSH, "currently disabled"
    assert PST is None, "currently disabled"

    if N is None:
        N = max(map(len, engram.keys()))
    if L is None:
        L = len(engram.items()[0]) - 1

    estimations = []
    mof_bs_list = []
    if silent:
        query_iter = enumerate(queries)
    else:
        query_iter = tqdm(enumerate(queries), total=len(queries))
    for qid, (str_q, delta) in query_iter:
        # if len(str_q) == 3 and delta == 3:
        #     print("while debug")
        mbs = ut.gen_minimal_base_substring(str_q, delta)
        sig_list = []
        for str_b in mbs:
            if len(str_b) <= N:
                if str_b in engram:
                    sig = engram[str_b]
                    sig.append(str_b)
                    sig_list.append(sig)
            else:  # slicing
                sig = long_pattern_signature(str_b, engram, N)
                if len(sig) > 1:
                    if sig[0] > 0:
                        sig.append(str_b)
                        sig_list.append(sig)
        if len(sig_list) == 0:
            estimation = 0
            mof_info = [0, [-1] * L, r"\empty", 0]
        else:
            max_sig = max(sig_list, key=lambda x: x[0])
            max_card, max_min_hash, max_str_b = max_sig[0], max_sig[1:-1], max_sig[-1]
            for sig in sig_list:
                sig.pop()
            union_min_hash = np.array(sig_list, dtype=int)[:, 1:].min(axis=0)
            sim_est = jaccard_similarity_estimation(max_min_hash, union_min_hash)
            if sim_est == 0:
                sim_est = coverage
            estimation = max_card / sim_est
            mof_info = [max_card, max_min_hash, max_str_b, sim_est]

        mof_bs_list.append(mof_info)
        estimations.append(estimation)
    return estimations, mof_bs_list


class LBS(Estimator):
    def __init__(self, conf):
        super().__init__(conf)
        assert self.conf.usePST is None
        assert self.conf.is_LSH is None
        self.engram_hash = None
        self.class_name = get_class_name(self)
        self.coverage = None
        self.model_path = None
        self.silent = False

    def build(self, sample_data, over_write=False):
        dconf = self.db_factory.conf
        econf = self.conf
        # qconf = self.db_factory.query_strategy.conf
        dname = dconf.name
        # qstrat = self.db_factory.query_strategy
        # q_ratio = qconf.q_ratio
        # l_range = qconf.l_range
        # w = qconf.w
        n = dconf.n
        N = econf.N
        L = econf.L
        PT = econf.PT

        seed = econf.seed  # for engram_hash & permutation

        db = self.db_factory.get_db()
        self.engram_hash = ut.load_extended_ngram_table_hash(
            db, n=n, name=dname, N=N, PT=PT, L=L, seed=seed, over_write=over_write)
        # engram_dict = {'n': n, 'name': dname, 'N': N, 'PT': PT, 'L': L, 'seed': seed}
        # filepath = ut.kwargs_and_function_to_file_path(ut.load_extended_ngram_table_hash, engram_dict, "bin")
        # self.model_path = filepath

        # print("start calculating coverage")
        # pos_data = ut.load_total_positive_query_result(db, qstrat, n=n, dname=dname, q_ratio=q_ratio, l_range=l_range,
        #                                                w=w)
        # coverage = estimate_coverage(pos_data, self.engram_hash, N, seed=seed)
        # self.coverage = coverage
        # print("coverage:", coverage)
        # coverage = self.coverage

        print("start calculating coverage")
        if self.coverage is None:
            self.calculate_coverage(sample_data)
        coverage = self.coverage
        # assert self.coverage is not None
        print("coverage:", coverage)

        print("build from %s" % self.conf.name)

    def model_size(self, *args):
        size = os.path.getsize(self.model_path)
        return size

    def calculate_coverage(self, test_data, is_LSH=None, use_PST=None):
        assert not is_LSH, "currently disabled"
        assert not use_PST, "currently disabled"
        econf = self.conf
        N = econf.N
        L = econf.L
        PT = econf.PT
        seed = econf.seed
        assert self.engram_hash is not None
        print("start from querying")
        coverage = estimate_coverage(test_data, self.engram_hash, N, self.engram_hash, seed=seed)
        self.coverage = coverage
        return coverage

    def estimate_latency_anlysis(self, test_data):
        return super().estimate_latency_anlysis(test_data)

    def estimate(self, test_data, use_PST=None, is_LSH=None):
        assert not use_PST, "currently disabled"
        assert not is_LSH, "currently disabled"
        econf = self.conf
        N = econf.N
        L = econf.L
        PT = econf.PT
        seed = econf.seed

        engram_hash = self.engram_hash

        if not self.silent:
            print("start from querying")

        coverage = self.coverage
        if len(test_data[0]) == 2:  # query strings and cards for all threshold
            test_data_bak = test_data
            test_data = []
            for query_string, cards in test_data_bak:
                for d, card in enumerate(cards):
                    test_data.append([query_string, d, card])

        test_query = []
        query_result = []
        for q_s, d, y in test_data:
            test_query.append([q_s, d])
            query_result.append(y)

        if hasattr(self, "resdir"):
            resdir = self.resdir
        else:
            resdir = None

        query_estimate, mof_list = LBS_with_one_minima(engram_hash, test_query, coverage, N, L, PST=None, is_LSH=False,
                                                       y=query_result, res_dir=resdir, PT=PT, silent=self.silent)
        self.mof_list = mof_list
        assert len(mof_list) == len(query_estimate)

        return query_estimate, query_result
