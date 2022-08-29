from src.util import Map, distinct_prefix, is_learning_model
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
import redis
import time
import yaml
import src.util as ut
import datetime
import numpy as np
import pandas as pd
import os
import seaborn as sns
from matplotlib import pyplot as plt
from abc import ABCMeta, abstractmethod
from typing import Type
import sys
import json
import matplotlib

matplotlib.use('Agg')


class Estimator(metaclass=ABCMeta):
    est_names = ["NN", "LBS", "NEW"]

    def __init__(self, conf):
        self.conf = conf
        self.db_factory: DBFactory = None
        self._SummaryWriter = None
        self._is_time = None
        self.name = None

    @abstractmethod
    def build(self, *args):
        pass

    @abstractmethod
    def model_size(self, *args):
        pass

    @abstractmethod
    def estimate(self, test_data):
        pass

    @abstractmethod
    def estimate_latency_anlysis(self, test_data):
        y_pred = []
        y_data = []
        latencies = []
        for test_qry in test_data:
            start = time.time()
            y_pred_s, y_data_s = self.estimate([test_qry])
            end = time.time()
            latency = end - start
            y_pred.append(y_pred_s)
            y_data.append(y_data_s)
            latencies.append([latency] * len(y_data_s))
        y_pred = np.vstack(y_pred)
        y_data = np.vstack(y_data)
        latencies = np.array(latencies, dtype=float)
        y_pred = y_pred.reshape(-1)
        y_data = y_data.reshape(-1)
        latencies = latencies.reshape(-1)
        return y_pred, y_data, latencies

    @property
    def summary_path(self):
        assert self.name is not None
        if self._is_time is not None and self._is_time:
            return f"log/{self.name}_{datetime.now().strftime('%m-%d-%H:%M')}"
        else:
            return f"log/{self.name}"

    @property
    def summaryWriter(self):
        if self._SummaryWriter is None:
            self._SummaryWriter = SummaryWriter(log_dir=self.summary_path, flush_secs=10)
        return self._SummaryWriter

    def set_db_factory(self, db_factory):
        self.db_factory = db_factory


class DBFactory(object):
    """
        conf: Map (extended dictionary) contains 'name'
    """

    def __init__(self, conf: Map):
        self.conf = conf
        self.db = None
        self.db = self.get_db()
        self.validate_conf()

    def get_db(self):
        if self.db is None:
            db = ut.read_string_db_by_name(self.conf.name)
            self.db = db
        return self.db

    def validate_conf(self):
        self.conf.n = len(self.db)


def learn_Deep_Learning_model_rewrite(model_name, dname, exp_name, logger=None):
    file_path = f"exp_result/{dname}/{exp_name}/estimate.csv"
    assert os.path.exists(file_path), file_path
    # df = pd.read_csv(file_path, )
    df = pd.read_csv(file_path, keep_default_na=False, na_values=[''])
    df['len'] = [len(x) for x in df['s_q']]
    df['q_err'] = [float(ut.q_error([x], [y])) for x, y in zip(df['true_count'], df['est_count'])]
    # warnings.filterwarnings(action='ignore')
    df['q_err10'] = [float(ut.q_error_np(x, y, eps=10)) for x, y in zip(df['true_count'], df['est_count'])]
    df['log_q_value'] = [float(np.log10(max(x, 1) / max(y, 1))) for x, y in zip(df['est_count'], df['true_count'])]
    df['(l,d)'] = [(x, y) for x, y in zip(df['len'], df['d'])]
    # warnings.filterwarnings(action='default')
    file_dir = os.path.dirname(file_path)
    # print(df)
    # print(df[['q_err', 'q_err10']].aggregate(['mean']))
    df.to_csv(file_dir + "/estimate_analysis.csv")
    count_ct = pd.crosstab(df['d'], df['len'], values=1, aggfunc='sum', margins=True)
    count_ct.to_csv(file_dir + "/count.csv")
    q_err_ct = pd.crosstab(df['d'], df['len'], values=df['q_err'], aggfunc='mean', margins=True)
    q_err_ct.to_csv(file_dir + "/q_err.csv")
    q_err10_ct = pd.crosstab(df['d'], df['len'], values=df['q_err10'], aggfunc='mean', margins=True)
    q_err10_ct.to_csv(file_dir + "/q_err10.csv")

    if logger is not None:
        logger.report_table("count", "PD with index", 0, table_plot=count_ct)
        logger.report_table("q_err", "PD with index", 0, table_plot=q_err_ct)
        logger.report_table("q_err10", "PD with index", 0, table_plot=q_err10_ct)

    if logger is not None:
        logger.report_image("log_q", "image Image", 0, local_path=file_path)

    if logger is not None:
        logger.report_image("log_q_vio", "image Image", 0, local_path=file_path)

    if "LBS" in model_name:
        file_path = file_dir + "/mof.csv"
        df = pd.read_csv(file_path)
        df['len'] = [len(x) for x in df['s_q']]
        df['q_err'] = [float(ut.q_error([x], [y])) for x, y in zip(df['true_count'], df['est_count'])]
        df['q_err10'] = [float(ut.q_error_np(x, y, eps=10)) for x, y in zip(df['true_count'], df['est_count'])]
        df['log_q_value'] = [float(np.log10(max(x, 1) / max(y, 1))) for x, y in zip(df['est_count'], df['true_count'])]
        file_path = file_dir + "/mof_exp.csv"
        df.to_csv(file_path, index=False)
        file_path = file_dir + "/mof_exp.xlsx"
        df.to_excel(file_path, index=False)
        if logger is not None:
            logger.report_table("MOF info", "PD with index", 0, table_plot=df)


def learn_Deep_Learning_model(cm, est_class: Type[Estimator], data, is_rewirte=False, logger=None, exp_key=None, overwrite=None, n_qry=None, n_prfx=None, n_update=None, analysis=None, analysis_latency=None):
    """

    Args:
        cm:
        est_class:
        data: (valid, test) or (train, valid, test)
        is_rewirte:

    Returns:

    """
    assert exp_key is not None
    # print(cm)
    # print(cm.data)
    model_name = cm.alg.name
    dname = cm.data.name
    exp_name = cm.save.exp_name
    res_dir = f"exp_result/{dname}/{exp_name}"
    conn = redis.Redis()
    # nr = cm.alg.neg_ratio

    if is_rewirte:
        learn_Deep_Learning_model_rewrite(model_name, dname, exp_name, logger)
        exit()

    if analysis and not analysis_latency:
        df = pd.read_csv(res_dir + "/estimate.csv")
        y = df['true_count'].to_list()
        y_hat = df['est_count'].to_list()
        q_error = ut.q_error(y, y_hat)
        q_error_dict = {}
        test_q_error = [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99, 100]
        for percentile in test_q_error:
            q_error_dict[f"q{percentile}"] = ut.q_error(y, y_hat, f"q{percentile}")
        exp_json_str = ut.get_model_exp_json_str(model_name, 0, err=q_error, **q_error_dict)
        conn.set(f"anal:{exp_key}", exp_json_str)
        exit()

    os.makedirs(res_dir, exist_ok=True)
    yaml.dump(cm.get_dict(), open(res_dir + "/conf.yaml", "w"))
    with open(res_dir + "/args.txt", "w") as f:
        f.write(" ".join(sys.argv) + "\n")

    is_learner = is_learning_model(model_name)
    if is_learner:
        train_data, valid_data, test_data = data
    else:
        valid_data, test_data = data
    if not analysis:  # save qs
        ut.verify_test_data_packed(test_data)
        df = pd.DataFrame(data=list(zip(*test_data))[0], columns=["s_q"])
        res_path = res_dir + f"/analysis_qs.csv"
        # print("saved at", res_path)
        df.to_csv(res_path, index=False)

    est = est_class(cm.alg)
    # print(f"Start building the {est.class_name} class")
    dfact = DBFactory(cm.data)
    # q_strat = QryGenStrategy(cm.qconf)
    # dfact.set_query_strategy(q_strat)
    est.set_db_factory(dfact)

    additional_info = {}
    start = time.time()
    if is_learner:
        train_data, valid_data, test_data = data
        model_dir = f"model/{dname}/{exp_name}"
        log_dir = f"log/{dname}/{exp_name}"

        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        yaml.dump(cm.get_dict(), open(model_dir + "/conf.yaml", "w"))
        est.save_dir = model_dir
        est.save_path = model_dir + "/saved_model.pth"
        est.logdir = log_dir
        if analysis:
            assert os.path.exists(est.save_path), f"analysis mode: {est.save_path} does not exists"
        est.build(train_data, valid_data, test_data, over_write=overwrite)
        if hasattr(est, "best_epoch"):
            additional_info["best_epoch"] = est.best_epoch
        if hasattr(est, "last_epoch"):
            additional_info["last_epoch"] = est.last_epoch
        if hasattr(est, "best_epoch_vae"):
            additional_info["best_epoch_vae"] = est.best_epoch_vae
        if hasattr(est, "last_epoch_vae"):
            additional_info["last_epoch_vae"] = est.last_epoch_vae
    else:
        valid_data, test_data = data
        est.resdir = res_dir
        est.model_path = ut.gen_LBS_model_path(cm)
        if analysis:
            assert os.path.exists(est.model_path), f"analysis mode: {est.model_path} does not exists"
        est.build(valid_data, over_write=overwrite)
    end = time.time()
    duration = end - start
    model_size = est.model_size()
    # print("additional info:", additional_info)
    if not analysis:
        exp_json_str = ut.get_model_exp_json_str(
            model_name, duration, size=model_size, query=n_qry, prefix=n_prfx, update=n_update, **additional_info)
        conn.set(f"model:{exp_key}", exp_json_str)

    pd_data = []
    pd_data.append(["s_q", "d", "true_count", "est_count"])
    start = time.time()
    max_d = cm.alg.max_d
    if "LBS" in cm.alg.name:
        est.silent = True
    if analysis:  # ts
        q_test = list(zip(*test_data))[0]
        if analysis and analysis_latency:
            test_data = ut.enumerate_sequence_test_data_to_packed(test_data)
            assert len(distinct_prefix(q_test)) == len(
                test_data), f"{len(distinct_prefix(q_test))} should be equal to {len(test_data)}"
        # print(test_data[:10])
        # exit()
    if "LBS" in cm.alg.name:
        test_data = ut.unpack_test_data(test_data)
        # print("LBS test data:", test_data[:5])

    num_qry_str = len(test_data)
    est.estimate_latency_anlysis(test_data[:10])
    # y_hat, y, latencies = est.estimate_latency_anlysis(valid_data)
    # print(ut.q_error(y, y_hat))
    y_hat, y, latencies = est.estimate_latency_anlysis(test_data)
    # print(test_data[:10])
    # y_hat, y= est.estimate_timeseries_anlysis(test_data)
    qry_list = np.array(list(zip(*test_data))[0])
    delta_list = list(range(max_d+1)) * num_qry_str
    if not "LBS" in cm.alg.name:
        qry_list = qry_list.repeat(max_d+1)

    pd_data = list(zip(qry_list, delta_list, y, y_hat, latencies))
    df = pd.DataFrame(data=pd_data, columns=["s_q", "d", "true_count", "est_count", "latency"])
    print("The estimated cardinalities are written as", res_dir + f"/analysis.csv")
    df.to_csv(res_dir + f"/analysis.csv", index=False)
    # else:
    #     y_hat, y = est.estimate(test_data)
    end = time.time()
    est_duration = end - start

    if "LBS" not in cm.alg.name:
        test_data = ut.unpack_test_data(test_data)

    pd_data = []
    pd_data.append(["s_q", "d", "true_count", "est_count"])
    for query, r_y, r_y_hat in zip(test_data, y, y_hat):
        s_q, d, y_origin = query
        assert y_origin == r_y
        pd_data.append([s_q, d, r_y, r_y_hat])
    # test_data = tmp_test_data

    if "LBS" in est.class_name:
        mof_data = []
        mof_data.append(["s_q", "d", "max_str_b", "max_est_count", "sim_est", "true_count", "est_count"])
        for r_test, r_mof, r_y, r_y_hat in zip(test_data, est.mof_list, y, y_hat):
            s_q, d, y_origin = r_test
            max_est_count, max_est_hash, max_str_b, sim_est = r_mof
            assert y_origin == r_y
            mof_data.append([s_q, d, max_str_b, max_est_count, sim_est, y_origin, r_y_hat])
        df_mof = pd.DataFrame(mof_data[1:], columns=mof_data[0])
        df_mof.to_csv(res_dir + "/mof.csv", index=False)
        df_mof.to_excel(res_dir + "/mof.xlsx", index=False)

    df = pd.DataFrame(data=pd_data[1:], columns=pd_data[0])
    if logger is not None:
        logger.report_table("estimate", "PD with index", 0, table_plot=df)

    df.to_csv(res_dir + "/estimate.csv", index=False)
    df.to_excel(res_dir + "/estimate.xlsx", index=False)
    # print("EXACT   :", y[:5])
    # print(f"{cm.alg.name}:", y_hat[:5])
    # q_error = ut.q_error_np(y, y_hat, eps=10.)
    q_error = ut.q_error(y, y_hat)
    q_error90 = ut.q_error(y, y_hat, "q90")
    print(f"average q-error: {q_error:.2f}")

    learn_Deep_Learning_model_rewrite(model_name, dname, exp_name, logger)

    exp_json_str = ut.get_model_exp_json_str(model_name, est_duration, err=q_error, q90=q_error90)
    if not analysis:
        conn.set(f"est:{exp_key}", exp_json_str)

    return est


if __name__ == "__main__":
    pass
