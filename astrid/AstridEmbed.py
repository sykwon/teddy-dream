import redis
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, dataset
import pandas as pd
import misc_utils
from string_dataset_helpers import TripletStringDataset, StringSelectivityDataset
import EmbeddingLearner
import SupervisedSelectivityEstimator
import argparse
import os
import prepare_datasets
import sys
import util as ut
import time
from shutil import copyfile
from tqdm import tqdm
import json
embedding_learner_configs, frequency_configs, selectivity_learner_configs = None, None, None
path = "datasets/DBLP/"
model_path = ""
# This assumes that prepare_dataset function was called to output the files.
# If not, please change the file names appropriately
train_triplet_prefix = None
train_file_name_prefix = "dblp_titles"
total_file_name_prefix = "dblp_titles"
model_name_prefix = None
split_seed = 0
delta = None
conn = redis.Redis()
exp_key = None
args = None

# This function gives a single place to change all the necessary configurations.
# Please see misc_utils for some additional descriptions of what these attributes mean


def setup_configs():
    global args
    global embedding_learner_configs, frequency_configs, selectivity_learner_configs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_learner_configs = misc_utils.AstridEmbedLearnerConfigs(embedding_dimension=args.es, batch_size=args.bs,
                                                                     num_epochs=args.emb_epoch, margin=0.2, device=device, lr=args.lr,
                                                                     channel_size=8)

    frequency_configs = misc_utils.StringFrequencyConfigs(
        string_list_file_name=path + total_file_name_prefix + ".csv",
        train_string_list_file_name=path + train_file_name_prefix + ".csv",
        selectivity_file_name=path + total_file_name_prefix + "_counts.csv",
        triplets_file_name=path + train_triplet_prefix + "_triplets.csv"
    )

    selectivity_learner_configs = misc_utils.SelectivityEstimatorConfigs(
        embedding_dimension=args.es, decoder_scale=args.dsc, batch_size=args.bs, num_epochs=args.epoch, device=device, lr=args.lr,
        # will be updated in train_selectivity_estimator
        min_val=0.0, max_val=1.0,
        embedding_model_file_name=model_path + model_name_prefix + f"embedding_model_{delta}.pth",
        selectivity_model_file_name=model_path + model_name_prefix + f"selectivity_model_{delta}.pth"
    )

    return embedding_learner_configs, frequency_configs, selectivity_learner_configs


# This function trains and returns the embedding model
def train_astrid_embedding_model(string_helper, model_output_file_name=None, overwrite=None, load_only=None):
    global embedding_learner_configs, frequency_configs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Some times special strings such as nan or those that start with a number confuses Pandas
    df = pd.read_csv(frequency_configs.triplets_file_name)
    # print("learn embedding model from the file --", frequency_configs.triplets_file_name)
    print("start pretraining embedding model")
    df["Anchor"] = df["Anchor"].astype(str)
    df["Positive"] = df["Positive"].astype(str)
    df["Negative"] = df["Negative"].astype(str)

    triplet_dataset = TripletStringDataset(df, string_helper)
    train_loader = DataLoader(
        triplet_dataset, batch_size=embedding_learner_configs.batch_size, shuffle=True)

    # --- return model without training for debug --- #
    # embedding_model = EmbeddingLearner.train_embedding_model(
    #     embedding_learner_configs, train_loader, string_helper, load_model=True)
    # torch.save(embedding_model.state_dict(), model_output_file_name)
    # return embedding_model
    if load_only:
        embedding_model = EmbeddingLearner.train_embedding_model(
            embedding_learner_configs, train_loader, string_helper, load_model=True)
        if os.path.exists(model_output_file_name):
            embedding_model.load_state_dict(torch.load(model_output_file_name, map_location=device))
        else:
            raise FileNotFoundError(model_output_file_name)
    else:
        if os.path.exists(model_output_file_name) and not overwrite:  # load
            embedding_model = EmbeddingLearner.train_embedding_model(
                embedding_learner_configs, train_loader, string_helper, load_model=True)
            embedding_model.load_state_dict(torch.load(model_output_file_name, map_location=device))
        else:  # train
            embedding_model = EmbeddingLearner.train_embedding_model(
                embedding_learner_configs, train_loader, string_helper)
            if model_output_file_name is not None:
                print("The pretrained embedding model are written as", model_output_file_name)
                os.makedirs(os.path.dirname(model_output_file_name), exist_ok=True)
                torch.save(embedding_model.state_dict(), model_output_file_name)
    return embedding_model


# This function performs min-max scaling over logarithmic data.
# Typically, the selectivities are very skewed.
# This transformation reduces the skew and makes it easier for DL to learn the models
def compute_normalized_selectivities(df):
    global selectivity_learner_configs
    normalized_selectivities, min_val, max_val = misc_utils.normalize_labels(
        df["selectivity"])
    df["normalized_selectivities"] = normalized_selectivities

    # namedtuple's are immutable - so replace them with new instances
    selectivity_learner_configs = selectivity_learner_configs._replace(
        min_val=min_val)
    selectivity_learner_configs = selectivity_learner_configs._replace(
        max_val=max_val)
    return df, selectivity_learner_configs


# This function trains and returns the selectivity estimator.
def train_selectivity_estimator(train_df, string_helper, embedding_model, model_output_file_name=None, overwrite=None, load_only=None):
    global selectivity_learner_configs, frequency_configs

    # --- sampling training data --- #
    # train_df = train_df.sample(frac=0.01)

    print("start training estimator model")
    if load_only:
        selectivity_model = load_selectivity_estimation_model(model_output_file_name, string_helper)
        selectivity_model = selectivity_model.to(selectivity_learner_configs.device)
    else:
        if os.path.exists(model_output_file_name) and not overwrite:  # load
            selectivity_model = load_selectivity_estimation_model(model_output_file_name, string_helper)
            selectivity_model = selectivity_model.to(selectivity_learner_configs.device)
        else:  # train
            string_dataset = StringSelectivityDataset(
                train_df, string_helper, embedding_model)
            train_loader = DataLoader(
                string_dataset, batch_size=selectivity_learner_configs.batch_size, shuffle=True)

            selectivity_model = SupervisedSelectivityEstimator.train_selEst_model(selectivity_learner_configs, train_loader,
                                                                                  string_helper)
            if model_output_file_name is not None:
                print("The estimated cardinalities are written as", model_output_file_name)
                os.makedirs(os.path.dirname(model_output_file_name), exist_ok=True)
                torch.save(selectivity_model.state_dict(), model_output_file_name)
    return selectivity_model


# This is a helper function to get selectivity estimates for an iterator of strings
def get_selectivity_for_strings(strings, embedding_model, selectivity_model, string_helper, analysis=None):
    global selectivity_learner_configs
    from SupervisedSelectivityEstimator import SelectivityEstimator
    embedding_model.eval()
    selectivity_model.eval()
    strings_as_tensors = []
    latencies = []
    normalized_predictions = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        if analysis is not None:
            for string in strings[:10]:
                string_as_tensor = string_helper.string_to_tensor(string).to(device)
                string_as_tensor = string_as_tensor.view(-1, *string_as_tensor.shape)
                selectivity_model(embedding_model(string_as_tensor))
            for string in strings:
                start = time.time()
                string_as_tensor = string_helper.string_to_tensor(string).to(device)
                # By default embedding mode expects a tensor of [batch size x alphabet_size * max_string_length]
                # so create a "fake" dimension that converts the 2D matrix into a 3D tensor
                string_as_tensor = string_as_tensor.view(-1, *string_as_tensor.shape)
                normalized_prediction = selectivity_model(embedding_model(string_as_tensor))
                end = time.time()
                latency = end - start
                latencies.append(latency)
                normalized_predictions.append(normalized_prediction)
            normalized_predictions = torch.tensor(normalized_predictions).to(device)
            denormalized_predictions = misc_utils.unnormalize_torch(normalized_predictions,
                                                                    selectivity_learner_configs.min_val,
                                                                    selectivity_learner_configs.max_val)
        else:
            # print("n_strings:", len(strings))  # 23924
            for string in strings:
                string_as_tensor = string_helper.string_to_tensor(string).to(device)
                # By default embedding mode expects a tensor of [batch size x alphabet_size * max_string_length]
                # so create a "fake" dimension that converts the 2D matrix into a 3D tensor
                string_as_tensor = string_as_tensor.view(-1, *string_as_tensor.shape)
                # print("string:", string_as_tensor.size()) # [1, 63, 20]
                strings_as_tensors.append(embedding_model(string_as_tensor).detach().cpu().numpy())
            strings_as_tensors = np.concatenate(strings_as_tensors)
            # normalized_selectivities= between 0 to 1 after the min-max and log scaling.
            # denormalized_predictions are the frequencies between 0 to N
            normalized_predictions = selectivity_model(
                torch.tensor(strings_as_tensors).to(device))
            denormalized_predictions = misc_utils.unnormalize_torch(normalized_predictions,
                                                                    selectivity_learner_configs.min_val,
                                                                    selectivity_learner_configs.max_val)
        return normalized_predictions, denormalized_predictions, latencies


def load_embedding_model(model_file_name, string_helper):
    from EmbeddingLearner import EmbeddingCNNNetwork
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_model = EmbeddingCNNNetwork(
        string_helper, embedding_learner_configs)
    embedding_model.load_state_dict(torch.load(model_file_name, map_location=device))
    return embedding_model


def load_selectivity_estimation_model(model_file_name, string_helper):
    from SupervisedSelectivityEstimator import SelectivityEstimator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selectivity_model = SelectivityEstimator(
        string_helper, selectivity_learner_configs)
    selectivity_model.load_state_dict(torch.load(model_file_name, map_location=device))
    return selectivity_model


def transform_df_to_my_df(df):
    df = df.copy()
    df = df.assign(d=args.delta)
    df = df.drop('normalized_selectivities', axis=1)
    df = df.rename(columns={'string': 's_q', 'selectivity': 'true_count'})
    df = df[['s_q', 'd', 'true_count']]
    return df


def split_old(df, random_seed):
    """
        split old dataset (prefix)
    """
    query_len = [len(x) for x in df["string"]]
    sampled_indices, test_indices = train_test_split(
        df.index, random_state=random_seed, test_size=0.1, stratify=query_len)
    df_sampled = df.iloc[sampled_indices, :]
    sampled_len = [len(x) for x in df_sampled["string"]]
    # train_indices, test_indices = train_test_split( df.index, random_state=random_seed, test_size=0.5)
    train_indices, _ = train_test_split(
        sampled_indices, random_state=random_seed, test_size=(1-0.2), stratify=sampled_len)
    train_df, test_df = df.iloc[train_indices], df.iloc[test_indices]
    return train_df, test_df


def get_indices_from_value_list(target_col, vals):
    indexes = []
    index_dict = {}
    for index, val in target_col.items():
        assert val not in index_dict
        index_dict[val] = index
    # for val in tqdm(vals, total=len(vals)):
    for val in vals:
        # match_idx = target_col.index[target_col == val].tolist()
        # assert len(match_idx) == 1
        match_idx = index_dict[val]
        indexes.append(match_idx)
    return indexes


def analysis_block1_query_string(df, train_df, test_df, test_qs):
    # global train_file_name_prefix
    global model_name_prefix
    test_qs = pd.DataFrame(test_qs)
    save_prefix = model_name_prefix.replace("qs_", "")
    dname = save_prefix.split("_")[0]
    log_dir = f"log/{dname}/{save_prefix}"
    os.makedirs(log_dir, exist_ok=True)

    df_my = transform_df_to_my_df(df)
    train_df_my = transform_df_to_my_df(train_df)
    test_df_my = transform_df_to_my_df(test_df)

    data_path = log_dir + f"data_{args.delta}.csv"
    train_path = log_dir + f"train_data_{args.delta}.csv"
    qs_path = log_dir + f"analysis_qs.csv"
    test_path = log_dir + f"test_data_{args.delta}.csv"
    # print("saved at:", data_path)
    # df_my.to_csv(data_path, index=False)
    test_qs.rename({0: 's_q'}, axis=1)
    # print("saved at:", qs_path)
    test_qs.to_csv(qs_path, index=False)
    # print("saved at:", train_path)
    # train_df_my.to_csv(train_path, index=False)
    # print("saved at:", test_path)
    # test_df_my.to_csv(test_path, index=False)
    return log_dir, df_my, train_df_my, test_df_my


def analysis_block2_estimations(test_df_my_prfx, denormalized_predictions, latencies, log_dir):
    # train_df_my = transform_df_to_my_df(train_df)
    if isinstance(denormalized_predictions, torch.Tensor):
        denormalized_predictions = denormalized_predictions.cpu()
    if isinstance(latencies, torch.Tensor):
        latencies = latencies.cpu()

    test_df_my_prfx['est_count'] = denormalized_predictions
    test_df_my_prfx['latency'] = latencies
    ts_path = log_dir + f"analysis_ts_{args.delta}.csv"
    print("The estimated cardinalities are written as", ts_path)
    test_df_my_prfx.to_csv(ts_path, index=False)


def main():
    model_name = "astrid"
    global args
    global split_seed
    misc_utils.initialize_random_seeds(args.seed)
    global embedding_learner_configs
    global frequency_configs
    global selectivity_learner_configs
    # Set the configs
    embedding_learner_configs, frequency_configs, selectivity_learner_configs = setup_configs()
    assert os.path.exists(
        frequency_configs.selectivity_file_name), frequency_configs.selectivity_file_name

    embedding_model_file_name = selectivity_learner_configs.embedding_model_file_name
    selectivity_model_file_name = selectivity_learner_configs.selectivity_model_file_name
    # print(embedding_model_file_name)
    # print(selectivity_model_file_name)

    additional_info = {}
    start_emb = time.time()
    # print("read query strings at", frequency_configs.string_list_file_name)  # total
    qs_df = pd.read_csv(frequency_configs.string_list_file_name, header=None,
                        keep_default_na=False, delimiter=",,,,,,,,,", engine='python')
    qs_df_train = pd.read_csv(frequency_configs.train_string_list_file_name, header=None,
                              keep_default_na=False, delimiter=",,,,,,,,,", engine='python')
    qs_df = qs_df.rename({0: 's_q'}, axis=1)
    qs_df_train = qs_df_train.rename({0: 's_q'}, axis=1)
    string_helper = misc_utils.setup_vocabulary(
        frequency_configs.string_list_file_name)

    # You can comment/uncomment the following lines based on whether you
    # want to train from scratch or just reload a previously trained embedding model.
    embedding_model = train_astrid_embedding_model(
        string_helper, embedding_model_file_name, overwrite=args.overwrite, load_only=args.analysis_latency)
    end_emb = time.time()
    # embedding_model = load_embedding_model(embedding_model_file_name, string_helper)

    # Load the input file and split into 50-50 train, test split
    # print("read training data at", frequency_configs.selectivity_file_name)  # total
    df = pd.read_csv(frequency_configs.selectivity_file_name, keep_default_na=False)
    # Some times strings that start with numbers or
    # special strings such as nan which confuses Pandas' type inference algorithm
    df["string"] = df["string"].astype(str)
    df, selectivity_learner_configs = compute_normalized_selectivities(df)

    # data split
    q_train, q_valid, q_test = get_splited_train_valid_test_each_len(list(qs_df['s_q']), split_seed, args)
    train_qs = q_train
    train_qs.extend(q_valid)
    test_qs = q_test
    # query_len = [len(x) for x in qs_df["s_q"]]
    # sampled_indices, test_indices = train_test_split(
    #     qs_df.index, random_state=split_seed, test_size=0.1, stratify=query_len)
    # qs_df_sampled = qs_df.iloc[sampled_indices, :]
    # sampled_len = [len(x) for x in qs_df_sampled["s_q"]]
    # train_indices, test_indices = train_test_split( df.index, random_state=random_seed, test_size=0.5)
    # train_indices, _ = train_test_split( sampled_indices, random_state=random_seed, test_size=(1-0.2), stratify=sampled_len)
    # train_indices = sampled_indices
    # train_qs_df, test_qs_df = qs_df.iloc[train_indices], qs_df.iloc[test_indices]
    # train_df = df[df["string"].isin(train_qs_df['s_q'])]
    # test_df = df[df["string"].isin(test_qs_df['s_q'])]
    # if args.analysis is None:
    # train_qs = train_qs_df['s_q']
    # train_qs = qs_df_train['s_q']
    # train_qs = qs_df_train['s_q']
    # print("# of train query string:", len(train_qs))
    train_indices_df = get_indices_from_value_list(df["string"], ut.distinct_prefix(train_qs))
    train_df = df.iloc[train_indices_df]

    # test_qs = test_qs_df['s_q']
    test_indices_df = get_indices_from_value_list(df["string"], test_qs)
    test_indices_df_prfx = get_indices_from_value_list(df["string"], ut.distinct_prefix(test_qs))
    # test_indices_df = get_indices_from_value_list(df["string"], test_qs)
    # print(ut.distinct_prefix(test_qs)[:20])
    test_df = df.iloc[test_indices_df]
    test_df_prfx = df.iloc[test_indices_df_prfx]
    # if args.analysis_latency:
    #     print(len(test_qs), len(test_df), len(test_df_prfx))

    log_dir, df_my, train_df_my, test_df_my_prfx = analysis_block1_query_string(df, train_df, test_df_prfx, test_qs)

    # You can comment/uncomment the following lines based on whether you
    # want to train from scratch or just reload a previously trained embedding model.
    start_est = time.time()
    selectivity_model = train_selectivity_estimator(train_df, string_helper,
                                                    embedding_model, selectivity_model_file_name, overwrite=args.overwrite, load_only=args.analysis_latency)
    end_est = time.time()
    duration_emb = end_emb - start_emb
    duration_est = end_est - start_est
    duration = duration_emb + duration_est
    emb_size = os.path.getsize(embedding_model_file_name)
    est_size = os.path.getsize(selectivity_model_file_name)
    model_size = emb_size + est_size

    additional_info["emb_size"] = emb_size
    additional_info["est_size"] = est_size
    additional_info["best_epoch"] = -1
    additional_info["last_epoch"] = -1
    additional_info["best_epoch_pt"] = -1
    additional_info["last_epoch_pt"] = -1
    additional_info["duration_emb"] = duration_emb
    additional_info["duration_est"] = duration_est
    n_qry = len(q_train)
    n_prfx = len(distinct_prefix(q_train))
    n_update = -1
    # print(additional_info)
    exp_json_str = ut.get_model_exp_json_str(model_name, duration, size=model_size,
                                             query=n_qry, prefix=n_prfx, update=n_update, **additional_info)
    if not args.analysis_latency:
        conn.set(f"model:{exp_key}", exp_json_str)

    # selectivity_model = load_selectivity_estimation_model(selectivity_model_file_name, string_helper)

    # --- to skip test code when we debug --- #
    # test_df = test_df.sample(frac=0.01)

    # Get the predictions from the learned model and compute basic summary statistics
    start = time.time()
    normalized_predictions, denormalized_predictions, latencies = get_selectivity_for_strings(
        test_df["string"].values, embedding_model, selectivity_model, string_helper, analysis=True)
    end = time.time()
    denormalized_predictions = denormalized_predictions.detach().cpu().numpy()

    normalized_predictions_prfx, denormalized_predictions_prfx, latencies_prfx = get_selectivity_for_strings(
        test_df_prfx["string"].values, embedding_model, selectivity_model, string_helper, analysis=True)
    # analysis
    analysis_block2_estimations(test_df_my_prfx, denormalized_predictions_prfx, latencies_prfx, log_dir)
    if args.analysis_latency:
        exit()
    est_duration = end - start
    actual = torch.tensor(test_df["normalized_selectivities"].values)
    actual_prfx = torch.tensor(test_df_prfx["normalized_selectivities"].values)
    # denormalized_actual = torch.tensor(test_df["selectivity"].values)
    # test_q_error_denom = misc_utils.compute_qerrors(denormalized_predictions, denormalized_actual,
    #                                           selectivity_learner_configs.min_val, selectivity_learner_configs.max_val)
    # test_q_error_denom = torch.vstack(test_q_error_denom).detach().cpu().numpy()
    test_q_error = misc_utils.compute_qerrors(normalized_predictions, actual,
                                              selectivity_learner_configs.min_val, selectivity_learner_configs.max_val)
    test_q_error = torch.vstack(test_q_error).detach().cpu().numpy()
    test_q_error_prfx = misc_utils.compute_qerrors(normalized_predictions_prfx, actual_prfx,
                                                   selectivity_learner_configs.min_val, selectivity_learner_configs.max_val)
    test_q_error_prfx = torch.vstack(test_q_error_prfx).detach().cpu().numpy()
    q_error = float(np.mean(test_q_error))
    q_error90 = float(np.quantile(test_q_error, q=0.9))
    q_error_prfx = float(np.mean(test_q_error_prfx))
    q_error90_prfx = float(np.quantile(test_q_error_prfx, q=0.9))
    print("average q-error: {:.2f}".format(np.mean(test_q_error)))
    # print("Test data: Summary stats of Loss: Percentile: [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99] ",
    #       [np.quantile(test_q_error, q) for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]])

    exp_json_str = ut.get_model_exp_json_str(model_name, est_duration, err=q_error,
                                             q90=q_error90, err_prfx=q_error_prfx, q90_prfx=q_error90_prfx)
    if not args.analysis_latency:
        conn.set(f"est:{exp_key}", exp_json_str)


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
    return output


def get_splited_train_valid_test_each_len(query_strings, split_seed, args):
    # query strings

    p_train = args.p_train
    p_valid = args.p_val
    p_test = args.p_test
    seed = split_seed

    query_len = [len(x) for x in query_strings]

    q_train, q_test = train_test_split(query_strings, test_size=p_test, random_state=seed, stratify=query_len)

    q_train_len = [len(x) for x in q_train]
    q_train, q_valid = train_test_split(q_train, test_size=p_valid / (1 - p_test), random_state=seed,
                                        stratify=q_train_len)

    # q_train: random shuffled
    q_train = clip_query_string_each_len(q_train, p_train)
    return q_train, q_valid, q_test


def save_train_query_strings_including_valid_data(path, total_prefix, train_prefix):
    qs_df = pd.read_csv(path + total_prefix + ".csv", header=None,
                        keep_default_na=False, delimiter=",,,,,,,,,", engine='python')
    qs_df = qs_df.rename({0: 's_q'}, axis=1)
    query_strings = list(qs_df['s_q'])
    q_train, q_valid, q_test = get_splited_train_valid_test_each_len(query_strings, split_seed, args)
    q_train.extend(q_valid)
    q_train = sorted(q_train)
    train_path = path + train_prefix + ".csv"
    # print("train(+valid) saved at", train_path)
    with open(train_path, 'w') as f:
        for query in q_train:
            f.write(query + "\n")
    # del q_valid
    # print(len(query_strings), len(q_train), len(q_valid), len(q_test))
    # print(q_test[:10], q_valid[:10])


if __name__ == "__main__":
    args, exp_key = ut.get_model_args(verbose=False)
    path = f"datasets/{args.dname}/"
    model_path = f"log/{args.dname}/"
    delta = args.delta
    assert args.p_val == 0.1
    assert args.p_test == 0.1
    if not args.analysis and not args.overwrite:
        res = conn.get(f"est:{exp_key}")
        res_model = conn.get(f"model:{exp_key}")
        res_conf = conn.get(f"conf:{exp_key}")
        if res is not None:
            assert res_model is not None
            print("This experiment is already done")
            print(res)
            print(res_model)
            print(res_conf)
            exit()

    max_d = 5
    assert args.delta <= max_d
    dname = args.dname
    qs_prefix = f"qs_{dname}"
    res_prefix = qs_prefix.replace("qs_", "res_")

    # load path
    total_qs_load_prefix = qs_prefix  # total: train + test
    train_qs_load_prefix = total_qs_load_prefix + f"_{args.p_train}"
    total_card_load_prefix = res_prefix + f"_{max_d}"

    # input path
    train_triplet_prefix = qs_prefix + f"_{args.delta}"
    total_file_name_prefix = qs_prefix + f"_{args.delta}"  # total: train + test
    train_file_name_prefix = total_file_name_prefix + f"_{args.p_train}"

    # model path
    model_name_prefix = dname + \
        f"_{args.seed}_{args.es}_{args.bs}_{args.lr}_{args.epoch}_{args.p_train}_{args.emb_epoch}_{args.dsc}/"

    if args.analysis:
        print("analysis mode")
        # assert args.delta < 0, args.delta
        max_delta = args.delta
        est_dict = {}
        exp_key = exp_key.replace(f"--delta {max_delta}", f"--delta -{max_delta}")
        res = conn.get(f"anal:{exp_key}")
        if res is not None:
            print(res)
            if not args.overwrite:
                exit()
        for args.delta in range(max_delta+1):
            # options
            model_name_prefix = qs_prefix + \
                f"_{args.seed}_{args.es}_{args.bs}_{args.lr}_{args.epoch}_{args.p_train}_{args.emb_epoch}_{args.dsc}"
            save_prefix = model_name_prefix.replace("qs_", "")
            log_dir = f"log/{dname}/{save_prefix}/"
            ts_path = log_dir + f"analysis_ts_{args.delta}.csv"
            qs_path = log_dir + f"analysis_qs.csv"
            # print(args)
            # print(exp_key)
            # print(model_name_prefix)
            # print(save_prefix)
            # print(log_dir)
            # print(ts_path)
            # print(qs_path)

            # estimation result
            print("ts path:", ts_path)
            print("qs path:", qs_path)
            df = pd.read_csv(ts_path)
            # print(df[:5])
            # print(df['s_q'][:5])

            # test queries
            df_qry = pd.read_csv(qs_path)
            qry_list = df_qry['0'].tolist()
            qry_set = set(qry_list)

            for i, line in df.iterrows():
                if line.s_q in qry_set:
                    est_dict[(line.s_q, line.d)] = line.to_list()
            exp_key_delta = exp_key.replace(f"--delta -{max_delta}", f"--delta {args.delta}")
            print(f"delta {args.delta}:", conn.get(f"est:{exp_key_delta}"))
        print(f"delta -{max_delta}:", conn.get(f"est:{exp_key}"))

        est_lines = []
        for qry in qry_list:
            for delta in range(max_delta + 1):
                est_lines.append(est_dict[(qry, delta)])
        print(est_lines[:5])
        df_selected = pd.DataFrame(est_lines, columns=df.columns)
        print(df_selected[:5])
        y = df_selected['true_count']
        y_hat = df_selected['est_count']
        q_errors = misc_utils.compute_qerrors_wo_unnorm(y_hat, y)
        df_selected['q_error'] = q_errors
        df_selected = df_selected.sort_values('q_error', ascending=False)
        print(df_selected[:10])

        res_dict = {}
        q_err_avg = np.mean(q_errors)
        res_dict['err'] = q_err_avg
        # q_list = [x / 10 for x in range(11)]
        # q_list.extend([0.25, 0.75, 0.95, 0.99])
        q_list = [x / 100 for x in range(101)]
        for q in q_list:
            q_err_agg = np.quantile(q_errors, q)
            res_dict[f"q{int(q*100):d}"] = q_err_agg
        print(res_dict)
        exp_json_str = ut.get_model_exp_json_str("astrid", **res_dict)
        conn.set(f"anal:{exp_key}", exp_json_str)
        conn.set(f"est:{exp_key}", exp_json_str)
        print("[anal]", conn.get(f"anal:{exp_key}"))
        print("[est ]", conn.get(f"est:{exp_key}"))
        exit()

    conn.set(f"conf:{exp_key}", model_name_prefix)

    args_dict_str = json.dumps(args.__dict__)
    # print("args:", args)
    # print("args_str:", args_dict_str)
    conn.set(f"args:{exp_key}", args_dict_str)

    # print("dname:", dname)
    # print("path:", path)
    os.makedirs(path, exist_ok=True)
    # print("total_card_load_prefix:", total_card_load_prefix)
    # print("total_qs_load_prefix:", total_qs_load_prefix)
    # print("train_qs_load_prefix:", train_qs_load_prefix)
    # print("triplet_prefix:", train_triplet_prefix)
    # print("total_file_name_prefix:", total_file_name_prefix)
    # print("train_file_name_prefix:", train_file_name_prefix)
    # print("file_name_prefix:", train_file_name_prefix)
    # print("model_name_prefix:", model_name_prefix)
    # print("delta:", args.delta)
    # print("p_train:", args.p_train)
    # print("seed:", args.seed)
    # print("overwrite:", args.overwrite)
    assert os.path.exists(f"data/{total_card_load_prefix}/p_pref.csv")
    assert os.path.exists(f"data/{total_qs_load_prefix}.txt")
    assert os.path.exists(f"data/{train_qs_load_prefix}.txt")

    copyfile(f"data/{total_card_load_prefix}/p_pref.csv", path +
             total_qs_load_prefix + "_counts.csv")  # total cardinality
    copyfile(f"data/{total_qs_load_prefix}.txt", path + total_qs_load_prefix + ".csv")  # total query string
    # copyfile(f"data/{train_prefix}.txt", path + train_prefix + ".csv")  # train query string
    save_train_query_strings_including_valid_data(path, total_qs_load_prefix, train_qs_load_prefix)

    prepare_datasets.prepare_dataset(path, total_qs_load_prefix, total_qs_load_prefix,
                                     total_file_name_prefix, train_triplet_prefix, args.delta)
    prepare_datasets.prepare_dataset(path, train_qs_load_prefix, total_qs_load_prefix,
                                     train_file_name_prefix, train_triplet_prefix, args.delta)
    # print(path + train_qs_load_prefix + ".csv", path + train_file_name_prefix + ".csv")
    # print(path + total_qs_load_prefix + ".csv", path + total_file_name_prefix + ".csv")
    copyfile(path + train_qs_load_prefix + ".csv", path + train_file_name_prefix + ".csv")  # train query string
    copyfile(path + total_qs_load_prefix + ".csv", path + total_file_name_prefix + ".csv")  # total query string

    # if args.analysis_latency:
    #     print("analysis mode")
    # else:
    #     print("normal mode")
    main()
