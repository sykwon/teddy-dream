from collections import namedtuple
import numpy as np
import random
import torch
from string_dataset_helpers import StringDatasetHelper

# Instead of using argparse, we set all configs in a namedtuple

# This namedtuple is for the embedding learner
# embedding_dimension: must match SelectivityEstimatorConfigs.embedding_dimension
# margin: for triplet learning.
# channel_size: how many channels of information needed for Conv1D function. 8 is a good value
AstridEmbedLearnerConfigs = namedtuple('AstridEmbedLearnerConfigs',
                                       ['embedding_dimension', 'batch_size', 'num_epochs', 'margin', 'device', 'lr',
                                        'channel_size'])

# StringFrequencyConfigs contains the info needed for extracting vocabulary and training embedding model.
# string_list_file_name => list of strings one in each line.
#   This is used to get the alphabets, find max size of strings etc
# selectivity_file_name => csv file with two columns: string, selectivity
# triplets_file_name => each row contains (anchor, positive, negative) triplets
StringFrequencyConfigs = namedtuple('StringFrequencyConfigs',
                                    ['string_list_file_name', 'train_string_list_file_name', 'selectivity_file_name', 'triplets_file_name'])

# configuration for selectivity estimator.
# min_val and max_val are for unnormalizing the selectivities.
# it will be set in the main function
SelectivityEstimatorConfigs = namedtuple('SelectivityEstimatorConfigs',
                                         ['embedding_dimension', 'decoder_scale', 'batch_size', 'num_epochs', 'device', 'lr',
                                          'min_val', 'max_val',
                                          'embedding_model_file_name', 'selectivity_model_file_name'])


def initialize_random_seeds(seed_val=1234):
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)
    random.seed(seed_val)


def setup_vocabulary(string_list_file_name):
    string_helper = StringDatasetHelper()
    string_helper.extract_alphabets(string_list_file_name)
    return string_helper

# Copied from learned cardinalities paper


def normalize_labels(labels, min_val=None, max_val=None):
    labels = np.array([np.log(float(max(1, l))) for l in labels])
    if min_val is None:
        min_val = labels.min()
    if max_val is None:
        max_val = labels.max()
    labels_norm = (labels - min_val) / (max_val - min_val)
    # Threshold labels
    labels_norm = np.minimum(labels_norm, 1)
    labels_norm = np.maximum(labels_norm, 0)
    return labels_norm, min_val, max_val


def get_number_of_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# copied from learned cardinalities paper
def unnormalize_torch(vals, min_val, max_val):
    vals = (vals * (max_val - min_val)) + min_val
    return torch.exp(vals)


# Copied from learned cardinalities paper
def qerror_loss_bak(preds=None, targets=None, min_val=0.0, max_val=1.0):
    qerror = []
    preds = unnormalize_torch(preds, min_val, max_val)
    targets = unnormalize_torch(targets, min_val, max_val)

    for i in range(len(targets)):
        if (preds[i] > targets[i]).cpu().data.numpy()[0]:
            qerror.append(preds[i] / targets[i])
        else:
            qerror.append(targets[i] / preds[i])
    return torch.mean(torch.cat(qerror))


# Copied from learned cardinalities paper
def qerror_loss(preds=None, targets=None, min_val=0.0, max_val=1.0):
    qerror = []
    preds = unnormalize_torch(preds, min_val, max_val)
    targets = unnormalize_torch(targets, min_val, max_val)
    preds = torch.clip(preds, min=1.0)
    targets = torch.clip(targets, min=1.0)

    for i in range(len(targets)):
        if (preds[i] > targets[i]).cpu().data.numpy()[0]:
            qerror.append(preds[i] / targets[i])
        else:
            qerror.append(targets[i] / preds[i])
    return torch.mean(torch.cat(qerror))

# Same as qerror_loss but returns the computed values


def compute_qerrors(preds=None, targets=None, min_val=0.0, max_val=1.0):
    qerror = []
    preds = unnormalize_torch(preds, min_val, max_val)
    targets = unnormalize_torch(targets, min_val, max_val)
    assert len(targets) == len(preds)
    # print("targets:", targets)
    # print("preds  :", preds)

    for i in range(len(targets)):
        if preds[i] < 1.0:
            preds[i] = 1.0
        if targets[i] < 1.0:
            targets[i] = 1.0
        if (preds[i] > targets[i]).cpu().data.numpy():
            qerror.append(preds[i] / targets[i])
        else:
            qerror.append(targets[i] / preds[i])
    return qerror


def compute_qerrors_wo_unnorm(preds, targets):
    qerror = []
    targets = targets.copy()
    for i in range(len(targets)):
        if preds[i] < 1.0:
            preds[i] = 1.0
        if targets[i] < 1.0:
            targets[i] = 1.0
        if (preds[i] > targets[i]):
            qerror.append(preds[i] / targets[i])
        else:
            qerror.append(targets[i] / preds[i])
    return qerror
