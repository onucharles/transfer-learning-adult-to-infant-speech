from collections import Counter
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data

from .utils import evalutils

import random

from sklearn import metrics

def compute_eval(scores, labels):
    batch_size = labels.size(0)
    predictions = torch.max(scores, 1)[1].view(batch_size).data

    metric_value = (predictions == labels.data).float().sum() / batch_size
    return metric_value

def confusion_matrix(scores, labels, classes=None):
    batch_size = labels.size(0)
    predictions = torch.max(scores, 1)[1].view(batch_size).data
    cm = metrics.confusion_matrix(labels.data, predictions, labels=classes)
    return cm

def print_eval(name, eval_score, loss, lr,  end="\n"):
    print("{0}: {1}, loss: {2}, learning_rate: {3}".format(name, eval_score, loss.item(), lr))

def print_test_eval(name, avg_acc, conf_mat, compute_f1=False):
    if compute_f1:
        tp, tn, fp, fn, p, n = evalutils.read_conf_matrix(conf_mat, pos_class=3)
        f1, precision, recall = evalutils.f1_prec_recall(tp, tn, fp, fn, p, n)
        print("Confusion matrix: {}".format(conf_mat))
        print("{} accuracy: {}\tF1 = {}\tPrecision={}\tRecall={}".format(name, avg_acc, f1, precision, recall))
    else:
        print("{} accuracy: {}".format(name, avg_acc))


def set_seed(config):
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_loss_fxn(config):
    # NOTE: @charles this seems to be what we're using in evaluate. assuming we
    # use it for training too?
    criterion = nn.CrossEntropyLoss()
    return criterion
    #if config['loss'] == 'crossent':
    #    criterion = nn.CrossEntropyLoss()
    #elif config['loss'] == 'hinge':
    #    criterion = nn.MultiMarginLoss()
    #else:
    #    raise ValueError("Unsupported loss function '{0}' was specified".format(config['loss']))
    #return criterion

def get_sampler(train_set, config):
    # TODO need to also add support in 'config' for sampler, such that no sampler is also valid.

    # TODO  ideally get class prob from train_set
    # class_prob = [0, 0, 0.76, 0.24]
    # TODO fix this turkey
    #sample_weights = []

    sample_weights = np.zeros(len(train_set)) + (1 / config['n_labels'])
    #for i in range(len(train_set)):
    #    _, label = train_set[i]
    #    sample_weights.append(1 / class_prob[label])

    sample_weights = torch.tensor(sample_weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(train_set))

    return sampler

# TODO: log experiment results
def evaluate(n_labels, compute_f1, model, device, test_loader):
    classes = np.arange(n_labels)
#    model.to(device)
#    model.eval()
    criterion = nn.CrossEntropyLoss()
    results = []
    total = 0
    conf_mat = np.zeros((n_labels,n_labels))
    prediction_log = []
    for model_in, labels in test_loader:
        model_in = model_in.to(device)
        labels = labels.to(device)
        scores = model(model_in.clone().detach())
        loss = criterion(scores, labels)
        results.append((compute_eval(scores, labels) * model_in.size(0)).detach().cpu())
        total += model_in.cpu().size(0)
        conf_mat += confusion_matrix(scores.detach().cpu(), labels.detach().cpu(), np.arange(n_labels))
        prediction_log.append((scores, labels))
    acc = sum(results) / total
    print_test_eval("Testing", acc, conf_mat, compute_f1)
    return prediction_log
