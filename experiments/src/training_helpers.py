from collections import Counter
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data

from .comet_logger import CometLogger
from .utils import evalutils
from .model import find_model

from collections import ChainMap
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


def print_f1_confusion_matrix(name, avg_acc, conf_mat):
    tp, tn, fp, fn, p, n = evalutils.read_conf_matrix(conf_mat, pos_class=3)
    f1, precision, recall = evalutils.f1_prec_recall(tp, tn, fp, fn, p, n)
    print("Confusion matrix: \n {}".format(conf_mat))
    print("{} accuracy: {}\tF1 = {}\tPrecision={}\tRecall={}".format(name, avg_acc, f1, precision, recall))


def set_seed(config):
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# TODO: log experiment results
def evaluate(n_labels, model, device, test_loader, print_confusion_matrix):
    print("evaluate")
    classes = np.arange(n_labels)
    criterion = nn.CrossEntropyLoss()
    results = []
    total = 0
    conf_mat = np.zeros((n_labels,n_labels))
    prediction_log = []
    with torch.no_grad():
        for model_in, labels in test_loader:
            model_in = model_in.to(device)
            labels = labels.to(device)
            scores = model(model_in.clone().detach())
            loss = criterion(scores, labels)
            model_in_size = model_in.size(0)            
            results.append(compute_eval(scores, labels) * model_in_size)
            #total += model_in.cpu().size(0)
            if print_confusion_matrix:
                conf_mat += confusion_matrix(scores.detach().cpu(), labels.detach().cpu(), np.arange(n_labels))
            print("appending", model_in_size)
            prediction_log.append((scores, labels))
    acc = sum(results) / len(test_loader)
    print("acc", acc)
    if print_confusion_matrix:
        print_f1_confusion_matrix("Testing", acc, conf_mat)
    return prediction_log
