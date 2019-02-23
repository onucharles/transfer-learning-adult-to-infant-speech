from collections import ChainMap
import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
import torch

from src.comet_logger import CometLogger
from src.settings import MODEL_CLASS
from src.model import find_model
from src.trainer_and_evaluator import TrainerAndEvaluator

def build_data_loaders(config, dataset_class, sampler_func):
    train_set, dev_set, test_set = dataset_class.splits(config)

    print("training set: ", len(train_set))
    print("dev set", len(dev_set))
    print("test set", len(test_set))

    sampler = sampler_func(train_set, config)
    train= data.DataLoader(train_set, num_workers=4, batch_size=config["batch_size"], shuffle=False, drop_last=True, sampler=sampler)
    dev= data.DataLoader(dev_set, num_workers=4, batch_size=1000)
    test= data.DataLoader(test_set,  num_workers=4,batch_size=1000)
    return train, dev, test


def task_config(custom_config={}):
    """ Reasonable defaults for the training task """
    default_config = {
        'n_epochs': 1,
        'lr': [0.1, 0.01, 0.001],
        'schedule': [0, 30000, 60000],
        'batch_size': 128,
        'weight_decay': 0.00001,
        'dev_every': 1,
        'use_nesterov': False,
        'weight_decay': False,
        'momentum': False,
        'seed': 1,
        'model_class': MODEL_CLASS,
        'print_confusion_matrix': False,
    }
    return dict(ChainMap(custom_config, default_config))

def setup_task(config, data_loaders, n_labels):
    """ returns all the objects (including data loaders) for training """
    logger = CometLogger(project=config['project'])
    experiment = logger.experiment()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = find_model(config['model_class'], n_labels)
    return {
        'experiment': experiment,
        'model': model,
        'device': device,
        'config': config,
        'loaders': data_loaders
    }


def task_train_and_evaluate(task_params):
    """
    Task details:
        - Performs a training and evaluation loop.
        - Evaluates model performance using F1 and Confusion Matrix
        - Logs results to comet experiment
        - Outputs the best model candidate
    """
    TrainerAndEvaluator(task_params).perform_training()
