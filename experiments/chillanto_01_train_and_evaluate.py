from pathlib import Path
import torch
import torch.utils.data as data
import numpy as np

from src.settings import CHILLANTO_DATA_FOLDER, CHILLANTO_LOGGING_FOLDER, CHILLANTO_MODELS_FOLDER
from pathlib import Path
from src.datasets.chillanto import ChillantoDataset

from collections import ChainMap

from src.tasks.train_and_evaluate import task_train_and_evaluate, task_config, setup_task

def get_sampler(train_set, config):
    # TODO fix this turkey
    class_prob = [0, 0, 0.76, 0.24]
    sample_weights = []

    for i in range(len(train_set)):
        _, label = train_set[i]
        sample_weights.append(1 / class_prob[label])

    sample_weights = torch.Tensor(sample_weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(train_set))

    return sampler


def build_data_loaders(config):
    train_set, dev_set, test_set = ChillantoDataset.splits(config)

    print("training set: ", len(train_set))
    print("dev set", len(dev_set))
    print("test set", len(test_set))

    sampler = get_sampler(train_set, config)
    train= data.DataLoader(train_set, batch_size=config["batch_size"], shuffle=False, drop_last=True, sampler=sampler)
    dev= data.DataLoader(dev_set, batch_size=16)
    test= data.DataLoader(test_set, batch_size=16)
    return train, dev, test

def build_config():
    config = task_config({
            'project': 'chillanto_train_and_evaluate',
            'model_path': CHILLANTO_MODELS_FOLDER / 'chill',
            'log_file_path': CHILLANTO_LOGGING_FOLDER /  'logs.pkl',
            'predictions_path': CHILLANTO_LOGGING_FOLDER / 'predictions.pkl',
            "data_folder": CHILLANTO_DATA_FOLDER,
            'print_confusion_matrix': True,
            'lr': [0.001, 0.0001],
            'weight_decay': 0.00001,
            'momentum': 0.9,
            'schedule': [500],
            'n_epochs': 50,
            'n_labels': 4,
            'silence_prob': 0.0,
            'noise_prob': 0.0,
            'unknown_prob': 0.0,
            'input_length': 8000,
            'wanted_words': ['normal8k', 'asphyxia8k'],
            'batch_size': 50,
            'cache_size': 0,
            "dev_pct": 5,
            "test_pct": 40,
            "sampling_freq": 8000,
            'cache_size':32768,
            })

    # Merge together the model, training and dataset configuration:
    return dict(ChainMap(ChillantoDataset.default_config(config), config))


def train_and_evaluate():
    config = build_config()
    data_loaders = build_data_loaders(config)
    params = setup_task(config, data_loaders, 4)
    task_train_and_evaluate(params)
