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
    sample_weights = torch.tensor(np.zeros(len(train_set)) + (1 / config['n_labels']))
    return torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(train_set))

def build_data_loaders(config):
    train_set, dev_set, test_set = ChillantoDataset.splits(config)

    print("training set: ", len(train_set))
    print("dev set", len(dev_set))
    print("test set", len(test_set))

    sampler = get_sampler(train_set, config)
    train= data.DataLoader(train_set, batch_size=config["batch_size"], shuffle=False, drop_last=True, sampler=sampler)
    dev= data.DataLoader(dev_set, batch_size=min(len(dev_set), 16), shuffle=True)
    test= data.DataLoader(test_set, batch_size=min(len(test_set), 16), shuffle=True)
    return train, dev, test

def build_config():
    config = task_config({
            'project': 'chillanto_train_and_evaluate',
            'model_path': CHILLANTO_MODELS_FOLDER / 'chill',
            'log_file_path': CHILLANTO_LOGGING_FOLDER /  'logs.pkl',
            'predictions_path': CHILLANTO_LOGGING_FOLDER / 'predictions.pkl',
            'print_confusion_matrix': True,
            'n_epochs': 1,
            'batch_size': 64
            })

    # Merge together the model, training and dataset configuration:
    return dict(ChainMap(ChillantoDataset.default_config({"data_folder": CHILLANTO_DATA_FOLDER}), config))


def train_and_evaluate():
    config = build_config()
    data_loaders = build_data_loaders(config)
    params = setup_task(config, data_loaders, 4)
    task_train_and_evaluate(params)
