from pathlib import Path
import numpy as np
import torch
import torch.utils.data as data

from src.settings import AUDIOSET_DATA_FOLDER, AUDIOSET_MODELS_FOLDER, AUDIOSET_LOGGING_FOLDER
from src.datasets.audioset import AudioSetDataset
from src.tasks.train_and_evaluate import task_train_and_evaluate, task_config, setup_task
from src.training_helpers import set_seed

def audioset_sampler(distribution, total, labels, train_set):
    weights = np.zeros(len(labels))
    for (lbl, idx) in labels.items():
        if lbl in distribution:
            weights[idx] = distribution[lbl] / total
    return torch.utils.data.sampler.WeightedRandomSampler(weights, len(train_set))

def build_data_loaders(config, splits):
    train_set, dev_set, test_set = splits['datasets']
    print("training set: ", len(train_set))
    print("dev set", len(dev_set))
    print("test set", len(test_set))
    sampler = audioset_sampler(splits['distribution'], splits['total'], splits['labels'], train_set)
#    train= data.DataLoader(train_set, batch_size=config["batch_size"], shuffle=False,
#            num_workers=6,
#            sampler=sampler)
    train= data.DataLoader(train_set,  batch_size=config["batch_size"], shuffle=True, drop_last=True)
    dev= data.DataLoader(dev_set,  batch_size=min(len(dev_set),
        64), shuffle=True)
    test= data.DataLoader(test_set, batch_size=min(len(test_set),
        64), shuffle=True)
    return train, dev, test

def build_config():
    config = task_config({
            'project': 'audioset1_train_and_evaluate',
            'model_path': AUDIOSET_MODELS_FOLDER ,
            'log_file_path': AUDIOSET_LOGGING_FOLDER ,
            'predictions_path': AUDIOSET_LOGGING_FOLDER ,
            'data_folder': AUDIOSET_DATA_FOLDER,
            'print_confusion_matrix': False,
            'n_epochs': 30,
            'lr': [0.001, 0.0001],
            'schedule': [15000],
            'batch_size': 64,
            'model_class': 'res8',
            'weight_decay': 0.000001,
            'momentum': 0.9,
            'loss': 'hinge',
            'seed': 9
            })
    # Merge together the model, training and dataset configuration:
    return AudioSetDataset.default_config(config)


def train_and_evaluate():
    config = build_config()
    set_seed(config)
    splits = AudioSetDataset.splits(config)

    config['n_labels'] = splits['n_labels']
    data_loaders = build_data_loaders(config, splits)
    params = setup_task(config, data_loaders, config['n_labels'])
    task_train_and_evaluate(params)
